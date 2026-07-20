import os
import psycopg2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import time

load_dotenv()

app = Flask(__name__)
CORS(app)

# Cache en memoria del modelo de precio (entrenar Random Forest + validacion cruzada
# es costoso: 6 entrenamientos por solicitud). Se reutiliza el modelo por 5 minutos
# o hasta que cambie la cantidad de productos activos, para que solo la primera
# solicitud pague el costo completo y las siguientes respondan casi al instante.
_cache_modelo_precio = { "timestamp": 0, "total_productos": None, "modelo_rf": None, "scaler": None, "columnas_X": None, "mae": None }
CACHE_TTL_SEGUNDOS = 300

# Cache del resultado de segmentacion (K-Means). Se recalcula si cambia el numero
# de clientes o pasan mas de CACHE_TTL_SEGUNDOS.
_cache_segmentacion = { "timestamp": 0, "total_clientes": None, "resultado": None }

def get_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def obtener_productos():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            p.id,
            p.nombre,
            p.material_principal,
            c.nombre AS categoria,
            p.peso_gramos,
            p.dias_fabricacion,
            p.permite_personalizacion,
            p.imagen_principal,
            p.precio_venta
        FROM productos p
        JOIN categorias c ON p.categoria_id = c.id
        WHERE p.activo = true
        ORDER BY p.id
    """)
    columnas = ['id', 'nombre', 'material_principal', 'categoria', 'peso_gramos', 'dias_fabricacion', 'permite_personalizacion', 'imagen_principal', 'precio_venta']
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return pd.DataFrame(rows, columns=columnas)

def calcular_similitud(df):
    df_features = df.copy()

    # One-Hot Encoding para variables categóricas SIN orden real (material, categoría).
    # LabelEncoder no es correcto aquí: asignaría números arbitrarios (ej. Plata=4, Oro=2)
    # que el algoritmo interpretaría como si tuvieran una relación de orden/distancia real.
    dummies_material = pd.get_dummies(df_features['material_principal'], prefix='material')
    dummies_categoria = pd.get_dummies(df_features['categoria'], prefix='categoria')

    df_features['personalizacion_enc'] = df['permite_personalizacion'].astype(int)

    # Estandarización de variables numéricas (media 0, desviación 1).
    # Sin esto, peso_gramos (rango ~1 a 12) dominaría el cálculo frente a las columnas
    # binarias de One-Hot (rango 0 a 1), sesgando la similitud coseno.
    numericas = df_features[['peso_gramos', 'dias_fabricacion']]
    numericas_escaladas = pd.DataFrame(
        StandardScaler().fit_transform(numericas),
        columns=['peso_gramos_esc', 'dias_fabricacion_esc'],
        index=df_features.index
    )

    X = pd.concat([
        dummies_material,
        dummies_categoria,
        numericas_escaladas,
        df_features['personalizacion_enc']
    ], axis=1).values

    return cosine_similarity(X)

def productos_similares(producto_id, df, sim_matrix, excluir_nombres, top_n=3):
    indices = df[df['id'] == producto_id].index
    if len(indices) == 0:
        return []
    idx = indices[0]

    similitudes = list(enumerate(sim_matrix[idx]))
    similitudes = sorted(similitudes, key=lambda x: x[1], reverse=True)
    similitudes = [s for s in similitudes if s[0] != idx]

    resultado = []
    for i, score in similitudes:
        nombre = df.iloc[i]['nombre']
        if nombre in excluir_nombres:
            continue
        resultado.append({
            "id": int(df.iloc[i]['id']),
            "nombre": nombre,
            "imagen_principal": df.iloc[i]['imagen_principal'],
            "precio_venta": float(df.iloc[i]['precio_venta']) if df.iloc[i]['precio_venta'] is not None else None,
            "similitud": round(float(score), 4)
        })
        if len(resultado) >= top_n:
            break
    return resultado

def entrenar_modelo_precio(df):
    """
    Entrena un RandomForestRegressor sobre el catalogo activo (precio_venta como Y).
    Mismo criterio de preprocesamiento que calcular_similitud: One-Hot para variables
    categoricas sin orden real, StandardScaler para variables numericas.
    """
    df_features = df.copy()

    dummies_material = pd.get_dummies(df_features['material_principal'], prefix='material')
    dummies_categoria = pd.get_dummies(df_features['categoria'], prefix='categoria')
    df_features['personalizacion_enc'] = df['permite_personalizacion'].astype(int)

    numericas = df_features[['peso_gramos', 'dias_fabricacion']]
    scaler = StandardScaler()
    numericas_escaladas = pd.DataFrame(
        scaler.fit_transform(numericas),
        columns=['peso_gramos_esc', 'dias_fabricacion_esc'],
        index=df_features.index
    )

    X = pd.concat([dummies_material, dummies_categoria, numericas_escaladas, df_features['personalizacion_enc']], axis=1)
    y = df['precio_venta'].astype(float)

    modelo_rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    modelo_rf.fit(X, y)

    # MAE por validacion cruzada, usado como margen del rango sugerido (mas robusto que un solo split)
    n_folds = min(5, len(X))
    mae = None
    if n_folds >= 2:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        mae_scores = -cross_val_score(modelo_rf, X, y, cv=kf, scoring='neg_mean_absolute_error')
        mae = float(mae_scores.mean())

    return modelo_rf, scaler, list(X.columns), mae

def predecir_precio_producto(modelo_rf, scaler, columnas_X, material, categoria, peso_gramos, dias_fabricacion, permite_personalizacion):
    fila = pd.DataFrame([{
        'material_principal': material,
        'categoria': categoria,
        'peso_gramos': peso_gramos,
        'dias_fabricacion': dias_fabricacion,
        'permite_personalizacion': permite_personalizacion
    }])

    dummies_mat = pd.get_dummies(fila['material_principal'], prefix='material')
    dummies_cat = pd.get_dummies(fila['categoria'], prefix='categoria')
    fila['personalizacion_enc'] = fila['permite_personalizacion'].astype(int)

    numericas_fila = scaler.transform(fila[['peso_gramos', 'dias_fabricacion']])
    numericas_fila = pd.DataFrame(numericas_fila, columns=['peso_gramos_esc', 'dias_fabricacion_esc'])

    fila_X = pd.concat([dummies_mat, dummies_cat, numericas_fila, fila['personalizacion_enc']], axis=1)
    fila_X = fila_X.reindex(columns=columnas_X, fill_value=0)

    return float(modelo_rf.predict(fila_X)[0])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({ "status": "ok", "servicio": "recomendacion", "algoritmo": "content-based-filtering-coseno" })

@app.route('/recomendar', methods=['POST'])
def recomendar():
    data = request.get_json()
    productos_carrito = data.get('productos', [])

    if not productos_carrito:
        return jsonify({ "error": "Debes enviar al menos un producto" }), 400

    try:
        df = obtener_productos()

        if len(df) < 2:
            return jsonify({ "recomendaciones": [], "mensaje": "Catálogo insuficiente para generar recomendaciones" })

        sim_matrix = calcular_similitud(df)

        # IDs de los productos que ya están en el carrito (para no recomendarlos de vuelta)
        ids_carrito = df[df['nombre'].isin(productos_carrito)]['id'].tolist()
        if not ids_carrito:
            return jsonify({ "recomendaciones": [], "mensaje": "Ninguno de los productos del carrito está en el catálogo activo" })

        candidatos = {}
        for producto_id in ids_carrito:
            similares = productos_similares(producto_id, df, sim_matrix, excluir_nombres=set(productos_carrito), top_n=3)
            for item in similares:
                actual = candidatos.get(item['nombre'])
                if actual is None or item['similitud'] > actual['similitud']:
                    candidatos[item['nombre']] = item

        resultado = sorted(candidatos.values(), key=lambda x: x['similitud'], reverse=True)[:5]
        resultado = [{
            "id": r["id"],
            "nombre": r["nombre"],
            "imagen_principal": r["imagen_principal"],
            "precio_venta": r["precio_venta"]
        } for r in resultado]

        return jsonify({
            "recomendaciones": resultado,
            "basado_en": productos_carrito,
            "total_productos_catalogo": len(df)
        })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

def obtener_modelo_precio_cacheado(df):
    """
    Reutiliza el modelo entrenado si tiene menos de CACHE_TTL_SEGUNDOS y el catalogo
    no cambio de tamano. Entrenar (5-fold cross-validation + fit final) es costoso;
    sin cache, cada solicitud tardaria ~10s en el plan gratuito de Render.
    """
    ahora = time.time()
    cache_valida = (
        _cache_modelo_precio["modelo_rf"] is not None
        and (ahora - _cache_modelo_precio["timestamp"]) < CACHE_TTL_SEGUNDOS
        and _cache_modelo_precio["total_productos"] == len(df)
    )

    if cache_valida:
        return (
            _cache_modelo_precio["modelo_rf"],
            _cache_modelo_precio["scaler"],
            _cache_modelo_precio["columnas_X"],
            _cache_modelo_precio["mae"],
        )

    modelo_rf, scaler, columnas_X, mae = entrenar_modelo_precio(df)
    _cache_modelo_precio.update({
        "timestamp": ahora,
        "total_productos": len(df),
        "modelo_rf": modelo_rf,
        "scaler": scaler,
        "columnas_X": columnas_X,
        "mae": mae,
    })
    return modelo_rf, scaler, columnas_X, mae

@app.route('/predecir-precio', methods=['POST'])
def predecir_precio():
    data = request.get_json() or {}

    material_principal = data.get('material_principal')
    categoria_nombre = data.get('categoria_nombre')
    peso_gramos = data.get('peso_gramos')
    dias_fabricacion = data.get('dias_fabricacion', 0)
    permite_personalizacion = data.get('permite_personalizacion', False)

    if not material_principal or not categoria_nombre or peso_gramos is None:
        return jsonify({ "error": "Faltan campos: material_principal, categoria_nombre y peso_gramos son obligatorios" }), 400

    try:
        df = obtener_productos()

        if len(df) < 5:
            return jsonify({ "error": "Catálogo insuficiente para entrenar el modelo de precio" }), 400

        modelo_rf, scaler, columnas_X, mae = obtener_modelo_precio_cacheado(df)

        precio_sugerido = predecir_precio_producto(
            modelo_rf, scaler, columnas_X,
            material_principal, categoria_nombre,
            float(peso_gramos), float(dias_fabricacion), bool(permite_personalizacion)
        )

        margen = mae if mae is not None else precio_sugerido * 0.15
        rango_min = max(0, precio_sugerido - margen)
        rango_max = precio_sugerido + margen

        return jsonify({
            "precio_sugerido": round(precio_sugerido, 2),
            "rango_min": round(rango_min, 2),
            "rango_max": round(rango_max, 2),
            "margen_error_promedio": round(margen, 2),
            "total_productos_entrenamiento": len(df)
        })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

def obtener_clientes():
    """
    Perfil de comportamiento por cliente. Usa CTEs independientes por tabla
    (compras, apartados, puntualidad) en vez de un JOIN directo de las 3 tablas,
    para evitar el "fan-out" que inflaria AVG(ventas.total) en clientes con
    varios apartados.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        WITH compras AS (
            SELECT cliente_id, COUNT(*) AS num_compras, AVG(total) AS ticket_promedio
            FROM ventas
            GROUP BY cliente_id
        ),
        apartados_cli AS (
            SELECT cliente_id, COUNT(*) AS num_apartados, AVG(monto_total) AS monto_apartado_promedio
            FROM apartados
            GROUP BY cliente_id
        ),
        puntualidad AS (
            SELECT a.cliente_id,
                COALESCE(
                    COUNT(ab.id) FILTER (WHERE ab.estado = 'pagado' AND ab.fecha_abono <= ab.fecha_limite_siguiente)::numeric
                    / NULLIF(COUNT(ab.id), 0),
                    0.5
                ) AS puntualidad_pago
            FROM apartados a
            LEFT JOIN abonos ab ON ab.apartado_id = a.id
            GROUP BY a.cliente_id
        )
        SELECT
            c.id AS cliente_id,
            c.nombre,
            COALESCE(co.num_compras, 0) AS num_compras,
            COALESCE(co.ticket_promedio, 0) AS ticket_promedio,
            COALESCE(ap.num_apartados, 0) AS num_apartados,
            COALESCE(ap.monto_apartado_promedio, 0) AS monto_apartado_promedio,
            CASE WHEN COALESCE(ap.num_apartados, 0) > 0 THEN 1 ELSE 0 END AS usa_apartados,
            COALESCE(pu.puntualidad_pago, 0.5) AS puntualidad_pago
        FROM clientes c
        LEFT JOIN compras co ON co.cliente_id = c.id
        LEFT JOIN apartados_cli ap ON ap.cliente_id = c.id
        LEFT JOIN puntualidad pu ON pu.cliente_id = c.id
        ORDER BY c.id
    """)
    columnas = ['cliente_id', 'nombre', 'num_compras', 'ticket_promedio', 'num_apartados', 'monto_apartado_promedio', 'usa_apartados', 'puntualidad_pago']
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=columnas)

DESCRIPCIONES_SEGMENTO = {
    'Cliente Frecuente de Alto Gasto': 'Muchas compras, ticket alto',
    'Cliente Ocasional': 'Pocas compras, ticket bajo',
    'Cliente Apartador': 'Usa el sistema de apartados',
}
ACCIONES_SEGMENTO = {
    'Cliente Frecuente de Alto Gasto': 'Atencion personalizada y piezas exclusivas',
    'Cliente Ocasional': 'Campanas de reactivacion y promociones especiales',
    'Cliente Apartador': 'Recordatorio de saldo pendiente y oferta de nuevo apartado al liquidar',
}

def entrenar_segmentacion(df):
    """
    K-Means (k=3) sobre variables de comportamiento del cliente, todas estandarizadas.
    El nombre de cada cluster se asigna dinamicamente segun las caracteristicas de su
    centroide (no por indice fijo, ya que el orden de los clusters de K-Means es arbitrario).
    """
    variables = ['num_compras', 'ticket_promedio', 'num_apartados', 'monto_apartado_promedio', 'usa_apartados', 'puntualidad_pago']
    X = df[variables].values

    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = kmeans.fit_predict(X_escalado)

    centroides = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=variables)
    cluster_apartador = centroides['usa_apartados'].idxmax()
    restantes = [c for c in centroides.index if c != cluster_apartador]
    cluster_frecuente = centroides.loc[restantes, 'ticket_promedio'].idxmax()
    cluster_ocasional = [c for c in restantes if c != cluster_frecuente][0]

    nombres_segmento = {
        cluster_frecuente: 'Cliente Frecuente de Alto Gasto',
        cluster_ocasional: 'Cliente Ocasional',
        cluster_apartador: 'Cliente Apartador',
    }
    df['segmento'] = df['cluster'].map(nombres_segmento)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_escalado)
    df['pca_x'] = coords[:, 0]
    df['pca_y'] = coords[:, 1]

    return df

def obtener_segmentacion_cacheada():
    ahora = time.time()
    df = obtener_clientes()

    cache_valida = (
        _cache_segmentacion["resultado"] is not None
        and (ahora - _cache_segmentacion["timestamp"]) < CACHE_TTL_SEGUNDOS
        and _cache_segmentacion["total_clientes"] == len(df)
    )
    if cache_valida:
        return _cache_segmentacion["resultado"]

    df_segmentado = entrenar_segmentacion(df)
    _cache_segmentacion.update({
        "timestamp": ahora,
        "total_clientes": len(df),
        "resultado": df_segmentado,
    })
    return df_segmentado

@app.route('/segmentos', methods=['GET'])
def ver_segmentos():
    try:
        df = obtener_segmentacion_cacheada()

        if len(df) < 6:
            return jsonify({ "error": "Catalogo de clientes insuficiente para segmentar (minimo 6)" }), 400

        resumen = df['segmento'].value_counts().to_dict()
        segmentos = [{
            "nombre": nombre,
            "clientes": int(count),
            "descripcion": DESCRIPCIONES_SEGMENTO.get(nombre, ''),
            "accion": ACCIONES_SEGMENTO.get(nombre, ''),
        } for nombre, count in resumen.items()]

        clientes = [{
            "id": int(row['cliente_id']),
            "nombre": row['nombre'],
            "num_compras": int(row['num_compras']),
            "ticket_promedio": round(float(row['ticket_promedio']), 2),
            "num_apartados": int(row['num_apartados']),
            "monto_apartado_promedio": round(float(row['monto_apartado_promedio']), 2),
            "usa_apartados": bool(row['usa_apartados']),
            "segmento": row['segmento'],
            "pca_x": round(float(row['pca_x']), 3),
            "pca_y": round(float(row['pca_y']), 3),
        } for _, row in df.iterrows()]

        return jsonify({
            "algoritmo": "kmeans",
            "total_clientes": len(df),
            "segmentos": segmentos,
            "clientes": clientes,
        })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

@app.route('/precios-ejemplo', methods=['GET'])
def ver_precios_ejemplo():
    """
    Endpoint de verificación, abrible directo en el navegador (GET).
    Entrena el modelo de precio y muestra predicciones para combinaciones
    de ejemplo (una por cada material x categoria real del catalogo),
    sin necesidad de Postman.
    """
    try:
        df = obtener_productos()

        if len(df) < 5:
            return jsonify({ "resultado": [], "mensaje": "Catálogo insuficiente" })

        modelo_rf, scaler, columnas_X, mae = obtener_modelo_precio_cacheado(df)

        materiales = sorted(df['material_principal'].unique())
        categorias = sorted(df['categoria'].unique())

        resultado = []
        for material in materiales:
            for categoria in categorias:
                precio = predecir_precio_producto(
                    modelo_rf, scaler, columnas_X,
                    material, categoria, peso_gramos=5.0, dias_fabricacion=0, permite_personalizacion=False
                )
                resultado.append({
                    "material": material,
                    "categoria": categoria,
                    "peso_gramos_usado": 5.0,
                    "precio_sugerido": round(precio, 2)
                })

        return jsonify({
            "algoritmo": "random-forest-regressor",
            "total_productos_entrenamiento": len(df),
            "mae_validacion_cruzada": round(mae, 2) if mae is not None else None,
            "nota": "Ejemplos calculados con peso_gramos=5.0 fijo, para comparar el efecto de material y categoria",
            "resultado": resultado
        })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

@app.route('/similitudes', methods=['GET'])
def ver_similitudes():
    """
    Endpoint de verificación, abrible directo en el navegador (GET).
    Muestra, para cada producto activo del catálogo, sus 3 productos más similares
    según el modelo Content-Based Filtering (similitud coseno).
    """
    try:
        df = obtener_productos()

        if len(df) < 2:
            return jsonify({ "resultado": [], "mensaje": "Catálogo insuficiente" })

        sim_matrix = calcular_similitud(df)

        resultado = []
        for _, row in df.iterrows():
            similares = productos_similares(row['id'], df, sim_matrix, excluir_nombres=set(), top_n=3)
            resultado.append({
                "producto": row['nombre'],
                "material": row['material_principal'],
                "categoria": row['categoria'],
                "similares": similares
            })

        return jsonify({ "algoritmo": "content-based-filtering-coseno", "total_productos": len(df), "resultado": resultado })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    app.run(debug=True, port=port)
