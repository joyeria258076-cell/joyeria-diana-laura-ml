import os
import psycopg2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

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

        modelo_rf, scaler, columnas_X, mae = entrenar_modelo_precio(df)

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
