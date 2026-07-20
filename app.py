import os
import psycopg2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
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
            p.permite_personalizacion
        FROM productos p
        JOIN categorias c ON p.categoria_id = c.id
        WHERE p.activo = true
        ORDER BY p.id
    """)
    columnas = ['id', 'nombre', 'material_principal', 'categoria', 'peso_gramos', 'dias_fabricacion', 'permite_personalizacion']
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return pd.DataFrame(rows, columns=columnas)

def calcular_similitud(df):
    df_features = df.copy()

    le_material = LabelEncoder()
    le_categoria = LabelEncoder()

    df_features['material_enc'] = le_material.fit_transform(df['material_principal'])
    df_features['categoria_enc'] = le_categoria.fit_transform(df['categoria'])
    df_features['personalizacion_enc'] = df['permite_personalizacion'].astype(int)

    feature_cols = ['material_enc', 'categoria_enc', 'peso_gramos', 'dias_fabricacion', 'personalizacion_enc']
    X = df_features[feature_cols].values

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
        resultado.append({ "id": int(df.iloc[i]['id']), "nombre": nombre, "similitud": round(float(score), 4) })
        if len(resultado) >= top_n:
            break
    return resultado

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
        resultado = [{ "id": r["id"], "nombre": r["nombre"] } for r in resultado]

        return jsonify({
            "recomendaciones": resultado,
            "basado_en": productos_carrito,
            "total_productos_catalogo": len(df)
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
