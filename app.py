import os
import psycopg2
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

def get_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def obtener_tickets():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT v.id AS venta_id, dv.producto_nombre
        FROM ventas v
        JOIN detalle_ventas dv ON dv.venta_id = v.id
        ORDER BY v.id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    tickets = {}
    for venta_id, producto in rows:
        if venta_id not in tickets:
            tickets[venta_id] = []
        tickets[venta_id].append(producto)

    return list(tickets.values())

def entrenar_modelo(tickets):
    te = TransactionEncoder()
    te_array = te.fit(tickets).transform(tickets)
    df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

    if frequent_itemsets.empty:
        return None

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1, num_itemsets=len(frequent_itemsets))
    return rules

@app.route('/health', methods=['GET'])
def health():
    return jsonify({ "status": "ok", "servicio": "recomendacion" })

@app.route('/recomendar', methods=['POST'])
def recomendar():
    data = request.get_json()
    productos_carrito = data.get('productos', [])

    if not productos_carrito:
        return jsonify({ "error": "Debes enviar al menos un producto" }), 400

    try:
        tickets = obtener_tickets()

        if len(tickets) < 5:
            return jsonify({ "recomendaciones": [], "mensaje": "Datos insuficientes para generar recomendaciones" })

        rules = entrenar_modelo(tickets)

        if rules is None or rules.empty:
            return jsonify({ "recomendaciones": [], "mensaje": "No se encontraron patrones suficientes" })

        recomendaciones = set()
        for producto in productos_carrito:
            filtro = rules[rules['antecedents'].apply(lambda x: producto in x)]
            for _, row in filtro.iterrows():
                for rec in row['consequents']:
                    if rec not in productos_carrito:
                        recomendaciones.add(rec)

        return jsonify({
            "recomendaciones": list(recomendaciones),
            "basado_en": productos_carrito,
            "total_tickets_analizados": len(tickets)
        })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

@app.route('/reglas', methods=['GET'])
def ver_reglas():
    try:
        tickets = obtener_tickets()
        rules = entrenar_modelo(tickets)

        if rules is None or rules.empty:
            return jsonify({ "reglas": [], "mensaje": "Sin patrones encontrados" })

        resultado = []
        for _, row in rules.iterrows():
            resultado.append({
                "si_compra": list(row['antecedents']),
                "tambien_lleva": list(row['consequents']),
                "soporte": round(float(row['support']), 3),
                "confianza": round(float(row['confidence']), 3),
                "lift": round(float(row['lift']), 3)
            })

        return jsonify({ "reglas": resultado, "total": len(resultado) })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    app.run(debug=True, port=port)
