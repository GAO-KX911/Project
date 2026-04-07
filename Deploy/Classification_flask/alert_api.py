import sqlite3
from flask import Flask, jsonify, request
from datetime import datetime

DB_PATH = "/home/njust/Fire/Deploy/monitor.db"
app = Flask(__name__)

def query_alerts(limit=50, since=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    sql = "SELECT id, captured_at, camera_id, class_label, probability, image_path FROM alerts"
    params = []
    if since:
        sql += " WHERE captured_at >= ?"
        params.append(since)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    cur.execute(sql, params)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows

@app.route("/alerts")
def get_alerts():
    limit = request.args.get("limit", default=50, type=int)
    since = request.args.get("since")  # ISO8601 字符串
    alerts = query_alerts(limit=limit, since=since)
    return jsonify({"count": len(alerts), "data": alerts})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5099)
