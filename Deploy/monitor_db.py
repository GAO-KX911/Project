import sqlite3, os
DB_PATH = "/home/njust/Fire/Deploy/monitor.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        captured_at DATETIME NOT NULL,
        camera_id TEXT,
        class_label TEXT NOT NULL,
        probability REAL NOT NULL,
        detection_json TEXT,
        image_path TEXT NOT NULL
    );
""")
conn.commit()
conn.close()
