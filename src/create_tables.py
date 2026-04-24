import sqlite3
from datetime import datetime

DB_PATH = "memory_center.db"  # 可根据需要修改数据库文件名

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 创建 session_logs 表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS session_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        epoch_id INTEGER,
        turn_id INTEGER,
        role TEXT,
        raw_content TEXT,
        intermediate_steps TEXT,
        meta_data TEXT,
        is_compressed BOOLEAN
    )
    ''')

    # 创建 memory_summaries 表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memory_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        epoch_id INTEGER,
        summary TEXT,
        outcome TEXT,
        key_facts TEXT,
        created_at DATETIME DEFAULT (datetime('now','localtime'))
    )
    ''')

    conn.commit()
    conn.close()
    print("数据库和数据表创建完成！")

if __name__ == "__main__":
    create_tables()
