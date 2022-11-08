import argparse
import sqlite3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, help='/path/to/database/SQLDB/wikisqlite.db')
    args = parser.parse_args()

    con = sqlite3.connect(args.db_path)
    cur = con.cursor()
    res = cur.execute("SELECT wikipage FROM documents WHERE id = ?", ("82675", ))
    wikipages = res.fetchall()
    print(wikipages)