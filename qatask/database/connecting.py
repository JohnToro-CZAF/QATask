import sqlite3
con = sqlite3.connect("/home/ubuntu/hoang.pn200243/AQ/QATask/qatask/database/SQLDB/wikisqlite.db")
cur = con.cursor()
res = cur.execute("SELECT wikipage FROM documents WHERE id = ?", ("82675", ))
wikipages = res.fetchall()
print(wikipages)