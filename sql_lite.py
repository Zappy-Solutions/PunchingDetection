import sqlite3

conn = sqlite3.connect("violations.db")
cursor = conn.cursor()

cursor.executescript("""
DROP TABLE IF EXISTS Violations;
CREATE TABLE Violations (
    UserID TEXT,
    Time TEXT,
    Issue TEXT,
    ImagePath TEXT
);
""")
conn.commit()
conn.close()
print("Database created and table ensured.")