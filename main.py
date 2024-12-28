import pymysql
from flask import Flask, request, jsonify

app = Flask(__name__)

def get_db_connection():
    return pymysql.connect(
        host="157.173.104.238",
        user="balabi4359",
        password="Aliolkac4310*",
        db="movierec",   # senin veritabanı adın (phpMyAdmin’de oluşturduğun)
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data["username"]
    password_hash = data["password_hash"]  # Düz şifre tutmayın; hash kullanın

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = "INSERT INTO users (username, password_hash) VALUES (%s, %s)"
            cursor.execute(sql, (username, password_hash))
        conn.commit()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)}), 400
    finally:
        conn.close()
