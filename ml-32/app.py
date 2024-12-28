from flask import Flask, request, jsonify  # Flask ile API oluşturmak ve JSON işlemleri için modüller
from flask_cors import CORS  # Cross-Origin Resource Sharing (CORS) kontrolü için modül
import os  # Dosya sistemi ile çalışmak için os modülü
import traceback  # Hata ayıklama için traceback modülü
from recommender import (  # recommender.py'den gerekli fonksiyonları içe aktarma
    load_and_preprocess_data,  # Veri yükleme ve ön işleme fonksiyonu
    train_and_save_model,  # Model eğitimi ve kaydetme fonksiyonu
    load_model,  # Eğitilmiş modeli yükleme fonksiyonu
    build_user_vector,  # Kullanıcı vektörü oluşturma fonksiyonu
    generate_recommendations  # Öneri üretme fonksiyonu
)

# Flask uygulamasını başlat
app = Flask(__name__)

# CORS yapılandırması - Herhangi bir origin'den gelen isteklere izin veriliyor
CORS(app, resources={r"/*": {"origins": "*"}})

# Model dosyasının yolu
MODEL_PATH = 'knn_model.pkl'

# Uygulama başlatıldığında modelin varlığını kontrol et
try:
    if not os.path.exists(MODEL_PATH):  # Eğer model dosyası yoksa
        print("Model dosyası bulunamadı. Yeni model eğitiliyor...")  # Bilgilendirme mesajı
        # Verileri yükle ve işle
        movies_df, ratings_df = load_and_preprocess_data()
        # Modeli eğit ve kaydet
        train_and_save_model(ratings_df, model_path=MODEL_PATH, n_neighbors=10)
    else:
        print("Var olan model kullanılacak...")  # Bilgilendirme mesajı
except Exception as e:  # Hata durumunda
    print("Hata:", e)  # Hata mesajını yazdır
    traceback.print_exc()  # Detaylı hata bilgisi yazdır

# /recommend endpoint'i - Kullanıcıdan gelen puanlara göre film önerisi döndürür
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    JSON formatında kullanıcıdan gelen puanlara göre film önerir.
    """
    try:
        # Kullanıcıdan gelen JSON verisini al
        data = request.get_json()
        print("Gelen veri:", data)  # Gelen veriyi konsola yazdır (debugging için)

        # Kullanıcı puanlamalarını ve öneri sayısını al
        user_ratings = data.get("user_ratings", {})  # Varsayılan olarak boş bir sözlük
        top_n = data.get("top_n", 10)  # Varsayılan olarak 10 öneri

        # Eğitilmiş modeli ve etkileşim matrisini yükle
        knn_model, interaction_matrix = load_model(MODEL_PATH)

        # Filmleri ve puanları yükle
        movies_df, ratings_df = load_and_preprocess_data()

        # Kullanıcının izlediği filmleri bir vektöre dönüştür
        user_vector = build_user_vector(user_ratings, movies_df)

        # Önerileri üret
        recommendations = generate_recommendations(
            user_vector=user_vector,  # Kullanıcı vektörü
            movies=movies_df,  # Filmler
            knn_model=knn_model,  # Eğitilmiş model
            interaction_matrix=interaction_matrix,  # Etkileşim matrisi
            top_k=10,  # En yakın 10 kullanıcı
            top_n=top_n  # Döndürülecek öneri sayısı
        )

        # Önerileri JSON formatına dönüştür
        response = [
            {"movieId": int(row["movieId"]),"tmdbId": int(row["tmdbId"]), "title": row["title"], "genres": row["genres"]}
            for _, row in recommendations.iterrows()
        ]

        print("Dönen öneriler:", response)  # Dönen önerileri konsola yazdır (debugging için)
        return jsonify(response)  # Önerileri JSON olarak döndür
    except Exception as e:  # Hata durumunda
        print("Hata:", e)  # Hata mesajını yazdır
        traceback.print_exc()  # Detaylı hata bilgisi yazdır
        return jsonify({"error": str(e)}), 400  # 400 Bad Request ile hata mesajı döndür

# Ana endpoint - API'nin çalıştığını doğrulamak için
@app.route("/", methods=["GET"])
def index():
    """
    API'nin ana endpoint'i.
    """
    return "Film Öneri Sistemi API - Hoş Geldiniz!"  # Hoş geldiniz mesajı döndür

# Uygulamayı başlat
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Flask uygulamasını localhost'ta çalıştır
