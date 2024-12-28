# movie_recommendation_with_titles.py

# Gerekli kütüphanelerin import edilmesi
import pandas as pd  # Veri manipülasyonu ve analizi için
import numpy as np  # Sayısal işlemler için
from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity hesaplaması için
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test olarak bölmek için
from sklearn.metrics import mean_squared_error  # Model performansını değerlendirmek için
from math import sqrt  # Karekök almak için

# 1. Veri Setinin Yüklenmesi
# MovieLens 100k veri setini kullanacağız. Veri setini indirip proje dizininizde uygun bir yere yerleştirdiğinizi varsayıyoruz.
ratings_path = 'ml-100k/u.data'  # Kullanıcı değerlendirmelerini içeren dosyanın yolu
movies_path = 'ml-100k/u.item'  # Film bilgilerini içeren dosyanın yolu

# 2. Verilerin Okunması ve DataFrame Oluşturulması
# Kullanıcı değerlendirmeleri
ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']  # Sütun isimleri
ratings = pd.read_csv(ratings_path, sep='\t', names=ratings_columns, encoding='latin-1')  # CSV dosyasını okuma

# Film bilgileri
movies_columns = [
    'movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
    'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]  # Sütun isimleri
movies = pd.read_csv(movies_path, sep='|', names=movies_columns, encoding='latin-1')  # CSV dosyasını okuma

# 3. Veri Ön İşleme
# Kullanıcı ve film ID'lerini pivot tabloya çevirerek kullanıcı-film matrisini oluşturuyoruz.
user_movie_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
# Her satır bir kullanıcıyı, her sütun bir filmi temsil eder ve boş hücreler sıfır ile doldurulur.

# 4. Eğitim ve Test Setine Bölme
# Veriyi %80 eğitim, %20 test olacak şekilde böleceğiz. Rastgelelik için random_state=42 kullanıyoruz.
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Eğitim verisi için kullanıcı-film matrisini oluşturma
train_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Test verisi için kullanıcı-film matrisini oluşturma
test_matrix = test_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# 5. Kullanıcı Benzerliğinin Hesaplanması
# Cosine similarity kullanarak kullanıcılar arasındaki benzerlikleri hesaplıyoruz.
user_similarity = cosine_similarity(train_matrix)  # Kullanıcı benzerlik matrisi
user_similarity_df = pd.DataFrame(user_similarity, index=train_matrix.index, columns=train_matrix.index)


# Benzerlik matrisini DataFrame'e çeviriyoruz, satır ve sütun isimlerini kullanıcı ID'leri olarak ayarlıyoruz.

# 6. Tahmin Fonksiyonunun Oluşturulması
def predict_rating(user_id, movie_id, train_matrix, user_similarity_df):
    """
    Kullanıcı tabanlı işbirlikçi filtreleme ile belirli bir kullanıcının belirli bir film için tahmin edilen puanını hesaplar.

    Args:
        user_id (int): Tahmin yapılacak kullanıcı ID'si.
        movie_id (int): Tahmin yapılacak film ID'si.
        train_matrix (DataFrame): Eğitim verisinin kullanıcı-film matrisi.
        user_similarity_df (DataFrame): Kullanıcı benzerlik matrisi.

    Returns:
        float: Tahmin edilen puan.
    """
    # Eğer film eğitim setinde yoksa puan verilemez.
    if movie_id not in train_matrix.columns:
        return 0

        # Kullanıcının diğer kullanıcılarla olan benzerlik skorlarını alıyoruz.
    similarity_scores = user_similarity_df[user_id]

    # İlgili filmi izleyen diğer kullanıcıların puanlarını alıyoruz.
    movie_ratings = train_matrix[movie_id]

    # Filmi izleyen (puan vermiş) kullanıcıların ID'lerini buluyoruz.
    non_zero_indices = movie_ratings[movie_ratings > 0].index

    # Eğer hiç kimse filmi izlemediyse puan verilemez.
    if len(non_zero_indices) == 0:
        return 0

        # Benzerlik skorlarını ve puanları alıyoruz.
    relevant_similarities = similarity_scores[non_zero_indices]
    relevant_ratings = movie_ratings[non_zero_indices]

    # Eğer benzerlik skorları toplamı sıfırsa puan verilemez.
    if relevant_similarities.sum() == 0:
        return 0

        # Ağırlıklı ortalamayı hesaplıyoruz.
    predicted_rating = np.dot(relevant_similarities, relevant_ratings) / relevant_similarities.sum()
    return predicted_rating


# 7. Tahminlerin Yapılması ve Değerlendirilmesi
predictions = []  # Tahmin edilen puanlar
actuals = []  # Gerçek puanlar

# Test verisindeki her kullanıcı-film çifti için tahmin yapıyoruz.
for index, row in test_data.iterrows():
    user = row['user_id']  # Kullanıcı ID'si
    movie = row['movie_id']  # Film ID'si
    actual = row['rating']  # Gerçek puan
    predicted = predict_rating(user, movie, train_matrix, user_similarity_df)  # Tahmin edilen puan
    predictions.append(predicted)  # Tahmin edilen puanı listeye ekleme
    actuals.append(actual)  # Gerçek puanı listeye ekleme

# Root Mean Squared Error (RMSE) hesaplama
rmse = sqrt(mean_squared_error(actuals, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")  # Model performansını ekrana yazdırma


# 8. Yardımcı Fonksiyon: Film İsmi ile Film ID'si Bulma
def get_movie_id(movie_title, movies_df):
    """
    Verilen film ismine karşılık gelen film_id'yi döndürür.
    Eğer birden fazla eşleşme varsa, kullanıcıya seçim yapmasını ister.
    Eğer hiç eşleşme yoksa, hata mesajı verir.

    Args:
        movie_title (str): Film ismi.
        movies_df (DataFrame): Film bilgilerini içeren DataFrame.

    Returns:
        int or None: Film ID'si veya bulunamazsa None.
    """
    # Film ismine göre eşleşen filmleri bulma (büyük/küçük harf duyarlı değil)
    matched_movies = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)]

    if matched_movies.empty:
        # Eğer hiçbir film bulunamazsa
        print(f"'{movie_title}' adlı bir film bulunamadı.")
        return None
    elif len(matched_movies) == 1:
        # Tek bir film bulunursa direkt ID'yi döndür
        return matched_movies.iloc[0]['movie_id']
    else:
        # Birden fazla film bulunursa kullanıcıdan seçim yapmasını iste
        print(f"Birden fazla '{movie_title}' adlı film bulundu. Lütfen tam isimle giriniz:")
        for idx, row in matched_movies.iterrows():
            print(f"{row['movie_id']}: {row['title']}")
        selected_id = input("Seçmek istediğiniz film ID'sini giriniz: ")
        try:
            selected_id = int(selected_id)
            if selected_id in matched_movies['movie_id'].values:
                return selected_id
            else:
                print("Geçersiz film ID'si seçtiniz.")
                return None
        except ValueError:
            print("Geçerli bir sayı girmediniz.")
            return None


# 9. Yeni Kullanıcı için Öneri Yapma (Film İsimleriyle)
def recommend_movies_new_user_with_titles(new_user_ratings_titles, train_matrix, user_similarity_df, movies, top_n=5):
    """
    Yeni bir kullanıcı için en iyi önerilen filmleri listeler.
    Kullanıcı, film isimlerini kullanarak puanlama yapar.

    Args:
        new_user_ratings_titles (dict): Film isimleri ve puanları (e.g., {"Toy Story (1995)": 5, ...})
        train_matrix (DataFrame): Eğitim verisinin kullanıcı-film matrisi.
        user_similarity_df (DataFrame): Kullanıcı benzerlik matrisi.
        movies (DataFrame): Film bilgilerini içeren DataFrame.
        top_n (int): Önerilecek film sayısı.

    Returns:
        list: Önerilen filmlerin isimleri ve tahmini puanları.
    """
    new_user_ratings = {}  # Yeni kullanıcının film ID'leri ve puanları

    # Kullanıcı tarafından sağlanan film isimlerini film ID'lerine çevirme
    for title, rating in new_user_ratings_titles.items():
        movie_id = get_movie_id(title, movies)  # Film ismine göre ID bulma
        if movie_id is not None:
            new_user_ratings[movie_id] = rating  # Eğer film bulunursa, ID ve puanı ekle
        else:
            print(f"'{title}' adlı film değerlendirmeye dahil edilemedi.")  # Film bulunamazsa uyarı ver

    if not new_user_ratings:
        # Eğer hiçbir geçerli değerlendirme bulunamazsa, fonksiyon boş liste döndürür
        print("Yeni kullanıcı için hiçbir geçerli değerlendirme bulunamadı.")
        return []

    # Yeni kullanıcıyı eğitim matrisine eklemek
    new_user_id = train_matrix.index.max() + 1  # Yeni kullanıcı ID'si (var olan ID'lerin sonuncusu + 1)
    new_user_data = pd.Series(new_user_ratings, name=new_user_id)  # Yeni kullanıcının puanlarını içeren Series
    train_matrix_extended = pd.concat([train_matrix, new_user_data.to_frame().T], sort=True).fillna(0)
    # Eğitim matrisine yeni kullanıcıyı ekliyoruz ve boş hücreleri sıfır ile dolduruyoruz

    # Benzerlik hesaplama (yeni matrisle)
    user_similarity_extended = cosine_similarity(train_matrix_extended)
    user_similarity_df_extended = pd.DataFrame(user_similarity_extended, index=train_matrix_extended.index,
                                               columns=train_matrix_extended.index)

    # Tahmin Fonksiyonunu Yeniden Tanımlama (yeni matrisle)
    def predict_new_user_rating(user_id, movie_id, train_matrix, user_similarity_df):
        """
        Yeni kullanıcının belirli bir film için tahmini puanını hesaplar.
        """
        if movie_id not in train_matrix.columns:
            return 0  # Film eğitim setinde yoksa puan verilemez.

        similarity_scores = user_similarity_df[user_id]  # Kullanıcının benzerlik skorlarını al
        movie_ratings = train_matrix[movie_id]  # Filmin tüm kullanıcılar tarafından verilen puanlarını al
        non_zero_indices = movie_ratings[movie_ratings > 0].index  # Filmi izleyen kullanıcıları bul

        if len(non_zero_indices) == 0:
            return 0  # Hiç kimse izlemediyse puan verilemez.

        relevant_similarities = similarity_scores[non_zero_indices]  # Benzerlik skorlarını filtrele
        relevant_ratings = movie_ratings[non_zero_indices]  # İlgili puanları al

        if relevant_similarities.sum() == 0:
            return 0  # Benzerlik skorları toplamı sıfırsa puan verilemez.

        # Ağırlıklı ortalamayı hesapla
        predicted_rating = np.dot(relevant_similarities, relevant_ratings) / relevant_similarities.sum()
        return predicted_rating

    # Yeni kullanıcının henüz değerlendirmediği filmleri bulma
    new_user_series = train_matrix_extended.loc[new_user_id]
    unrated_movies = new_user_series[new_user_series == 0].index.tolist()

    # Her bir henüz değerlendirilmemiş film için tahmin edilen puanı hesaplama
    movie_predictions = []
    for movie_id in unrated_movies:
        pred_rating = predict_new_user_rating(new_user_id, movie_id, train_matrix_extended, user_similarity_df_extended)
        movie_predictions.append((movie_id, pred_rating))

    # Tahmin edilen puanlara göre filmleri sıralama
    movie_predictions.sort(key=lambda x: x[1], reverse=True)

    # En yüksek puanlı top_n filmi seçme
    top_movies = movie_predictions[:top_n]

    # Seçilen filmlerin isimlerini ve tahmini puanlarını listeye ekleme
    recommendations = []
    for movie_id, rating in top_movies:
        # Film ID'sine göre film ismini bulma
        movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        recommendations.append((movie_title, rating))

    return recommendations



# 10. Yeni Kullanıcı Verileri ve Öneri (Film İsimleriyle)
# Kullanıcının değerlendirdiği filmleri film isimleriyle ve puanlarıyla tanımlayan bir sözlük
new_user_ratings_titles = {
    "Toy Story": 5,  # Toy Story (1995)
    "Four Rooms": 3,  # Star Wars (1977)
    "White Balloon": 4,  # Fargo (1996)
    "Taxi Driver": 2,  # The Godfather (1972)
    "Bad Boys": 5  # Die Hard (1988)
}

# Yeni kullanıcı için önerileri almak
recommended_new_user = recommend_movies_new_user_with_titles(
    new_user_ratings_titles,  # Yeni kullanıcının değerlendirmeleri
    train_matrix,  # Eğitim matrisimiz
    user_similarity_df,  # Kullanıcı benzerlik matrisi
    movies,  # Film bilgilerini içeren DataFrame
    top_n=30  # Önerilecek film sayısı
)

# Önerileri ekrana yazdırma
print(f"\nYeni Kullanıcı için Önerilen Filmler:")
for movie, rating in recommended_new_user:
    print(f"{movie} (Tahmini Puan: {rating:.2f})")
