import os
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# -------------------------------------------------------------------------
# 1) VERİYİ YÜKLEME VE ÖN İŞLEME
# -------------------------------------------------------------------------
def load_and_preprocess_data(movies_path='data/movies.csv',
                             ratings_path='data/ratings.csv',
                             min_user_ratings=20,
                             min_movie_ratings=20):
    """
    movies.csv ve ratings.csv dosyalarını yükleyerek gerekli filtrelemeleri yapar.

    Parametreler:
    ------------
    movies_path : str
        Filmlerin bulunduğu CSV dosya yolu.
    ratings_path : str
        Puanlamaların bulunduğu CSV dosya yolu.
    min_user_ratings : int
        Bir kullanıcının en az kaç film puanlamış olması gerektiği.
    min_movie_ratings : int
        Bir filmin en az kaç kullanıcı tarafından puanlanmış olması gerektiği.

    Dönüş:
    ------
    movies : pd.DataFrame
        Filmlerin bilgilerini tutan DataFrame (movieId, title, genres).
        Burada movieId yeniden indekslenmiş halidir, orijinali original_movieId.
    ratings : pd.DataFrame
        Uygun filtrelemelerden geçmiş kullanıcı-film puanları (userId, movieId, rating).
        movieId ve userId yeniden indekslenmiş haldedir.
    """

    # movies.csv dosyasını yükle
    movies = pd.read_csv(movies_path)
    # ratings.csv dosyasını yükle
    ratings = pd.read_csv(ratings_path)

    # movies tablosundaki "movieId" kolonunu "original_movieId" olarak yeniden adlandır
    movies.rename(columns={'movieId': 'original_movieId'}, inplace=True)


    # Kullanıcının film puanlama sayısını bul (her userId için)
    user_counts = ratings['userId'].value_counts()
    # min_user_ratings'tan az puanlaması olan kullanıcıları ele
    valid_users = user_counts[user_counts >= min_user_ratings].index
    # ratings tablosunu sadece geçerli kullanıcılarla filtrele
    ratings = ratings[ratings['userId'].isin(valid_users)]

    # Her filmin kaç kez puanlandığını bul (her movieId için)
    movie_counts = ratings['movieId'].value_counts()
    # min_movie_ratings'tan az puanlaması olan filmleri ele
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    # ratings tablosunu sadece geçerli filmlerle filtrele
    ratings = ratings[ratings['movieId'].isin(valid_movies)]

    # Filtrelenmiş userId'leri yeniden 0'dan başlayan index'lere dönüştür
    unique_user_ids = ratings['userId'].unique()
    user_to_index = {user_id: i for i, user_id in enumerate(unique_user_ids)}
    # Yeni userId değerlerini ratings tablosuna uygula
    ratings['userId'] = ratings['userId'].map(user_to_index)

    # Filtrelenmiş movieId'leri de yeniden 0'dan başlayan index'lere dönüştür
    unique_movie_ids = ratings['movieId'].unique()
    movie_to_index = {movie_id: i for i, movie_id in enumerate(unique_movie_ids)}
    # Yeni movieId değerlerini ratings tablosuna uygula
    ratings['movieId'] = ratings['movieId'].map(movie_to_index)

    # movies tablosunu yalnızca valid_movies içeren satırlarla sınırlayalım
    movies = movies[movies['original_movieId'].isin(valid_movies)]

    # Orijinal movieId => yeni movieId map'i oluştur (old_id -> 0..N)
    old_to_new_id = {old_id: movie_to_index[old_id]
                     for old_id in valid_movies
                     if old_id in movie_to_index}

    # movies tablosuna yeni movieId sütunu ekle
    movies['movieId'] = movies['original_movieId'].map(old_to_new_id)


    # Artık "movies" DataFrame'inde:
    # - "original_movieId" => CSV'deki asıl id
    # - "movieId" => yeniden indekslenmiş (0..N)
    # - "title" ve "genres" => film başlığı ve tür

    return movies, ratings


# -------------------------------------------------------------------------
# 2) MODEL EĞİTİMİ VE KAYDETME
# -------------------------------------------------------------------------
def train_and_save_model(ratings, model_path='knn_model.pkl', n_neighbors=5):
    """
    KNN tabanlı kullanıcı benzerlik modelini eğitir ve diske kaydeder.

    Parametreler:
    ------------
    ratings : pd.DataFrame
        Kullanıcı-film puanlarını içeren DataFrame (userId, movieId, rating).
        Burada movieId ve userId yeniden indekslenmiş haldedir.
    model_path : str
        Eğitilen modelin kaydedileceği dosya yolu.
    n_neighbors : int
        Benzerlik aramasında kullanılacak en yakın komşu sayısı.
    """

    # Toplam kullanıcı sayısını bul
    num_users = ratings['userId'].nunique()
    # Toplam film sayısını bul
    num_movies = ratings['movieId'].nunique()

    # Kullanıcı-film etkileşim matrisini oluşturma
    # satırlar userId, sütunlar movieId olacak
    user_ids = ratings['userId'].values
    movie_ids = ratings['movieId'].values
    user_ratings = ratings['rating'].values

    # CSR formatında sparse matris oluştur (daha az bellek kullanımı için)
    interaction_matrix = csr_matrix(
        (user_ratings, (user_ids, movie_ids)),
        shape=(num_users, num_movies)
    )

    # KNN modelini oluştur (user-based)
    # metric='cosine' => benzerlik ölçütü olarak kosinüs benzerliği
    # algorithm='brute' => veri seti büyüdükçe farklı yöntemler denenebilir
    knn_model = NearestNeighbors(metric='cosine',
                                 algorithm='brute',
                                 n_neighbors=n_neighbors)

    # KNN modelini etkileşim matrisi üzerinden eğit
    knn_model.fit(interaction_matrix)

    # Model ve etkileşim matrisini pickle ile kaydet
    with open(model_path, 'wb') as f:
        pickle.dump((knn_model, interaction_matrix), f)

    print(f"KNN modeli başarıyla eğitildi ve '{model_path}' dosyasına kaydedildi.")


# -------------------------------------------------------------------------
# 3) MODELİ YÜKLEME VE ÖNERİ ÜRETME
# -------------------------------------------------------------------------
def load_model(model_path='knn_model.pkl'):
    """
    Eğitilmiş modeli ve kullanıcı-film matrisini yükler.

    Parametreler:
    ------------
    model_path : str
        Pickle ile kaydedilmiş model dosyasının yolu.

    Dönüş:
    ------
    knn_model : NearestNeighbors
        Eğitilmiş KNN modeli.
    interaction_matrix : csr_matrix
        (num_users, num_movies) boyutundaki kullanıcı-film etkileşim matrisi.
    """
    # Dosyayı aç ve modeli + etkileşim matrisini pickle ile yükle
    with open(model_path, 'rb') as f:
        knn_model, interaction_matrix = pickle.load(f)
    return knn_model, interaction_matrix


def get_similar_users_ratings(user_vector,
                              knn_model,
                              interaction_matrix,
                              top_k=5):
    """
    Yeni bir kullanıcı vektörüne benzer kullanıcıları bulur
    ve bu kullanıcıların puanlamalarını döndürür.

    Parametreler:
    ------------
    user_vector : np.ndarray (shape: (1, num_movies))
        Yeni kullanıcının puanlarını içeren tek satırlık vektör.
    knn_model : NearestNeighbors
        Eğitilmiş KNN modeli.
    interaction_matrix : csr_matrix
        (num_users, num_movies) şeklinde kullanıcı-film matrisini tutar.
    top_k : int
        Kaç benzer kullanıcının döndürüleceği.

    Dönüş:
    ------
    similar_users_ratings : np.ndarray (shape: (top_k, num_movies))
        Benzer kullanıcıların film puanlarını tutan matris.
    indices : np.ndarray
        Benzer kullanıcıların index bilgileri (userId'ler).
    """

    # KNN modelinin kneighbors fonksiyonuyla en yakın kullanıcıları bul
    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=top_k)

    # Benzer kullanıcıların (top_k adet) matris satırlarını toarray ile al
    similar_users_ratings = interaction_matrix[indices[0]].toarray()

    return similar_users_ratings, indices[0]


def generate_recommendations(user_vector,
                             movies,
                             knn_model,
                             interaction_matrix,
                             top_k=5,
                             top_n=5):
    """
    Yeni kullanıcının puanlarını temsil eden vektöre dayanarak film önerisi yapar.

    Parametreler:
    ------------
    user_vector : np.ndarray (shape: (1, num_movies))
        Yeni kullanıcının puanlarını (varsa) içeren tek satırlık vektör.
    movies : pd.DataFrame
        Filmler hakkında bilgi (movieId, original_movieId, title, genres).
        Burada movieId => yeniden indekslenmiş, original_movieId => CSV'deki gerçek id.
    knn_model : NearestNeighbors
        Eğitilmiş KNN modeli.
    interaction_matrix : csr_matrix
        (num_users, num_movies) şeklinde kullanıcı-film matrisini tutar.
    top_k : int
        Kaç benzer kullanıcıya bakacağımız.
    top_n : int
        Kaç film önerisi döndürüleceği.

    Dönüş:
    ------
    recommendations : pd.DataFrame
        Önerilen filmlerin (movieId, title, genres) bilgileri.
        Burada movieId, CSV'deki orijinal id ile aynı olacak şekilde tekrar dönüştürülür.
    """

    # 1) Benzer kullanıcıların puanlarını al
    similar_users_ratings, user_indices = get_similar_users_ratings(
        user_vector, knn_model, interaction_matrix, top_k=top_k
    )

    # 2) Benzer kullanıcıların puanlarının ortalamasını al
    avg_ratings = similar_users_ratings.mean(axis=0)

    # Kullanıcının zaten puanladığı filmler:
    user_rated_indices = np.where(user_vector[0] > 0)[0]
    # Tekrar önermemek için bu film skorlarını -1 yap
    avg_ratings[user_rated_indices] = -1

    # 3) En yüksek ortalama puana sahip filmleri büyükten küçüğe doğru sırala
    top_movie_indices = np.argsort(avg_ratings)[::-1][:top_n]

    # 4) Önerilecek filmlerin meta bilgilerini çek
    #    Elimizdeki "movieId" sütunu 0..N (mapped id), bu index'lerin top_movie_indices ile kesişimini alacağız.
    recommended_movies = movies[movies['movieId'].isin(top_movie_indices)].copy()

    # 5) Tür çeşitliliği sağlama (opsiyonel):
    #    Burada basit bir fonksiyonla en fazla 2 film kısıtlaması örneği gösteriliyor.
    recommended_movies = diversify_recommendations(recommended_movies, top_n=top_n)

    # 6) Çıktıda orijinal movieId'yi dönmek istersek, "original_movieId" verisini tekrar "movieId" kolonu yapabiliriz.
    #    Yeni map'lenmiş "movieId"'yi "mappedId" gibi bir sütuna alıp orijinali "movieId" yapalım:
    recommended_movies.rename(columns={'movieId': 'mapped_movieId'}, inplace=True)
    recommended_movies.rename(columns={'original_movieId': 'movieId'}, inplace=True)
    #TMDB ID'leri sütun olarak ekleyelim
    links_map = load_links_map()
    recommended_movies['tmdbId'] = recommended_movies['movieId'].map(links_map)






    # Artık "movieId" sütunu, CSV'deki asıl id (örnek: 356) olacaktır.
    # "mapped_movieId" ise sistemin 0..N formatında kullandığı id.

    return recommended_movies


def diversify_recommendations(recommended_movies, top_n=10):
    """
    Tür çeşitliliği (genre diversity) sağlamak adına basit bir yaklaşım:
    - Filmler, öncelikle sıralı (örneğin skor sırası) kabul edilir.
    - Her türe en fazla 2 film koyar (daha sofistike yöntemler de mümkündür).

    Parametreler:
    ------------
    recommended_movies : pd.DataFrame
        Önerilen filmlerin bilgileri (mapped_movieId/original_movieId/title/genres).
    top_n : int
        Sonuçta kaç tane film tutacağımız.

    Dönüş:
    ------
    final_list_df : pd.DataFrame
        Seçilen filmlerin DataFrame formatı.
    """

    # Sonuca ekleyeceğimiz filmleri tutmak için boş liste
    final_list = []
    # Tür başına kaç adet film eklediğimizi takip eden dict
    genre_counts = {}
    # Her türe en fazla 2 tane eklemek istediğimizi varsayalım
    max_per_genre = 2

    # recommended_movies üzerindeki her satırı (film) sırayla incele
    for _, row in recommended_movies.iterrows():
        # Filmin türlerini '|' karakterine göre ayır
        genres = row['genres'].split('|')
        # Bu filmi ekleyip ekleyemeyeceğimizi belirlemek için bayrak
        can_add = False

        # Her tür üzerinde dön
        for g in genres:
            # Eğer o türde henüz 2'den az film eklediysek, bu filmi ekleyebilirsin
            if genre_counts.get(g, 0) < max_per_genre:
                can_add = True
                break  # Bir tür ekleyebiliyorsa yetiyor, can_add = True

        # Eğer eklenebilecekse
        if can_add:
            # final_list'e bu satırı ekle
            final_list.append(row)
            # Filmin bütün türlerinin sayaçlarını 1 artır (veya tek bir türü de artırabilirsiniz)
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + 1

        # Zaten istediğimiz kadar film (top_n) doldurmuşsak döngüyü kes
        if len(final_list) >= top_n:
            break

    # final_list'i DataFrame'e dönüştür
    final_list_df = pd.DataFrame(final_list)
    return final_list_df


# -------------------------------------------------------------------------
# Yardımcı Fonksiyon: Gelen kullanıcı puanlarını vektöre dönüştürme
# -------------------------------------------------------------------------
def build_user_vector(user_ratings, movies):
    """
    Kullanıcıdan gelen {'Toy Story (1995)': 5, 'Bad Boys (1995)': 4, ...} gibi
    sözlük yapısını (1, num_movies) boyutunda bir numpy vektörüne dönüştürür.

    Parametreler:
    ------------
    user_ratings : dict
        {'Film Adı': rating} şeklinde kullanıcının verdiği puanları içerir.
    movies : pd.DataFrame
        Filmler (movieId, original_movieId, title, genres).
        Burada 'movieId' => yeniden indekslenmiş (0..N), 'title' => film adı.

    Dönüş:
    ------
    user_vector : np.ndarray (shape: (1, num_movies))
        Kullanıcının puanlarını barındıran tek satırlık numpy array.
    """

    # Kaç farklı mapped (yeniden indekslenmiş) movieId varsa o kadar sütun olacak
    num_movies = movies['movieId'].nunique()
    # Sıfırlarla dolu bir vektör oluştur (1 x num_movies boyutunda)
    user_vector = np.zeros((1, num_movies))

    # "title => mapped movieId" eşleşmesi oluşturmak için bir sözlük yap
    # Burada movies['title'] => film adı, movies['movieId'] => yeniden indekslenmiş id
    title_to_id = dict(zip(movies['title'], movies['movieId']))

    # Kullanıcının puan verdiği her film başlığı için
    for film_title, rating in user_ratings.items():
        # Eğer film başlığı movies tablosunda yoksa atla
        if film_title not in title_to_id:
            continue
        # mapped id'yi al
        idx = title_to_id[film_title]
        # user_vector'da bu filme karşılık gelen sütuna kullanıcının puanını yerleştir
        user_vector[0, idx] = rating

    # Sonuç olarak (1, num_movies) boyutunda bir vektör elde etmiş olduk
    return user_vector



def load_links_map(links_path='data/links.csv'):
    """
    links.csv şu formatta olduğunu varsayalım:
      movieId,imdbId,tmdbId
      356,0109830,13
      780,0116629,602
      ...
    """
    links_df = pd.read_csv(links_path)

    # dictionary: {356: 13, 780: 602, ...}
    links_map = dict(zip(links_df['movieId'], links_df['tmdbId']))
    return links_map
