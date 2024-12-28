import requests

url = "http://localhost:5000/recommend"
payload = {"movies": [{"film_adi": "Toy Story", "yildiz_sayi": 5},
                      {"film_adi": "Sabrina", "yildiz_sayi": 5},
                      {"film_adi": "GoldenEye", "yildiz_sayi": 5},
                      {"film_adi": "Four Rooms", "yildiz_sayi": 5},
                      {"film_adi": "Assassins", "yildiz_sayi": 5}]}

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Öneriler geldi:")
    print(response.json())
else:
    print(f"Hata: {response.status_code}")
    print(response.json())
