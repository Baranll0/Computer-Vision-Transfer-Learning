import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# CSV dosyasının yolu
csv_file_path = 'image_src.csv'

# Resimlerin indirileceği klasör
output_folder = 'images'
os.makedirs(output_folder, exist_ok=True)

# CSV dosyasını oku
try:
    df = pd.read_csv(csv_file_path, encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 ile okuma başarısız. Latin1 ile tekrar denenecek...")
    df = pd.read_csv(csv_file_path, encoding='latin1')

# 'image_url' sütununu kullanarak resimleri indir
for index, row in df.iterrows():
    image_url = row['image_url'].split('?')[0]  # URL'yi temizle
    print(f'Deneme: {image_url}')  # Hangi URL'nin denendiğini göster

    try:
        # Resmi indir
        img_response = requests.get(image_url, timeout=10)  # Zaman aşımını 10 saniye olarak ayarla
        img_response.raise_for_status()  # Hata kontrolü

        # Resmi bellekten açarak kontrol et
        img = Image.open(BytesIO(img_response.content))

        # Uygun dosya uzantısını belirle
        file_extension = img.format.lower()  # Resmin formatını al
        valid_extensions = ['jpeg', 'jpg', 'png', 'webp']

        if file_extension in valid_extensions:
            # Resim dosya adını belirle
            image_name = f'image_{index}.{file_extension}'  # Dosya adında uygun uzantıyı kullan
            image_path = os.path.join(output_folder, image_name)

            # Resmi dosyaya kaydet
            img.save(image_path)
            print(f'{image_name} indirildi.')
        else:
            print(f'Uygun dosya uzantısı değil: {image_url}')
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP Hatası: {http_err} - {image_url}')
    except requests.exceptions.Timeout:
        print(f'Timeout hatası: {image_url}')
    except requests.exceptions.RequestException as e:
        print(f'İstek hatası: {e} - {image_url}')
    except Exception as e:
        print(f'Genel hata: {e} - {image_url}')
