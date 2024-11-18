import cv2
import requests
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np
import os

# YOLO modelini yükle
model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 model


def process_image(img, save_dir, save_name):
    """YOLO modelini kullanarak görüntüyü işler ve kaydeder."""
    try:
        # YOLO sınıflandırma
        results = model(img)
        annotated_img = results[0].plot()  # Sonuçları görüntü üzerine çizer

        # Kaydedilen görüntü yolu
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, annotated_img)
        print(f"Sonuç kaydedildi: {save_path}")

        # Görüntüyü göster
        cv2.imshow("Processed Image", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Görüntü işlenirken hata: {e}")


def classify_webcam():
    """Web kamerasından görüntü al ve sınıflandır."""
    cap = cv2.VideoCapture(0)  # Web kamerasını aç
    if not cap.isOpened():
        print("Web kamerası açılamadı!")
        return

    os.makedirs("runs/webcam", exist_ok=True)  # Kayıt klasörü oluştur
    print("Web kamerası açık, 'q' ile çıkabilirsiniz.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı!")
            break

        # YOLO sınıflandırma
        results = model(frame)
        annotated_frame = results[0].plot()  # Sonuçları görüntü üzerine çizer

        # Görüntüyü kaydet
        save_path = f"runs/webcam/frame_{frame_count}.jpg"
        cv2.imwrite(save_path, annotated_frame)
        frame_count += 1

        # Görüntüyü göster
        cv2.imshow("Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def classify_disk_image(image_path):
    """Diskteki bir görüntüyü al ve sınıflandır."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Görüntü yüklenemedi: {image_path}")
        return

    process_image(img, "runs/disk_image", "result.jpg")


def classify_url_image(image_url):
    """URL'den bir görüntü al ve sınıflandır."""
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # HTTP hatalarını kontrol et

        img = Image.open(BytesIO(response.content))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        process_image(img, "runs/url_image", "result.jpg")
    except requests.exceptions.RequestException as e:
        print(f"URL'den görüntü alınırken hata: {e}")
    except Exception as e:
        print(f"URL görüntüsü işlenirken hata: {e}")


# Fonksiyonları sırayla çalıştır
if __name__ == "__main__":
    print("1. Web kamera sınıflandırması başlıyor...")
    classify_webcam()

    print("2. Diskten görüntü sınıflandırması başlıyor...")
    classify_disk_image("image/bus.jpg")  # Test için uygun bir görüntü yolu verin

    print("3. URL'den görüntü sınıflandırması başlıyor...")
    classify_url_image("https://assets.volvo.com/is/image/VolvoInformationTechnologyAB/blue-bus?qlt=82&wid=1024&ts=1660212095501&dpr=off&fit=constrain")  # Test için uygun bir URL verin
