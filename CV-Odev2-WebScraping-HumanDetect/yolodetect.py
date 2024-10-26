import cv2
import os
from ultralytics import YOLO

# YOLO modelini yükle (modeli uygun yoldan yükleyin)
model = YOLO("yolov8n.pt")

# Test edilecek görüntülerin bulunduğu dizin
images_dir = 'images/'  # Görüntülerin bulunduğu klasörün yolunu belirtin

# Görüntüleri döngüye al
for image_name in os.listdir(images_dir):
    if image_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):  # Sadece görüntü dosyalarını kontrol et
        # Görüntüyü yükle
        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)

        # Eğer görüntü yüklenemediyse bir hata mesajı göster
        if img is None:
            print(f"{image_name}: Görüntü yüklenemedi.")
            continue

        # Model ile nesne tespiti yap
        results = model(img)

        # Tespit edilen nesneleri al
        detections = results[0].boxes  # Tespitleri al
        detected_classes = detections.cls.tolist()  # Sınıfları al
        probs = detections.conf.tolist()  # Olasılık değerlerini al

        # İnsan sınıfı (COCO datasetine göre) 0 indeksidir
        human_detected = any(cls == 0 for cls in detected_classes)

        if human_detected:
            print(f"{image_name}: İnsan tespit edildi.")
        else:
            # En yüksek olasılığa sahip nesneyi bul
            if detected_classes:  # Eğer tespit edilen sınıf varsa
                max_prob_index = probs.index(max(probs))
                max_class_index = detected_classes[max_prob_index]
                max_class_name = model.names[int(max_class_index)]  # Sınıf adını al
                max_prob_value = probs[max_prob_index]  # En yüksek olasılık değeri
                print(f"{image_name}: İnsan tespit edilmedi, en yüksek olasılıkla tespit edilen nesne: {max_class_name} (olasılık: {max_prob_value:.2f})")
            else:
                print(f"{image_name}: Tespit edilen nesne yok.")
