import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Model ve görüntü yolları
model = YOLO("/media/baran/Disk1/CVINonuders/CV-Odev4-Kayısı-Detect/src/results/train6/weights/best.pt")
image_path = '/media/baran/Disk1/CVINonuders/CV-Odev4-Kayısı-Detect/dataset/test/images/4_png.rf.b33148978bf1932fe34d1add9219a6af.jpg'  # Görüntü yolunu buraya ekleyin
save_folder = "/media/baran/Disk1/CVINonuders/CV-Odev4-Kayısı-Detect/src/predict_results/"  # Kaydedilecek klasör

# Kaydedilecek klasörün var olup olmadığını kontrol et
os.makedirs(save_folder, exist_ok=True)

# Görüntüyü yükle
img = Image.open(image_path)

# İleriye dönük (prediction) işlemi yap
results = model.predict(img)

# results bir liste olduğundan ilk öğeyi alıyoruz
result = results[0]  # İlk sonuç objesini al

# Tahminler: Kutular, Etiketler ve Güven skorları
boxes = result.boxes  # Kutuların koordinatları
labels = result.names  # Sınıf isimleri
confidences = boxes.conf  # Güven skorları

# Görüntü üzerinde çizim yapabilmek için bir kopyasını oluştur
draw = ImageDraw.Draw(img)

# Kutular üzerinde gezinip onları görüntüye çiz
for box, label, confidence in zip(boxes.xyxy.cpu().numpy(), result.names, confidences.cpu().numpy()):
    x1, y1, x2, y2 = box  # Kutunun koordinatları
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1), f'{label} {confidence:.2f}', fill="red")

# Orijinal dosya adını al
file_name = os.path.basename(image_path)

# Kaydetme yolunu ayarla
save_path = os.path.join(save_folder, file_name)

# Görüntüyü kaydet
img.save(save_path)

# İsteğe bağlı olarak görüntüyü göster
plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis('off')
plt.show()

print(f"Image saved at {save_path}")
