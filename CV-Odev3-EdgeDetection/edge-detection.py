import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resmi yükle
image_path = 'sekil.png'  # Resminizin yolunu buraya ekleyin
image = cv2.imread(image_path)

# OpenCV, görüntüleri BGR formatında yükler, bu yüzden RGB'ye dönüştürüyoruz
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görüntüyü gri tonlamaya dönüştür
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Kenar tespiti yapmak için Canny algoritmasını kullan
edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

# Görüntüleri görselleştir
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Orijinal Resim")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Kenar Tespiti")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
