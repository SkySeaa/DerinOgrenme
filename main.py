import random

# 1. Veri Seti (Örnek: OR Mantık Kapısı) ve Başlangıç Ağırlıkları
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 1]

# Ağırlıkları elle veya rastgele başlatıyoruz
w = [random.uniform(-1, 1), random.uniform(-1, 1)]
b = random.uniform(-1, 1)
learning_rate = 0.1
epoch = 0

# 2. Hata 0 olana kadar çalışacak döngü
while True:
  epoch += 1
  total_error = 0
  dw = [0.0, 0.0]
  db = 0.0

  # Batch Size = Epoch Size (Tüm veri seti üzerinde döner)
  for i in range(len(X)):
    # İleri Yayılım (Forward Pass): z = w1*x1 + w2*x2 + b
    z = sum(w[j] * X[i][j] for j in range(len(w))) + b

    # Aktivasyon Fonksiyonu (Eşikleme / Step Function)
    pred = 1 if z >= 0 else 0

    # Hata Hesabı
    err = y[i] - pred
    total_error += abs(err)

    # Gradyanları biriktir (Batch birikimi)
    for j in range(len(w)):
      dw[j] += err * X[i][j]
    db += err

  # Ağırlık Güncellemesi (Epoch sonunda gerçekleşir)
  for j in range(len(w)):
    w[j] += learning_rate * dw[j]
  b += learning_rate * db

  print(f"Epoch {epoch} - Toplam Hata: {total_error}")

  # Koşul: Toplam hata 0 olduğunda döngüyü kır
  if total_error == 0:
    print(f"\nModel başarıyla eğitildi! Toplam Epoch Sayısı: {epoch}")
    print(f"Öğrenilen Ağırlıklar (w): {w}, Bias (b): {b}")
    break