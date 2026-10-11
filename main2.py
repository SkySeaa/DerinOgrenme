x1, x2 = 1, 0
y = 1

w1, w2 = 0.5, -0.5
b = 0.0

learning_rate = 0.1
epoch_size = batch_size = 1

while True:
    z = x1 * w1 + x2 * w2 + b
    prediction = 1 if z >= 0 else 0
    error = y - prediction

    print("Tahmin:", prediction, "Hata:", error)

    if error == 0:
        break

    w1 += learning_rate * error * x1
    w2 += learning_rate * error * x2
    b += learning_rate * error

print("Son ağırlıklar:", w1, w2, b)