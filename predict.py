import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def extract_hsv_features_from_img(image):
    """Извлечение HSV-признаков из готового объекта изображения OpenCV"""
    image_resized = cv2.resize(image, (200, 200))
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
    
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    means = cv2.mean(hsv)[:3]
    stds = [np.std(hsv[:, :, i]) for i in range(3)]
    
    return np.concatenate([hist_h, hist_s, hist_v, means, stds])

def select_file():
    """Открывает графический диалог выбора файла"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) # Поднимаем окно выбора поверх всех окон
    
    file_path = filedialog.askopenfilename(
        title="Выберите изображение фрукта/овоща",
        filetypes=[
            ("Изображения", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Все файлы", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def predict_image():
    model_path = "ripeness_model.pkl"
    if not os.path.exists(model_path):
        print("Ошибка: Файл 'ripeness_model.pkl' не найден! Сначала запустите train.py.")
        return

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return

    print("Открываю диалог выбора файла...")
    image_path = select_file()

    if not image_path:
        print("Выбор файла отменен.")
        return

    print(f"Выбран файл: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: Не удалось прочитать изображение. Проверьте формат файла.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    features = extract_hsv_features_from_img(image)
    X_pred = np.array([features])

    prediction = model.predict(X_pred)[0] # Пример строки: "ripe apple"
    probabilities = model.predict_proba(X_pred)[0]
    confidence = max(probabilities) * 100

    parts = prediction.split(" ")
    ripeness_raw = parts[0] # ripe или unripe
    fruit_type = " ".join(parts[1:]).capitalize() # Apple, Dragon и т.д.

    ripeness_ru = "Спелый (Ripe)" if ripeness_raw == "ripe" else "Незрелый (Unripe)"

    print("\n" + "="*30)
    print(f"РЕЗУЛЬТАТ АНАЛИЗА:")
    print(f"Фрукт/Овощ: {fruit_type}")
    print(f"Статус:     {ripeness_ru}")
    print(f"Уверенность: {confidence:.1f}%")
    print("="*30)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(f"{fruit_type} — {ripeness_ru}\n(Уверенность: {confidence:.1f}%)", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_image()