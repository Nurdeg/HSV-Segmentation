import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def process_and_visualize(image_path):
    # 1. Загрузка модели
    if not os.path.exists("ripeness_model.pkl"):
        print("Сначала запустите train.py для обучения модели!")
        return
        
    with open("ripeness_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 2. Чтение изображения
    image = cv2.imread(image_path)
    if image is None:
        print("Не удалось прочитать файл изображения.")
        return
        
    image_resized = cv2.resize(image, (250, 250))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # --- ЭТАПЫ ОБРАБОТКИ (Computer Vision Pipeline) ---
    # Шаг 1: Фильтрация (Gaussian Blur)
    blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)
    
    # Шаг 2: Сегментация (HSV + Thresholding)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    _, thresh = cv2.threshold(hsv[:, :, 2], 40, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Шаг 3: Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_image = image_rgb.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Рисуем контур зеленой линией толщиной 2 пикселя
        cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
        
        contour_area = cv2.contourArea(largest_contour)
        contour_perimeter = cv2.arcLength(largest_contour, True)
    else:
        contour_area, contour_perimeter = 0, 0

    # Извлечение признаков для модели
    fruit_only = cv2.bitwise_and(hsv, hsv, mask=mask)
    hist_h = cv2.calcHist([fruit_only], [0], mask, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([fruit_only], [1], mask, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([fruit_only], [2], mask, [16], [0, 256]).flatten()
    cv2.normalize(hist_h, hist_h); cv2.normalize(hist_s, hist_s); cv2.normalize(hist_v, hist_v)
    mean_color = cv2.mean(hsv, mask=mask)[:3]
    
    features = np.concatenate([hist_h, hist_s, hist_v, mean_color, [contour_area, contour_perimeter]])
    
    # Предсказание модели
    prediction = model.predict([features])[0]
    probs = model.predict_proba([features])[0]
    confidence = max(probs) * 100
    
    parts = prediction.split(" ")
    ripeness = "Спелый (" + parts[0] + ")" if parts[0] == "ripe" else "Незрелый (" + parts[0] + ")"
    fruit_name = " ".join(parts[1:]).capitalize()

    print(f"\n[РЕЗУЛЬТАТ] Фрукт: {fruit_name} | Статус: {ripeness} | Уверенность: {confidence:.1f}%")

    # ВИЗУАЛИЗАЦИЯ ПРОМЕЖУТОЧНЫХ ЭТАПОВ (Сравнение До / После / Маска / Контуры)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title("1. Оригинал (Resize)")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.title("2. Фильтрация (Gaussian Blur)")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("3. Сегментация (HSV Маска)")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(contour_image)
    plt.title(f"4. Контуры и Анализ\n{fruit_name} - {ripeness} ({confidence:.1f}%)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def select_and_predict():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title="Выберите изображение фрукта", filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")])
    root.destroy()
    
    if file_path:
        process_and_visualize(file_path)

if __name__ == "__main__":
    select_and_predict()