import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

DATASET_PATH = r"C:\Users\Nurkanat\Desktop\HSV segmentation\Ripe & Unripe Fruits" # замените на ваш путь, если нужно

def extract_pure_cv_features(image_path):
    """
    Извлечение признаков с помощью исключительно OpenCV:
    1. HSV Гистограммы (цвет)
    2. Геометрия контура (округлость, пропорции, сплошность)
    3. Моменты Ху (инвариантные к форме дескрипторы, встроенные в cv2)
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image = cv2.resize(image, (250, 250))
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Сегментация по насыщенности для отделения от фона
    _, thresh = cv2.threshold(hsv[:, :, 1], 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if area < 100: # Отсекаем случайный мелкий мусор
        return None
        
    # Признаки формы
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0.0
    
    # Сплошность (Solidity) - отношение площади контура к площади его выпуклой оболочки
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    
    # Моменты Ху (встроены в OpenCV, идеально описывают форму объекта)
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Логарифмическое шкалирование для стабилизации малых чисел
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    # Цветовые гистограммы внутри маски
    fruit_only = cv2.bitwise_and(hsv, hsv, mask=mask)
    hist_h = cv2.calcHist([fruit_only], [0], mask, [24], [0, 180]).flatten()
    hist_s = cv2.calcHist([fruit_only], [1], mask, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([fruit_only], [2], mask, [16], [0, 256]).flatten()
    
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    mean_color = cv2.mean(hsv, mask=mask)[:3]
    
    # Объединяем всё в мощный вектор признаков
    features = np.concatenate([
        hist_h, hist_s, hist_v, 
        mean_color, 
        [circularity, aspect_ratio, solidity], 
        hu_moments
    ])
    return features

def load_dataset():
    X, y = [], []
    print(f"Сканирование датасета в: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print("Ошибка: Путь к датасету не найден!")
        return np.array(X), np.array(y)
        
    folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
    
    for folder in folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        images = os.listdir(folder_path)
        print(f"Обработка класса: {folder} ({len(images)} изобр.)")
        
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            features = extract_pure_cv_features(img_path)
            if features is not None:
                X.append(features)
                y.append(folder)
                
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_dataset()
    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("\nОбучение модели RandomForest...")
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print(f"\nТочность модели (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("\nОтчет классификации:")
        print(classification_report(y_test, y_pred))
        
        with open("ripeness_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Модель успешно обучена и сохранена в 'ripeness_model.pkl'!")
    else:
        print("Датасет пуст или изображения не были загружены.")