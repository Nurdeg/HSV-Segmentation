import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Ваш путь к папке с данными (буква r перед строкой нужна для Windows-путей)
DATASET_PATH = r"C:\Users\Nurkanat\Desktop\HSV segmentation\Ripe & Unripe Fruits"

def extract_hsv_features(image_path):
    """
    Извлекает улучшенные признаки: 
    гистограммы распределения цветов (H, S, V) + средние значения.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Уменьшаем картинку для ускорения работы
    image = cv2.resize(image, (200, 200))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 1. Считаем гистограммы для каждого канала (разбиваем на 16 корзин/bins)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    
    # 2. Нормализуем гистограммы (чтобы размер картинки не влиял)
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    # 3. Средние значения и стандартное отклонение
    means = cv2.mean(hsv)[:3]
    stds = [np.std(hsv[:, :, i]) for i in range(3)]
    
    # Объединяем всё в один длинный вектор признаков (массив чисел)
    features = np.concatenate([hist_h, hist_s, hist_v, means, stds])
    return features

def load_data():
    X = []
    y = []
    
    print(f"Сканирую папку: {DATASET_PATH}")
    
    # Проходим по всем папкам (ripe apple, unripe banana и т.д.)
    if not os.path.exists(DATASET_PATH):
        print("Ошибка: Папка не найдена. Проверьте путь!")
        return np.array(X), np.array(y)

    folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
    
    for folder_name in folders:
        folder_path = os.path.join(DATASET_PATH, folder_name)
        images = os.listdir(folder_path)
        
        print(f"Обработка класса '{folder_name}' ({len(images)} фото)...")
        
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            features = extract_hsv_features(img_path)
            
            if features is not None:
                X.append(features)
                y.append(folder_name) # Имя папки будет нашим классом (label)
                
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_data()
    
    if len(X) == 0:
        print("Ошибка: Не удалось загрузить картинки.")
    else:
        print(f"\nВсего загружено {len(X)} изображений.")
        
        # Делим данные (80% на обучение, 20% на проверку)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Обучаем модель (увеличили число деревьев до 200, т.к. классов теперь много)
        print("Обучение модели Random Forest...")
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Проверка точности
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nТочность модели: {acc * 100:.2f}%")
        
        # Сохраняем модель
        with open("ripeness_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Модель сохранена как 'ripeness_model.pkl'!")