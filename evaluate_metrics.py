import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
import cv2
from sklearn.model_selection import train_test_split

# Импортируем вашу функцию извлечения признаков (из train.py)
from train import load_dataset

def plot_confusion_matrix():
    # 1. Загружаем данные и модель
    X, y = load_dataset()
    if len(X) == 0:
        print("Датасет пуст!")
        return
        
    with open("ripeness_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Предсказания на тесте
    y_pred = model.predict(X_test)
    
    # 3. Строим матрицу ошибок
    labels = sorted(list(set(y)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # 4. Визуализация через Seaborn и Matplotlib (высокое качество dpi=300)
    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.title("Матрица ошибок (Confusion Matrix) классификатора", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Предсказанные классы", fontsize=12)
    plt.ylabel("Истинные классы (True labels)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Сохраняем картинку, чтобы вставить в отчет Word
    plt.savefig("confusion_matrix_report.png")
    print("Матрица ошибок успешно сохранена как 'confusion_matrix_report.png'!")
    
    plt.show()

if __name__ == "__main__":
    plot_confusion_matrix()