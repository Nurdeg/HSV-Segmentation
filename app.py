import streamlit as st
import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Определение зрелости фруктов (OpenCV)",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎🍌 Определение зрелости фруктов и овощей")
st.markdown("Пайплайн на чистом OpenCV: **Фильтрация -> HSV Сегментация -> Анализ формы и цвета -> RandomForest**")

@st.cache_resource
def load_model():
    model_path = "ripeness_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.warning("⚠️ Файл модели `ripeness_model.pkl` не найден! Сначала запустите `py train.py` в терминале.")
else:
    st.success("✅ Модель успешно загружена!")

st.markdown("---")
uploaded_file = st.file_uploader("Загрузите изображение фрукта/овоща", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is not None:
        image_resized = cv2.resize(image, (250, 250))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        st.image(image_rgb, caption="Загруженное изображение", use_container_width=True)
        
        if st.button("🔍 Запустить точный анализ", type="primary"):
            if model is None:
                st.error("Сначала обучите модель!")
            else:
                with st.spinner("Анализ геометрии, контуров и цвета..."):
                    # 1. Фильтрация
                    blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)
                    
                    # 2. Сегментация
                    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                    _, thresh = cv2.threshold(hsv[:, :, 1], 25, 255, cv2.THRESH_BINARY)
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
                    # 3. Контурный анализ
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    contour_image = image_rgb.copy()
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
                        area = cv2.contourArea(largest_contour)
                        perimeter = cv2.arcLength(largest_contour, True)
                        
                        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        aspect_ratio = float(w) / h if h > 0 else 0.0
                        
                        hull = cv2.convexHull(largest_contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area) / hull_area if hull_area > 0 else 0.0
                        
                        moments = cv2.moments(largest_contour)
                        hu_moments = cv2.HuMoments(moments).flatten()
                        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
                    else:
                        circularity, aspect_ratio, solidity = 0.0, 0.0, 0.0
                        hu_moments = np.zeros(7)
                        
                    # Извлечение признаков
                    fruit_only = cv2.bitwise_and(hsv, hsv, mask=mask)
                    hist_h = cv2.calcHist([fruit_only], [0], mask, [24], [0, 180]).flatten()
                    hist_s = cv2.calcHist([fruit_only], [1], mask, [16], [0, 256]).flatten()
                    hist_v = cv2.calcHist([fruit_only], [2], mask, [16], [0, 256]).flatten()
                    cv2.normalize(hist_h, hist_h)
                    cv2.normalize(hist_s, hist_s)
                    cv2.normalize(hist_v, hist_v)
                    mean_color = cv2.mean(hsv, mask=mask)[:3]
                    
                    features = np.concatenate([
                        hist_h, hist_s, hist_v, 
                        mean_color, 
                        [circularity, aspect_ratio, solidity], 
                        hu_moments
                    ])
                    
                    # 4. Классификация
                    prediction = model.predict([features])[0]
                    probs = model.predict_proba([features])[0]
                    confidence = max(probs) * 100
                    
                    parts = prediction.split(" ")
                    ripeness_raw = parts[0]
                    fruit_name = " ".join(parts[1:]).capitalize()
                    ripeness_status = "Спелый (Ripe)" if ripeness_raw == "ripe" else "Незрелый (Unripe)"
                
                # Вывод результатов
                st.markdown("---")
                st.subheader("📊 Результаты классификации")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Распознанный фрукт", value=fruit_name)
                with col2:
                    st.metric(label="Статус зрелости", value=ripeness_status, delta=f"{confidence:.1f}% уверенности")
                
                st.markdown("### 🔬 Визуализация этапов обработки")
                fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
                
                axs[0, 0].imshow(image_rgb)
                axs[0, 0].set_title("1. Оригинал (Resize)", fontsize=10)
                axs[0, 0].axis("off")
                
                axs[0, 1].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
                axs[0, 1].set_title("2. Фильтрация (Gaussian Blur)", fontsize=10)
                axs[0, 1].axis("off")
                
                axs[1, 0].imshow(mask, cmap="gray")
                axs[1, 0].set_title("3. Сегментация (HSV Mask)", fontsize=10)
                axs[1, 0].axis("off")
                
                axs[1, 1].imshow(contour_image)
                axs[1, 1].set_title(f"4. Контуры и Форма\n{fruit_name} ({confidence:.1f}%)", fontsize=10)
                axs[1, 1].axis("off")
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)