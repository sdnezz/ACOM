import numpy as np
import sys, os, cv2


def camshift(video_name: str, video_dimension: tuple):
    
    def manual_camshift(prob_image, window, max_iter=10, epsilon=1):
        """
        Ручная реализация CamShift на основе математической модели.
        
        Аргументы:
        - prob_image: карта вероятностей (dst из calcBackProject), 2D numpy array.
        - window: кортеж (x, y, w, h) - начальное окно.
        - max_iter: максимальное число итераций.
        - epsilon: порог сходимости (минимальный сдвиг для остановки).
        
        Возвращает:
        - ret: True, если сошелся.
        - new_window: обновленное окно (x, y, w, h).
        - angle: угол поворота bounding box.
        """
        x, y, w, h = window
        height, width = prob_image.shape
        
        # Сохраняем исходный размер для стабилизации
        original_area = w * h
        
        for iter_count in range(max_iter):
            # Обрезаем окно, чтобы не выходить за границы изображения
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            # Вырезаем ROI из карты вероятностей
            roi = prob_image[y:y + h, x:x + w]
            
            # Если сумма весов нулевая, останавливаемся
            total_weight = np.sum(roi)
            if total_weight <= 0:
                return False, (x, y, w, h), 0
            
            # Создаем сетку координат для ROI
            cols, rows = w, h
            local_y, local_x = np.mgrid[0:rows, 0:cols]
            global_x = x + local_x
            global_y = y + local_y
            
            # Вычисляем моменты
            m00 = total_weight
            m10 = np.sum(global_x * roi)
            m01 = np.sum(global_y * roi)
            
            # Центр масс
            center_x = m10 / m00
            center_y = m01 / m00
            
            # Вторые моменты для ориентации
            m20 = np.sum((global_x - center_x) ** 2 * roi)
            m02 = np.sum((global_y - center_y) ** 2 * roi)
            m11 = np.sum((global_x - center_x) * (global_y - center_y) * roi)
            
            # Вычисляем ориентацию (угол)
            if m02 > m20:
                num = m02 - m20 + np.sqrt((m02 - m20) ** 2 + 4 * m11 ** 2)
                den = 2 * m11
            else:
                num = 2 * m11
                den = m20 - m02 + np.sqrt((m20 - m02) ** 2 + 4 * m11 ** 2)
            
            if den != 0:
                angle = np.arctan(num / den)
            else:
                angle = 0
            
            # Вычисляем размеры окна на основе моментов
            a = m20 / m00
            b = m11 / m00  
            c = m02 / m00
            
            # Собственные значения ковариационной матрицы
            lambda1 = (a + c) / 2 + np.sqrt(((a - c) / 2) ** 2 + b ** 2)
            lambda2 = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + b ** 2)
            
            # Масштабируем размеры на основе собственных значений
            scale = 2.5  # Эмпирический коэффициент масштабирования
            new_w = int(np.sqrt(lambda1) * scale)
            new_h = int(np.sqrt(lambda2) * scale)
            
            # Стабилизируем размер (предотвращаем резкие изменения)
            area_ratio = (new_w * new_h) / original_area
            if area_ratio > 2.0 or area_ratio < 0.5:
                new_w = w
                new_h = h
            
            # Новые координаты (центрируем вокруг центра масс)
            new_x = int(center_x - new_w / 2)
            new_y = int(center_y - new_h / 2)
            
            # Проверяем сдвиг
            shift = abs(new_x - x) + abs(new_y - y) + abs(new_w - w) + abs(new_h - h)
            
            # Обновляем окно
            x, y, w, h = new_x, new_y, new_w, new_h
            
       
            if shift < epsilon:
                break
        

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        return True, (x, y, w, h), np.degrees(angle)
    
    # Основной код трекинга
    cap = cv2.VideoCapture(f"{video_name}.mp4")
    
    output_dir = "CAMSHIFT"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{video_name}-tracking.mp4")
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, video_dimension)
    
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать видео")
        return
    frame = cv2.resize(frame, video_dimension)
    bbox = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")
    
    roi = frame[int(bbox[1]):int(bbox[1] + bbox[3]), 
                int(bbox[0]):int(bbox[0] + bbox[2])]
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    max_iter = 10
    epsilon = 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, video_dimension)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        
        ret_manual, track_window, angle = manual_camshift(dst, bbox, max_iter, epsilon)
        
        if ret_manual:
            x, y, w, h = track_window
            bbox = (x, y, w, h)
            
            center = (int(x + w/2), int(y + h/2))
            size = (int(w/2), int(h/2))
            
            if size[0] > 0 and size[1] > 0:
                # Рисуем эллипс (аппроксимация повернутого прямоугольника)
                cv2.ellipse(frame, center, size, angle, 0, 360, (0, 255, 0), 2)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        out.write(frame)
        
        cv2.imshow('CamShift Tracking', frame)
        char = cv2.waitKey(30)
        if char == 27:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Пример использования
if __name__ == "__main__":
    video_name = "body"
    video_dimension = (1920, 1080)
    camshift(video_name, video_dimension)