import mediapipe as mp
import cv2
import numpy as np
from collections import deque
import time
import csv
import os
from datetime import datetime

# MediaPipe Face Mesh başlatma
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

# Yüz landmark indeksleri
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LEFT_BROW_TOP = 55
LEFT_BROW_BOTTOM = 65
RIGHT_BROW_TOP = 285
RIGHT_BROW_BOTTOM = 295
NOSE_TIP = 1

#dil değişimi için dictionary
EMOTION_TRANSLATIONS = {
    "Happy": "Mutlu",
    "Sad": "Uzgun",
    "Angry": "Kizgin",
    "Surprised": "Saskin",
    "Fear": "Korku",
    "Disgust": "Igrenme",
    "Contempt": "Kucumseme",
    "Neutral": "Notr"
}

EMOTION_COLORS = {
    "Happy": (0, 255, 0),      # Yeşil
    "Sad": (255, 0, 0),        # Mavi
    "Angry": (0, 0, 255),      # Kırmızı
    "Surprised": (0, 165, 255), # Turuncu
    "Fear": (128, 0, 128),     # Mor
    "Disgust": (0, 128, 0),    # Koyu yeşil
    "Contempt": (192, 192, 192), # Gri
    "Neutral": (255, 255, 255)  # Beyaz
}

# emotion history için deque (son 100 frame - daha uzun analiz için)
emotion_history = deque(maxlen=100)
emotion_timeline = deque(maxlen=200)  # time line için
fps_history = deque(maxlen=10)

# data logging için
data_log = []
recording = False
video_writer = None
screenshot_count = 0


def get_point(landmarks, index, frame_width, frame_height):
    """Landmark noktasını pixel koordinatlarına çevirir"""
    landmark = landmarks.landmark[index]
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    return np.array([x, y])


def calculate_eye_aspect_ratio(landmarks, eye_indices, frame_width, frame_height):
    """Göz açıklığını hesaplar (EAR - Eye Aspect Ratio)"""
    top = get_point(landmarks, eye_indices[0], frame_width, frame_height)
    bottom = get_point(landmarks, eye_indices[1], frame_width, frame_height)
    left = get_point(landmarks, eye_indices[2], frame_width, frame_height)
    right = get_point(landmarks, eye_indices[3], frame_width, frame_height)
    
    # Dikey mesafeler
    vertical1 = np.linalg.norm(top - bottom)
    vertical2 = np.linalg.norm(top - bottom)
    
    # Yatay mesafe
    horizontal = np.linalg.norm(left - right)
    
    # EAR hesaplama
    if horizontal > 0:
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
    else:
        ear = 0
    
    return ear


def calculate_mouth_aspect_ratio(landmarks, frame_width, frame_height):
    """Ağız açıklığını hesaplar (MAR - Mouth Aspect Ratio)"""
    top = get_point(landmarks, MOUTH_TOP, frame_width, frame_height)
    bottom = get_point(landmarks, MOUTH_BOTTOM, frame_width, frame_height)
    left = get_point(landmarks, MOUTH_LEFT, frame_width, frame_height)
    right = get_point(landmarks, MOUTH_RIGHT, frame_width, frame_height)
    
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    
    if horizontal > 0:
        mar = vertical / horizontal
    else:
        mar = 0
    
    return mar, horizontal


def calculate_brow_position(landmarks, frame_width, frame_height):
    """Kaş pozisyonunu hesaplar"""
    left_brow_top = get_point(landmarks, LEFT_BROW_TOP, frame_width, frame_height)
    left_brow_bottom = get_point(landmarks, LEFT_BROW_BOTTOM, frame_width, frame_height)
    left_eye_top = get_point(landmarks, LEFT_EYE_TOP, frame_width, frame_height)
    
    right_brow_top = get_point(landmarks, RIGHT_BROW_TOP, frame_width, frame_height)
    right_brow_bottom = get_point(landmarks, RIGHT_BROW_BOTTOM, frame_width, frame_height)
    right_eye_top = get_point(landmarks, RIGHT_EYE_TOP, frame_width, frame_height)
    
    # Kaş-göz mesafesi
    left_brow_eye_dist = np.linalg.norm(left_brow_bottom - left_eye_top)
    right_brow_eye_dist = np.linalg.norm(right_brow_bottom - right_eye_top)
    avg_brow_eye_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2
    
    # Kaş eğimi
    left_brow_slope = left_brow_top[1] - left_brow_bottom[1]
    right_brow_slope = right_brow_top[1] - right_brow_bottom[1]
    avg_brow_slope = (left_brow_slope + right_brow_slope) / 2
    
    return avg_brow_eye_dist, avg_brow_slope


def calculate_mouth_corners(landmarks, frame_width, frame_height):
    """Ağız köşelerinin pozisyonunu hesaplar"""
    left = get_point(landmarks, MOUTH_LEFT, frame_width, frame_height)
    right = get_point(landmarks, MOUTH_RIGHT, frame_width, frame_height)
    
    # Ağız köşelerinin yükseklik farkı (asimetri)
    corner_asymmetry = abs(left[1] - right[1])
    
    return corner_asymmetry


def detect_emotions(landmarks, frame_width, frame_height):
    """Gelişmiş duygu tanıma fonksiyonu"""
    # Özellik çıkarımı
    left_eye_ear = calculate_eye_aspect_ratio(
        landmarks, 
        [LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT],
        frame_width, frame_height
    )
    right_eye_ear = calculate_eye_aspect_ratio(
        landmarks,
        [RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT],
        frame_width, frame_height
    )
    avg_ear = (left_eye_ear + right_eye_ear) / 2
    
    mar, mouth_width = calculate_mouth_aspect_ratio(landmarks, frame_width, frame_height)
    brow_eye_dist, brow_slope = calculate_brow_position(landmarks, frame_width, frame_height)
    mouth_asymmetry = calculate_mouth_corners(landmarks, frame_width, frame_height)
    
    # Duygu skorları
    emotion_scores = {
        "Happy": 0.0,
        "Sad": 0.0,
        "Angry": 0.0,
        "Surprised": 0.0,
        "Fear": 0.0,
        "Disgust": 0.0,
        "Contempt": 0.0,
        "Neutral": 0.5  # neutral icin baslangic skoru verdim
    }
    
    # neutral: normal degerler (oncelikli kontrol)
    # normal goz acikligi, normal agiz, normal kas pozisyonu
    neutral_conditions = 0
    if 0.22 < avg_ear < 0.28:  # normal goz acikligi
        neutral_conditions += 1
        emotion_scores["Neutral"] += 0.15
    if 0.28 < mar < 0.38:  # nromal agiz acikligi
        neutral_conditions += 1
        emotion_scores["Neutral"] += 0.15
    if 16 < brow_eye_dist < 19:  # normal kas-goz mesafesi
        neutral_conditions += 1
        emotion_scores["Neutral"] += 0.15
    if -1 < brow_slope < 1:  # normal kas egimi
        neutral_conditions += 1
        emotion_scores["Neutral"] += 0.15
    if mouth_asymmetry < 3:  # simetrik agiz
        neutral_conditions += 1
        emotion_scores["Neutral"] += 0.15
    if 40 < mouth_width < 50:  # normal agiz genisligi
        neutral_conditions += 1
        emotion_scores["Neutral"] += 0.1
    
   
    if neutral_conditions >= 4:
        emotion_scores["Neutral"] += 0.2
    
    # Happy: belirgin agiz, yukarı kıvrılmış köşeler (daha yüksek threshold)
    if mouth_width > 60 and mar < 0.25:  
        emotion_scores["Happy"] += 0.3
    if mouth_asymmetry < 3 and mouth_width > 55:
        emotion_scores["Happy"] += 0.2
    if brow_slope < -3:  # Belirgin yükselmiş kaşlar
        emotion_scores["Happy"] += 0.2
    
    # Sad: BELİRGİN aşağı kıvrılmış ağız köşeleri, düşük kaşlar
    if mouth_asymmetry > 10:  # Daha belirgin asimetri
        emotion_scores["Sad"] += 0.3
    if brow_slope > 3:  # Belirgin düşük kaşlar
        emotion_scores["Sad"] += 0.3
    if mar > 0.45 and mouth_width < 35:
        emotion_scores["Sad"] += 0.2
    
    # Angry: Çatık kaşlar, dar gözler
    if brow_eye_dist < 14:  # Daha yakın kaş-göz mesafesi
        emotion_scores["Angry"] += 0.3
    if avg_ear < 0.18:  # Daha dar gözler
        emotion_scores["Angry"] += 0.3
    if brow_slope < -4:  # Belirgin çatık kaşlar
        emotion_scores["Angry"] += 0.2
    
    # Surprised: Açık gözler, açık ağız, yükselmiş kaşlar
    if avg_ear > 0.32:  # Daha açık gözler
        emotion_scores["Surprised"] += 0.3
    if mar > 0.55:  # Daha açık ağız
        emotion_scores["Surprised"] += 0.3
    if brow_eye_dist > 22:  # Daha yükselmiş kaşlar
        emotion_scores["Surprised"] += 0.2
    
    # Fear: Açık gözler, açık ağız, düşük kaşlar
    if avg_ear > 0.32:
        emotion_scores["Fear"] += 0.25
    if mar > 0.45:
        emotion_scores["Fear"] += 0.25
    if brow_eye_dist > 20 and brow_slope > 2:
        emotion_scores["Fear"] += 0.25
    
    # Disgust: Dar ağız, dar gözler
    if mar < 0.22 and mouth_width < 32:
        emotion_scores["Disgust"] += 0.3
    if avg_ear < 0.22:
        emotion_scores["Disgust"] += 0.3
    
    # Contempt: Asimetrik ağız, hafif yükselmiş köşe
    if 4 < mouth_asymmetry < 9:
        emotion_scores["Contempt"] += 0.4
    if brow_slope < -2:
        emotion_scores["Contempt"] += 0.2
    
    max_emotion = max(emotion_scores, key=emotion_scores.get)
    max_score = emotion_scores[max_emotion]
    
    if max_score < 0.4 or (emotion_scores["Neutral"] >= 0.6 and max_score - emotion_scores["Neutral"] < 0.15):
        return "Neutral", emotion_scores
    else:
        return max_emotion, emotion_scores


def draw_emotion_stats(frame, emotion_history, fps):
    """Ekran üzerine istatistikleri çizer"""
    h, w = frame.shape[:2]
    
    if emotion_history:
        emotion_counts = {}
        for emo in emotion_history:
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        
        y_offset = 30
        cv2.putText(frame, "Duygu Istatistikleri:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(emotion_history)) * 100
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            emotion_tr = EMOTION_TRANSLATIONS.get(emotion, emotion)
            text = f"{emotion_tr}: {percentage:.1f}%"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
    
    # FPS ve kayıt durumu
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Kayıt durumu
    if recording:
        cv2.putText(frame, "KAYIT: ACIK", (w - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Klavye kısayolları
    y_help = h - 100
    cv2.putText(frame, "Kisayollar:", (10, y_help), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "S: Ekran goruntusu | R: Video kayit | C: CSV kaydet | Q: Cikis", 
               (10, y_help + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def draw_emotion_bars(frame, emotion_history):
    """Duygu yoğunluğu çubuk grafiği çizer"""
    if not emotion_history:
        return
    
    h, w = frame.shape[:2]
    bar_width = 15
    bar_spacing = 5
    start_x = w - 200
    start_y = 100
    max_height = 80
    
    emotion_counts = {}
    for emo in emotion_history:
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
    
    total = len(emotion_history)
    if total == 0:
        return
    
    x_pos = start_x
    for emotion in EMOTION_COLORS.keys():
        count = emotion_counts.get(emotion, 0)
        height = int((count / total) * max_height)
        color = EMOTION_COLORS[emotion]
        
        # Çubuk çiz
        cv2.rectangle(frame, 
                     (x_pos, start_y + max_height - height),
                     (x_pos + bar_width, start_y + max_height),
                     color, -1)
        
        # Çerçeve
        cv2.rectangle(frame,
                     (x_pos, start_y),
                     (x_pos + bar_width, start_y + max_height),
                     (100, 100, 100), 1)
        
        x_pos += bar_width + bar_spacing


def save_screenshot(frame):
    """Ekran görüntüsü kaydeder"""
    global screenshot_count
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/ekran_goruntusu_{timestamp}_{screenshot_count}.jpg"
    cv2.imwrite(filename, frame)
    screenshot_count += 1
    print(f"Ekran goruntusu kaydedildi: {filename}")


def save_to_csv():
    """Duygu verilerini CSV dosyasına kaydeder"""
    if not data_log:
        print("Kaydedilecek veri yok!")
        return
    
    if not os.path.exists("veriler"):
        os.makedirs("veriler")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"veriler/duygu_verileri_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Zaman', 'Duygu', 'Güven Skoru'])
        
        for entry in data_log:
            writer.writerow(entry)
    
    print(f"Veriler kaydedildi: {filename} ({len(data_log)} kayit)")


def start_video_recording(frame):
    """Video kaydını başlatır"""
    global video_writer, recording
    if not os.path.exists("videolar"):
        os.makedirs("videolar")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"videolar/duygu_kaydi_{timestamp}.mp4"
    
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
    recording = True
    print(f"Video kaydi basladi: {filename}")


def stop_video_recording():
    """Video kaydını durdurur"""
    global video_writer, recording
    if video_writer:
        video_writer.release()
        video_writer = None
        recording = False
        print("Video kaydi durduruldu.")


def draw_emotion_info(frame, emotion, confidence, x, y, w, h):
    """Yüz etrafına duygu bilgisini çizer"""
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    emotion_tr = EMOTION_TRANSLATIONS.get(emotion, emotion)
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    text = f"{emotion_tr} ({confidence:.0%})"
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    
    cv2.rectangle(
        frame,
        (x, y - text_height - 10),
        (x + text_width, y),
        color,
        -1
    )
    
    
    cv2.putText(
        frame,
        text,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )


prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    fps_history.append(fps)
    avg_fps = np.mean(fps_history) if fps_history else fps
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    h, w, _ = frame.shape
    
    if results.multi_face_landmarks:
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            emotion, scores = detect_emotions(face_landmarks, w, h)
            confidence = scores[emotion]
            
            # data kaydı
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            data_log.append([timestamp, emotion, f"{confidence:.2f}"])
            emotion_timeline.append(emotion)
            
            emotion_history.append(emotion)
            
            x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            draw_emotion_info(
                frame, emotion, confidence,
                x_min - 10, y_min - 50,
                x_max - x_min + 20, y_max - y_min + 20
            )
    else:
        if len(emotion_history) > 0:
            emotion_history.clear()
    
    # istatistikler & grafikler
    draw_emotion_stats(frame, emotion_history, avg_fps)
    draw_emotion_bars(frame, emotion_history)
    
    # video recording
    if recording and video_writer:
        video_writer.write(frame)
    
    cv2.imshow("Gelismis Duygu Tanima", frame)
    
    # Klavye kontrolleri
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        save_screenshot(frame)
    elif key == ord("r"):
        if recording:
            stop_video_recording()
        else:
            start_video_recording(frame)
    elif key == ord("c"):
        save_to_csv()

# clean up
if recording:
    stop_video_recording()

if data_log:
    print(f"\nToplam {len(data_log)} veri kaydi toplandi.")
    save_choice = input("Verileri CSV dosyasina kaydetmek ister misiniz? (e/h): ")
    if save_choice.lower() == 'e':
        save_to_csv()

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("Program sonlandirildi.")
