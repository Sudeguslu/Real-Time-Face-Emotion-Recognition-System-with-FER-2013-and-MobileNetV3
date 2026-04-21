import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import numpy as np
from collections import deque


# ──────────────────────────────────────────────
# Temporal smoother + histerez
# Son N karenin olasılık ortalamasını alır.
# Yeni etiket ancak mevcut etiketin CHANGE_THRESH kadar üzerine çıkarsa değişir.
# ──────────────────────────────────────────────
class EmotionSmoother:
    def __init__(self, window=10, thresh=0.15):
        self.thresh       = thresh          # etiket değişimi için gereken minimum fark
        self.history      = deque(maxlen=window)
        self.stable_label = None
        self.stable_score = 0.0

    def update(self, probs, emotions):
        self.history.append(np.array(probs, dtype=np.float32))
        smoothed   = np.mean(self.history, axis=0)   # son N karenin ortalaması
        best_idx   = int(np.argmax(smoothed))
        best_score = float(smoothed[best_idx])
        best_label = emotions[best_idx]

        if self.stable_label is None:
            # ilk kare: direkt ata
            self.stable_label = best_label
            self.stable_score = best_score
        else:
            cur_score = float(smoothed[emotions.index(self.stable_label)])
            # yeni etiket ancak mevcut etiketten THRESH kadar daha güçlüyse değişir
            if best_score > cur_score + self.thresh:
                self.stable_label = best_label
                self.stable_score = best_score
            else:
                self.stable_score = cur_score   # skoru güncelle, etiketi sabitle

        return self.stable_label, self.stable_score


def main():
    print("Sistem başlatılıyor, lütfen bekleyin...")

    # 1. AYARLAR VE ETİKETLER
    EMOTIONS    = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Notr', 'Uzgun', 'Saskin']
    MODEL_PATH  = 'best_emotion_mobilenetv3.pth'
    TFLITE_PATH = 'blaze_face_short_range.tflite'

    # MediaPipe .tflite modeli yoksa indir
    if not os.path.exists(TFLITE_PATH):
        print("MediaPipe yüz tespit modeli indiriliyor...")
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/face_detector/'
            'blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
            TFLITE_PATH
        )
        print("İndirildi.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan donanım: {device}")

    # 2. MODELİ YÜKLE
    print("Duygu modeli yükleniyor...")
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 7)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device).eval()

    # 3. GÖRÜNTÜ DÖNÜŞÜMLERİ — test.py ile birebir aynı
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 4. MEDİAPİPE — yeni tasks API
    print("MediaPipe yükleniyor...")
    base_options     = mp_python.BaseOptions(model_asset_path=TFLITE_PATH)
    detector_options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.6,
    )
    face_detector = mp_vision.FaceDetector.create_from_options(detector_options)

    # 5. KAMERA
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı.")
    print("\nKamera açıldı! Çıkmak için 'q' tuşuna bas.")

    # Her tespit edilen yüz için ayrı smoother (basit: maks 4 yüz)
    smoothers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kameradan görüntü alınamadı!")
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = face_detector.detect(mp_image)

        detections = result.detections if result.detections else []

        # Smoother sayısını yüz sayısına göre ayarla
        while len(smoothers) < len(detections):
            smoothers.append(EmotionSmoother(window=10, thresh=0.10))
        if len(smoothers) > len(detections):
            smoothers = smoothers[:len(detections)]

        for i, detection in enumerate(detections):
            bb = detection.bounding_box   # piksel cinsinden (origin_x, origin_y, width, height)

            pad_x = int(bb.width  * 0.10)
            pad_y = int(bb.height * 0.10)

            x1 = max(0, bb.origin_x - pad_x)
            y1 = max(0, bb.origin_y - pad_y)
            x2 = min(w, bb.origin_x + bb.width  + pad_x)
            y2 = min(h, bb.origin_y + bb.height + pad_y)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # Ham tahmin (her kare)
            pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            tensor  = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].tolist()

            # Smoothing + histerez → kararlı etiket
            label, score = smoothers[i].update(probs, EMOTIONS)

            # Çizim
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label}  {score*100:.0f}%",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
            )

        cv2.imshow("Duygu Analizi  |  [Q] Cikis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()


if __name__ == '__main__':
    main()