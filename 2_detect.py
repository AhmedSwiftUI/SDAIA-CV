"""
الخطوة الثانية: تشغيل نموذج الكشف على فيديو الداش كام

الاستخدام:
  python3 2_detect.py videos/accident_video.mp4

ما يفعله:
  1. يقرأ الفيديو إطاراً بإطار
  2. يشغّل نموذج YOLOv8 المدرّب على كل إطار
  3. عند رصد حادث لـ {FRAMES_TO_CONFIRM} إطارات متتالية:
       → يحفظ صورة لحظة الحادث
       → ينشئ خريطة تفاعلية مع دائرة تنبيه 1 كم
       → يرسل إشعارات للإسعاف والمرور
  4. يحفظ فيديو مُعلَّم بنتائج الكشف

شغّل 1_train.py أولاً للحصول على النموذج.
"""

import sys
import time
from pathlib import Path

import cv2

import config
from utils.map_utils import create_accident_map
from utils.notify_utils import send_alert


def _find_model() -> Path:
    best = config.MODELS_DIR / "accident_detector" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(
            f"النموذج غير موجود في:\n  {best}\n"
            "شغّل أولاً:  python3 1_train.py"
        )
    return best


def _get_video_path(args) -> str:
    if len(args) >= 2:
        return args[1]
    videos = sorted(
        list(config.VIDEOS_DIR.glob("*.mp4"))
        + list(config.VIDEOS_DIR.glob("*.avi"))
        + list(config.VIDEOS_DIR.glob("*.mov"))
    )
    if videos:
        print(f"📹 الفيديو المُختار تلقائياً: {videos[0].name}")
        return str(videos[0])
    print(f"❌ ضع الفيديو في المجلد: {config.VIDEOS_DIR}")
    print("   ثم شغّل: python3 2_detect.py videos/اسم_الفيديو.mp4")
    sys.exit(1)


def _draw_overlay(frame, accident_confirmed: bool, consecutive: int, conf_needed: int):
    h, w = frame.shape[:2]

    # شريط حالة علوي شفاف
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if accident_confirmed:
        # خلفية حمراء وامضة
        cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 200), -1)
        cv2.putText(frame, "!!! ACCIDENT DETECTED — ALERT SENT !!!",
                    (12, 38), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
    else:
        bar_w = int((consecutive / conf_needed) * (w - 24))
        cv2.rectangle(frame, (12, 32), (w - 12, 48), (60, 60, 60), -1)
        cv2.rectangle(frame, (12, 32), (12 + bar_w, 48), (0, 200, 255), -1)
        label = f"Scanning... ({consecutive}/{conf_needed})"
        cv2.putText(frame, label, (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    return frame


def process_video(video_path: str):
    from ultralytics import YOLO

    model_path = _find_model()
    print(f"✅ النموذج: {model_path}")

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"تعذّر فتح الفيديو: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = Path(video_path).stem
    out_path = config.OUTPUT_DIR / f"{stem}_detected.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    print(f"🎥 الفيديو: {Path(video_path).name}  ({width}×{height} @ {fps:.0f}fps, {total} إطار)")
    print(f"📁 النتيجة: {out_path}\n")

    consecutive   = 0
    alert_sent    = False
    alert_time    = None
    frame_num     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # تشغيل YOLO
        results = model.predict(frame, conf=config.CONF_THRESHOLD, verbose=False)[0]
        detected_now = len(results.boxes) > 0

        if detected_now:
            consecutive += 1
        else:
            consecutive = 0

        # تأكيد الحادث وإرسال الإنذار (مرة واحدة فقط)
        if consecutive >= config.FRAMES_TO_CONFIRM and not alert_sent:
            alert_sent = True
            alert_time = frame_num / fps

            print(f"🚨 رُصد حادث عند الثانية {alert_time:.1f}")

            # حفظ صورة لحظة الحادث
            frame_img_path = str(config.OUTPUT_DIR / "accident_frame.jpg")
            cv2.imwrite(frame_img_path, frame)
            print(f"📸 صورة الحادث: {frame_img_path}")

            ts = time.strftime("%Y-%m-%d %H:%M:%S")

            # إنشاء الخريطة
            map_path = create_accident_map(
                lat=config.DEMO_GPS_LAT,
                lon=config.DEMO_GPS_LON,
                location_name=config.LOCATION_NAME,
                timestamp=ts,
            )

            # إرسال الإشعارات
            send_alert(
                lat=config.DEMO_GPS_LAT,
                lon=config.DEMO_GPS_LON,
                location_name=config.LOCATION_NAME,
                timestamp=ts,
                map_path=map_path,
            )

        # رسم نتائج YOLO على الإطار
        annotated = results.plot()

        # رسم شريط الحالة
        annotated = _draw_overlay(
            annotated, alert_sent, consecutive, config.FRAMES_TO_CONFIRM
        )

        writer.write(annotated)

        if frame_num % 60 == 0:
            pct = (frame_num / total * 100) if total > 0 else 0
            print(f"  [{pct:5.1f}%] إطار {frame_num}/{total}", end="\r")

    cap.release()
    writer.release()

    print(f"\n\n✅ اكتملت المعالجة!")
    print(f"📁 الفيديو المُعلَّم: {out_path}")

    if alert_sent:
        print(f"🚨 تم رصد حادث في الثانية {alert_time:.1f}")
        print(f"🗺️  الخريطة: {config.OUTPUT_DIR / 'accident_map.html'}")
        print(f"📸 صورة الحادث: {config.OUTPUT_DIR / 'accident_frame.jpg'}")
    else:
        print("ℹ️  لم يُرصد أي حادث — جرّب فيديو آخر أو خفّض CONF_THRESHOLD في config.py")


if __name__ == "__main__":
    video = _get_video_path(sys.argv)
    process_video(video)
