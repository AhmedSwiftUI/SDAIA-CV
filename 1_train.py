"""
الخطوة الأولى: تحميل Dataset من Roboflow وتدريب YOLOv8

قبل التشغيل:
  1. ثبّت المتطلبات:  pip install roboflow
  2. سجّل في roboflow.com (مجاني)
  3. احصل على API Key من: Account Settings → Copy API Key
  4. اذهب إلى universe.roboflow.com → ابحث عن "accident detection"
     واختر dataset → Versions → Export → YOLOv8 → Show Download Code
  5. عدّل config.py بالقيم الصحيحة (API Key, Workspace, Project, Version)

ثم شغّل:  python3 1_train.py
"""

import sys
from pathlib import Path
import torch

import config

def check_config():
    if config.ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ الرجاء تعديل config.py وإضافة ROBOFLOW_API_KEY")
        print("   احصل عليه من: roboflow.com → Account Settings → API Key")
        sys.exit(1)
    if config.ROBOFLOW_WORKSPACE == "YOUR_WORKSPACE":
        print("❌ الرجاء تعديل config.py وإضافة ROBOFLOW_WORKSPACE و ROBOFLOW_PROJECT")
        sys.exit(1)

def download_dataset():
    from roboflow import Roboflow
    print(f"📥 جاري الاتصال بـ Roboflow...")
    rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
    project = rf.workspace(config.ROBOFLOW_WORKSPACE).project(config.ROBOFLOW_PROJECT)
    print(f"📦 تحميل Dataset (الإصدار {config.ROBOFLOW_VERSION}) بصيغة YOLOv8...")
    dataset = project.version(config.ROBOFLOW_VERSION).download(
        "yolov8",
        location=str(config.DATA_DIR / "accident_dataset"),
    )
    print(f"✅ تم التحميل في: {dataset.location}")
    return Path(dataset.location)

def train(data_dir: Path):
    from ultralytics import YOLO

    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        # بعض datasets تضع data.yaml مباشرة في المجلد
        candidates = list(data_dir.rglob("data.yaml"))
        if not candidates:
            raise FileNotFoundError(f"لم يُعثر على data.yaml في: {data_dir}")
        data_yaml = candidates[0]

    print(f"\n📄 ملف البيانات: {data_yaml}")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"⚙️  جهاز التدريب: {device}")

    model = YOLO(config.BASE_MODEL)
    print(f"\n🚀 بدء التدريب — {config.EPOCHS} epoch...\n")

    model.train(
        data=str(data_yaml),
        epochs=config.EPOCHS,
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        device=device,
        project=str(config.MODELS_DIR),
        name="accident_detector",
        patience=10,
        save=True,
        plots=True,
        exist_ok=True,
    )

    best = config.MODELS_DIR / "accident_detector" / "weights" / "best.pt"
    print(f"\n✅ التدريب اكتمل!")
    print(f"📁 أفضل نموذج محفوظ في: {best}")
    return best

def validate(best_model: Path, data_dir: Path):
    from ultralytics import YOLO
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        data_yaml = list(data_dir.rglob("data.yaml"))[0]

    model = YOLO(str(best_model))
    print("\n📊 تقييم النموذج على مجموعة الاختبار...")
    metrics = model.val(data=str(data_yaml), verbose=False)
    print(f"   mAP@50:     {metrics.box.map50:.3f}")
    print(f"   mAP@50-95:  {metrics.box.map:.3f}")
    print(f"   Precision:  {metrics.box.mp:.3f}")
    print(f"   Recall:     {metrics.box.mr:.3f}")

if __name__ == "__main__":
    check_config()
    data_dir = download_dataset()
    best_model = train(data_dir)
    validate(best_model, data_dir)
    print("\n✅ جاهز! شغّل الآن:  python3 2_detect.py <مسار_الفيديو>")
