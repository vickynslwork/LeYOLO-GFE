import cv2
import os
import glob
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

# Path model YOLO Pose
model_path = "/home/aivision/Documents/realtimevicky/Tuning Channel-Tuning GCNet 8 Train Small-eff-2/LeYOLO-main/runs/pose/leyolo_/weights/best.pt"

# Folder input gambar
input_folder = "/home/aivision/Documents/realtimevicky/Data/images"

# Folder output hasil prediksi
output_folder = "output_pose"

# Confidence threshold
conf_threshold = 0.25

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(model_path)

# Buat folder output jika belum tersedia
os.makedirs(output_folder, exist_ok=True)

# ============================================================
# GET ALL IMAGES
# ============================================================

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

image_files = []

for ext in image_extensions:
    image_files.extend(
        glob.glob(os.path.join(input_folder, ext))
    )
    image_files.extend(
        glob.glob(os.path.join(input_folder, ext.upper()))
    )

print(f"Total images: {len(image_files)}")

# ============================================================
# INFERENCE
# ============================================================

for image_path in image_files:

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Cannot read image: {image_path}")
        continue

    # YOLO Pose prediction
    results = model.predict(
        source=image,
        conf=conf_threshold,
        verbose=False
    )

    # Draw bounding box and keypoints
    annotated_image = results[0].plot()

    # Output filename
    filename = os.path.basename(image_path)

    output_path = os.path.join(
        output_folder,
        filename
    )

    # Save image
    cv2.imwrite(output_path, annotated_image)

    print(f"Processed: {filename}")

# ============================================================
# FINISHED
# ============================================================

print("\n===================================")
print("POSE PREDICTION COMPLETED")
print(f"Output folder: {output_folder}")
print("===================================")
