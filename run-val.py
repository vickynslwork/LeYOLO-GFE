from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/home/aivision/Documents/realtimevicky/Tuning Channel-Tuning GCNet 8 Train Small-eff-2/LeYOLO-main/runs/pose/leyolo_/weights/best.pt")  # load a pretrained model (recommended for training)

metrics = model.val(split= 'val', data = '/home/aivision/Documents/realtimevicky/Data/coco_kpts.yaml')  # evaluate model performance on the validation set
#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#path = model.export(format="onnx")  # export the model to ONNX format
