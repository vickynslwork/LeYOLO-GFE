from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/media/aivision/Gamma/Vicky/Tuning Channel-Tuning GCNet 8 Train Small-eff-2/LeYOLO-main/runs/pose/multi-train-1/weights/last.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model.resume = True 

# Use the model
results = model.train(
   data='/media/aivision/Gamma/Vicky/coco_kpts.yaml',
   imgsz=640,
   epochs=100,
   batch=32,
   device='0',
   workers=2,
   name='leyolo_',
   resume=True
)

metrics = model.val(split= 'val')  
#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#path = model.export(format="onnx")  # export the model to ONNX format
