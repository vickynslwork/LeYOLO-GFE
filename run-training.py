from ultralytics import YOLO

# Load a model
model = YOLO("leyolosmall-pose.yaml")  # build a new model from scratch
#model = YOLO("weights/LeYOLONano.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(
   data='/media/aivision/Gamma/Vicky/coco_kpts.yaml',
   imgsz=640,
   epochs=100,
   batch=32,
   device = 0,
   name = "multi-train-2"
)
metrics = model.val(split= 'val')   
#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#path = model.export(format="onnx")  # export the model to ONNX format
