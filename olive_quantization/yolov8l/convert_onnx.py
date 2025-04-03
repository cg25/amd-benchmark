from ultralytics import YOLO

model = YOLO("./yolov8l.pt")

model.export(
    format="onnx",
    imgsz=(640, 640),
    opset=17,
    simplify=True,
    dynamic=False,
)