from ultralytics import YOLO


model = YOLO("yolo11s-cls.pt")

results=model.train(
    data="/media/baran/Disk1/CVINonuders/CV-Odev4-Kayısı-Detect/dataset",
    epochs=10,
    project="results",
    imgsz=640


)