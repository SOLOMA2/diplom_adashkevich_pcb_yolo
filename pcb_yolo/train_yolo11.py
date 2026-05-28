from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train-6/weights/last.pt')
    model.train(
    data='data/PCB_yolo12.yaml',
    epochs=60,              
    imgsz=640,
    batch=16,
    workers=4,
    cache='ram',
    device=0,
    mosaic=0.0,
    cos_lr=True,
    lr0=0.0005,
    warmup_epochs=2,
    close_mosaic=0,         
    weight_decay=0.0005,
    name='train-6-colab'
    )