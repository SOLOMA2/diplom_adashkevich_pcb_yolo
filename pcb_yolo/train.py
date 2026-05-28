from __future__ import annotations

import argparse
from pathlib import Path


def _resolve_imgsz(img_size: list[int]) -> int:
    if not img_size:
        return 608
    return int(img_size[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo12n.pt", help="initial checkpoint path")
    parser.add_argument("--cfg", type=str, default="", help="unused in YOLO12 wrapper (kept for compatibility)")
    parser.add_argument("--data", type=str, default="data/PCB_yolo12.yaml", help="dataset yaml path")
    parser.add_argument("--hyp", type=str, default="", help="optional hyperparameters yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size")
    parser.add_argument("--img-size", nargs="+", type=int, default=[608, 608], help="[train, val] image sizes")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume from last checkpoint")
    parser.add_argument("--nosave", action="store_true", help="do not save checkpoints")
    parser.add_argument("--device", default="", help="cuda device, e.g. 0 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="train as single class")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok")
    parser.add_argument("--optimizer", type=str, default="auto", help="optimizer: auto|SGD|Adam|AdamW")
    opt = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Ultralytics is not installed. Run: pip install -r requirements.txt") from e

    model = YOLO(opt.weights)
    train_kwargs = {
        "data": opt.data,
        "epochs": opt.epochs,
        "imgsz": _resolve_imgsz(opt.img_size),
        "batch": opt.batch_size,
        "device": opt.device,
        "workers": opt.workers,
        "project": opt.project,
        "name": opt.name,
        "exist_ok": opt.exist_ok,
        "single_cls": opt.single_cls,
        "save": not opt.nosave,
        "optimizer": opt.optimizer,
    }
    if opt.hyp:
        train_kwargs["hyp"] = opt.hyp
    if opt.resume:
        train_kwargs["resume"] = True

    result = model.train(**train_kwargs)
    save_dir = getattr(result, "save_dir", None)
    if save_dir:
        print(f"Training completed. Results saved to: {Path(save_dir)}")
    else:
        print("Training completed.")


if __name__ == "__main__":
    main()
