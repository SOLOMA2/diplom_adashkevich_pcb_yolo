"""
Convert VOC_PCB dataset (Pascal VOC XML) to Ultralytics YOLO dataset layout.

Input layout (default):
  VOC_PCB/
    Annotations/*.xml
    JPEGImages/*.jpg
    ImageSets/Main/{train,val,test}.txt

Output layout (default):
  PCB_dataset/
    images/{trainPCB,valPCB,testPCB}/*.jpg
    labels/{trainPCB,valPCB,testPCB}/*.txt
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SPLIT_TO_DIR = {"train": "trainPCB", "val": "valPCB", "test": "testPCB"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", type=Path, default=Path("VOC_PCB"), help="VOC dataset root")
    parser.add_argument("--out-root", type=Path, default=Path("PCB_dataset"), help="YOLO dataset output root")
    parser.add_argument("--copy-images", action="store_true", help="copy images instead of hardlinking")
    parser.add_argument("--overwrite", action="store_true", help="delete output folder before conversion")
    return parser.parse_args()


def _safe_float(value: str) -> float:
    return float(value.strip())


def convert_xml_to_yolo(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path.name}")
    width = _safe_float(size.findtext("width", "0"))
    height = _safe_float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path.name}: {width}x{height}")

    labels: list[str] = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if class_name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class '{class_name}' in {xml_path.name}")
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = _safe_float(box.findtext("xmin", "0"))
        ymin = _safe_float(box.findtext("ymin", "0"))
        xmax = _safe_float(box.findtext("xmax", "0"))
        ymax = _safe_float(box.findtext("ymax", "0"))

        # Clamp and validate coordinates before YOLO normalization.
        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        labels.append(
            f"{CLASS_TO_ID[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return labels


def read_split_ids(voc_root: Path, split: str) -> list[str]:
    split_file = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file is missing: {split_file}")
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_output_dirs(out_root: Path, overwrite: bool) -> None:
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    for split_dir in SPLIT_TO_DIR.values():
        (out_root / "images" / split_dir).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_dir).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def convert(voc_root: Path, out_root: Path, copy_images: bool, overwrite: bool) -> None:
    prepare_output_dirs(out_root, overwrite=overwrite)

    total_images = 0
    total_objects = 0

    for split, split_dir in SPLIT_TO_DIR.items():
        ids = read_split_ids(voc_root, split)
        for stem in ids:
            xml_path = voc_root / "Annotations" / f"{stem}.xml"
            img_path_jpg = voc_root / "JPEGImages" / f"{stem}.jpg"
            img_path_JPG = voc_root / "JPEGImages" / f"{stem}.JPG"
            img_path = img_path_jpg if img_path_jpg.exists() else img_path_JPG
            if not xml_path.exists():
                raise FileNotFoundError(f"Annotation not found: {xml_path}")
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path_jpg} or {img_path_JPG}")

            labels = convert_xml_to_yolo(xml_path)
            out_img = out_root / "images" / split_dir / img_path.name
            out_lbl = out_root / "labels" / split_dir / f"{stem}.txt"
            link_or_copy(img_path, out_img, copy_images=copy_images)
            out_lbl.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

            total_images += 1
            total_objects += len(labels)

        print(f"{split}: {len(ids)} images converted")

    print(f"Done. Total images: {total_images}, total labeled objects: {total_objects}")
    print(f"YOLO dataset ready at: {out_root}")


def main() -> None:
    args = parse_args()
    convert(
        voc_root=args.voc_root.resolve(),
        out_root=args.out_root.resolve(),
        copy_images=args.copy_images,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
