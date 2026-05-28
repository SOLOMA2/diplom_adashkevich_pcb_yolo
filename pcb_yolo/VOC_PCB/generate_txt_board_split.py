import os
import random
import re
from collections import defaultdict
from pathlib import Path

# Split by "board_key" to avoid context leakage:
# all crops/augmentations belonging to the same board_key go to one split only.
#
# Example stem:
#   rotation_90_light_08_mouse_bite_05_3_600
# board_key becomes:
#   light_08_mouse_bite_05
#
# Ratios match the original generator roughly:
# train ~= 0.48, val ~= 0.32, test ~= 0.20

TRAINVAL_PERCENT = 0.8
TRAIN_PERCENT_INSIDE_TRAINVAL = 0.6
SEED = 42

XML_DIR = Path("./Annotations")
SPLIT_DIR = Path("./ImageSets/Main")

PREFIX_PATTERN = re.compile(r"^(rotation_(90|270)_|l_)")
SUFFIX_PATTERN = re.compile(r"_(600|256)$")


def strip_aug(name: str) -> str:
    n = name
    while True:
        m = PREFIX_PATTERN.match(n)
        if not m:
            break
        n = n[m.end() :]
    return n


def canonical_id(stem: str) -> str:
    """Remove augmentation prefixes and size suffix to group identical base images."""
    n = strip_aug(stem)
    return SUFFIX_PATTERN.sub("", n)


def board_key(stem: str) -> str:
    """Coarser key to group multiple crops from the same scene/board."""
    base = canonical_id(stem)
    parts = base.split("_")
    if len(parts) >= 2:
        parts = parts[:-1]  # drop last crop index
    return "_".join(parts)


def main() -> None:
    xml_files = sorted([p for p in os.listdir(XML_DIR) if p.lower().endswith(".xml")])
    if not xml_files:
        raise SystemExit(f"No XML found in: {XML_DIR.resolve()}")

    groups = defaultdict(list)
    for xml_name in xml_files:
        stem = xml_name[:-4]
        groups[board_key(stem)].append(stem)

    keys = list(groups.keys())
    rng = random.Random(SEED)
    rng.shuffle(keys)

    n_trainval = int(len(keys) * TRAINVAL_PERCENT)
    n_train = int(n_trainval * TRAIN_PERCENT_INSIDE_TRAINVAL)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_trainval])
    test_keys = set(keys[n_trainval:])

    train, val, test = [], [], []
    for k in train_keys:
        train.extend(groups[k])
    for k in val_keys:
        val.extend(groups[k])
    for k in test_keys:
        test.extend(groups[k])

    train.sort()
    val.sort()
    test.sort()
    trainval = sorted(train + val)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (SPLIT_DIR / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    (SPLIT_DIR / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    (SPLIT_DIR / "trainval.txt").write_text("\n".join(trainval) + "\n", encoding="utf-8")

    print(f"Total XML: {len(xml_files)}")
    print(f"Unique board_keys: {len(keys)}")
    print(f"train: {len(train)}")
    print(f"val: {len(val)}")
    print(f"test: {len(test)}")


if __name__ == "__main__":
    main()

