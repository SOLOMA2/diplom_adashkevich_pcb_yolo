import os
import random
import re
from collections import defaultdict
from pathlib import Path

TRAINVAL_PERCENT = 0.8
TRAIN_PERCENT_INSIDE_TRAINVAL = 0.6
SEED = 42

XML_DIR = Path("./Annotations")
SPLIT_DIR = Path("./ImageSets/Main")

PREFIX_PATTERN = re.compile(r"^(rotation_(90|270)_|l_)")
SUFFIX_PATTERN = re.compile(r"_(600|256)$")


def canonical_id(stem: str) -> str:
    name = stem
    while True:
        match = PREFIX_PATTERN.match(name)
        if not match:
            break
        name = name[match.end() :]
    return SUFFIX_PATTERN.sub("", name)


def main() -> None:
    xml_files = sorted([p for p in os.listdir(XML_DIR) if p.lower().endswith(".xml")])
    groups = defaultdict(list)
    for xml_name in xml_files:
        stem = xml_name[:-4]
        groups[canonical_id(stem)].append(stem)

    keys = list(groups.keys())
    rng = random.Random(SEED)
    rng.shuffle(keys)

    n_trainval = int(len(keys) * TRAINVAL_PERCENT)
    n_train = int(n_trainval * TRAIN_PERCENT_INSIDE_TRAINVAL)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_trainval])
    test_keys = set(keys[n_trainval:])

    train, val, test = [], [], []
    for key in train_keys:
        train.extend(groups[key])
    for key in val_keys:
        val.extend(groups[key])
    for key in test_keys:
        test.extend(groups[key])

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
    print(f"Unique groups: {len(keys)}")
    print(f"train: {len(train)}")
    print(f"val: {len(val)}")
    print(f"test: {len(test)}")


if __name__ == "__main__":
    main()
