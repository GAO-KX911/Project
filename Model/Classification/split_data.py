import argparse
import os
import random
from shutil import copy2, rmtree


CLASS_MAPPING = {
    "fire": "abnormal",
    "smoke": "abnormal",
    "no_smokefire": "neutral",
}

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


def list_images(folder_path: str):
    return sorted(
        file_name
        for file_name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file_name))
        and os.path.splitext(file_name)[1].lower() in VALID_IMAGE_EXTS
    )


def build_split_dirs(output_root: str):
    target_classes = sorted(set(CLASS_MAPPING.values()))
    for split_name in ("Train", "Val", "Test"):
        split_root = os.path.join(output_root, split_name)
        mk_file(split_root)
        for class_name in target_classes:
            os.makedirs(os.path.join(split_root, class_name), exist_ok=True)


def copy_images(file_names, source_dir, target_dir):
    for file_name in file_names:
        copy2(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))


def merge_train_val(source_ori_root: str, output_root: str, split_rate: float):
    for source_class, target_class in CLASS_MAPPING.items():
        source_dir = os.path.join(source_ori_root, source_class)
        assert os.path.isdir(source_dir), f"missing source class directory: {source_dir}"

        images = list_images(source_dir)
        num_images = len(images)
        assert num_images > 0, f"no images found in {source_dir}"

        val_count = int(num_images * split_rate)
        val_images = set(random.sample(images, k=val_count))
        train_images = [image for image in images if image not in val_images]

        train_target = os.path.join(output_root, "Train", target_class)
        val_target = os.path.join(output_root, "Val", target_class)
        copy_images(train_images, source_dir, train_target)
        copy_images(val_images, source_dir, val_target)

        print(
            f"{source_class:>12} -> {target_class:<8} | "
            f"train={len(train_images):4d} val={len(val_images):4d}"
        )


def merge_test(source_test_root: str, output_root: str):
    if not os.path.isdir(source_test_root):
        print(f"skip test split, source does not exist: {source_test_root}")
        return

    for source_class, target_class in CLASS_MAPPING.items():
        source_dir = os.path.join(source_test_root, source_class)
        if not os.path.isdir(source_dir):
            continue

        images = list_images(source_dir)
        target_dir = os.path.join(output_root, "Test", target_class)
        copy_images(images, source_dir, target_dir)
        print(f"{source_class:>12} -> {target_class:<8} | test ={len(images):4d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.path.join(os.getcwd(), "DataSet_Njust_02"),
        help="dataset root that contains Ori/ and optional Test/",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=os.path.join(os.getcwd(), "DataSet_Njust_binary"),
        help="binary dataset output root",
    )
    parser.add_argument("--split_rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    source_ori_root = os.path.join(args.dataset_root, "Ori")
    source_test_root = os.path.join(args.dataset_root, "Test")
    assert os.path.isdir(source_ori_root), f"missing Ori directory: {source_ori_root}"

    build_split_dirs(args.output_root)
    merge_train_val(source_ori_root, args.output_root, args.split_rate)
    merge_test(source_test_root, args.output_root)
    print(f"binary dataset ready: {args.output_root}")


if __name__ == "__main__":
    main()
