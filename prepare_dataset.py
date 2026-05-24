"""Create webcam-like training data from segmented ISL alphabet images.

The source images in ``dataset/`` are light hands on nearly black backgrounds.
This script composites the segmented hands onto varied backgrounds and creates
an explicit ``no_sign`` class so the recognizer can reject invalid frames.
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


LABELS = list("abcdefghijklmnopqrstuvwxyz")
IMG_SIZE = 224
SKIN_TONES_BGR = (
    (82, 118, 176),
    (96, 137, 198),
    (112, 153, 210),
    (132, 175, 225),
    (158, 196, 235),
)


def make_background(rng, photo_backgrounds=None):
    if photo_backgrounds and rng.random() < 0.8:
        return photo_backgrounds[int(rng.integers(len(photo_backgrounds)))].copy()
    top = rng.integers(20, 190, size=3).astype(np.float32)
    bottom = rng.integers(20, 190, size=3).astype(np.float32)
    mix = np.linspace(0, 1, IMG_SIZE, dtype=np.float32)[:, None, None]
    background = top[None, None, :] * (1 - mix) + bottom[None, None, :] * mix
    background = np.repeat(background, IMG_SIZE, axis=1)
    noise = rng.normal(0, rng.uniform(3, 16), background.shape)
    background = np.clip(background + noise, 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(0, 3))):
        x1, y1 = rng.integers(0, IMG_SIZE - 30, size=2)
        x2 = int(min(IMG_SIZE, x1 + rng.integers(20, 110)))
        y2 = int(min(IMG_SIZE, y1 + rng.integers(20, 110)))
        color = tuple(int(c) for c in rng.integers(10, 225, size=3))
        cv2.rectangle(background, (int(x1), int(y1)), (x2, y2), color, -1)
    return cv2.GaussianBlur(background, (9, 9), 0)


def extract_foreground(image, rng):
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    shade = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    tone = np.asarray(SKIN_TONES_BGR[int(rng.integers(len(SKIN_TONES_BGR)))], dtype=np.float32)
    foreground = tone[None, None, :] * (0.38 + 0.72 * shade[:, :, None])
    return np.clip(foreground, 0, 255).astype(np.uint8), mask


def place_foreground(foreground, mask, background, rng, partial=False):
    scale = float(rng.uniform(0.72, 1.02))
    size = max(40, int(IMG_SIZE * scale))
    foreground = cv2.resize(foreground, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
    if partial:
        x = int(rng.choice([-rng.integers(size // 2, size - 15), rng.integers(IMG_SIZE - 20, IMG_SIZE - 5)]))
        y = int(rng.integers(-size // 3, IMG_SIZE - 15))
    else:
        max_offset = max(0, IMG_SIZE - size)
        x = int(rng.integers(0, max_offset + 1))
        y = int(rng.integers(0, max_offset + 1))
    output = background.copy()
    left, top = max(0, x), max(0, y)
    right, bottom = min(IMG_SIZE, x + size), min(IMG_SIZE, y + size)
    if right <= left or bottom <= top:
        return output
    src_x, src_y = left - x, top - y
    crop = foreground[src_y:src_y + bottom - top, src_x:src_x + right - left]
    alpha = mask[src_y:src_y + bottom - top, src_x:src_x + right - left, None]
    region = output[top:bottom, left:right].astype(np.float32)
    output[top:bottom, left:right] = (alpha * crop + (1 - alpha) * region).astype(np.uint8)
    return output


def render_positive(source_path, rng, photo_backgrounds=None):
    source = cv2.imread(str(source_path))
    foreground, mask = extract_foreground(source, rng)
    return place_foreground(foreground, mask, make_background(rng, photo_backgrounds), rng)


def render_no_sign(source_paths, rng, photo_backgrounds=None):
    background = make_background(rng, photo_backgrounds)
    if rng.random() < 0.45:
        source = cv2.imread(str(source_paths[int(rng.integers(len(source_paths)))]))
        foreground, mask = extract_foreground(source, rng)
        return place_foreground(foreground, mask, background, rng, partial=True)
    return background


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise OSError(f"Could not write image: {path}")


def prepare(args):
    source_root = Path(args.source)
    output_root = Path(args.output)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    photo_backgrounds = []
    if args.backgrounds:
        background_root = Path(args.backgrounds)
        background_paths = [
            path
            for path in background_root.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        for path in background_paths[:args.max_backgrounds]:
            image = cv2.imread(str(path))
            if image is not None:
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
                photo_backgrounds.append(cv2.GaussianBlur(image, (5, 5), 0))
        print(f"Using {len(photo_backgrounds)} cached photographic background images.")

    all_sources = []
    for label in LABELS:
        paths = sorted((source_root / label).glob("*.jpg"))
        if len(paths) < args.train_per_class + args.validation_per_class:
            raise ValueError(f"Not enough source images for class {label}: {len(paths)}")
        random.shuffle(paths)
        train_paths = paths[:args.train_per_class]
        validation_paths = paths[-args.validation_per_class:]
        all_sources.extend(train_paths)
        for index, path in enumerate(train_paths):
            for variant in range(args.train_variants):
                target = output_root / "train" / label / f"{index:04d}_{variant}.jpg"
                write_image(target, render_positive(path, rng, photo_backgrounds))
        for index, path in enumerate(validation_paths):
            target = output_root / "validation" / label / f"{index:04d}.jpg"
            write_image(target, render_positive(path, rng, photo_backgrounds))

    for split, count in (("train", args.no_sign_train), ("validation", args.no_sign_validation)):
        for index in range(count):
            target = output_root / split / "no_sign" / f"{index:05d}.jpg"
            write_image(target, render_no_sign(all_sources, rng, photo_backgrounds))

    train_count = len(LABELS) * args.train_per_class * args.train_variants + args.no_sign_train
    val_count = len(LABELS) * args.validation_per_class + args.no_sign_validation
    print(f"Prepared {train_count} training images and {val_count} validation images in {output_root}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="dataset")
    parser.add_argument("--output", default="prepared_dataset")
    parser.add_argument("--backgrounds", default="background_images")
    parser.add_argument("--max-backgrounds", type=int, default=2000)
    parser.add_argument("--train-per-class", type=int, default=350)
    parser.add_argument("--validation-per-class", type=int, default=80)
    parser.add_argument("--train-variants", type=int, default=1)
    parser.add_argument("--no-sign-train", type=int, default=2000)
    parser.add_argument("--no-sign-validation", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
