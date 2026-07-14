from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    id: int
    label: str
    color: str
    confidence: float
    area: float
    x: int
    y: int
    width: int
    height: int


COLOR_RANGES = {
    "red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    "orange": [((11, 70, 50), (25, 255, 255))],
    "yellow": [((26, 70, 50), (35, 255, 255))],
    "green": [((36, 50, 40), (85, 255, 255))],
    "blue": [((86, 50, 40), (130, 255, 255))],
    "purple": [((131, 40, 40), (160, 255, 255))],
    "pink": [((161, 40, 40), (169, 255, 255))],
    "white": [((0, 0, 200), (180, 50, 255))],
    "gray": [((0, 0, 60), (180, 50, 199))],
    "black": [((0, 0, 0), (180, 255, 59))],
}


def create_demo_image(path: Path) -> None:
    image = np.full((720, 1000, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (70, 90), (300, 290), (40, 80, 230), -1)
    cv2.circle(image, (520, 190), 115, (50, 180, 70), -1)
    cv2.drawContours(image, [np.array([[760, 305], [640, 95], [890, 95]])], 0, (230, 160, 30), -1)
    cv2.rectangle(image, (105, 430), (395, 610), (210, 80, 60), -1)
    cv2.ellipse(image, (650, 520), (150, 85), 0, 0, 360, (170, 80, 200), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 130)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated


def classify_shape(contour: np.ndarray) -> tuple[str, float]:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "unknown", 0.0

    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    vertices = len(approx)
    x, y, width, height = cv2.boundingRect(approx)
    aspect_ratio = width / float(height) if height else 0
    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    if vertices == 3:
        return "triangle", 0.94
    if vertices == 4:
        if 0.90 <= aspect_ratio <= 1.10:
            return "square", 0.92
        return "rectangle", 0.91
    if vertices == 5:
        return "pentagon", 0.88
    if circularity > 0.82:
        return "circle", min(0.98, float(circularity))
    if circularity > 0.58:
        return "ellipse", min(0.90, float(circularity))
    return "polygon", min(0.85, max(0.45, float(circularity)))


def dominant_color(image: np.ndarray, contour: np.ndarray) -> str:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    scores: dict[str, int] = {}

    for name, ranges in COLOR_RANGES.items():
        total = 0
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            color_mask = cv2.inRange(hsv, lower_np, upper_np)
            total += int(cv2.countNonZero(cv2.bitwise_and(color_mask, color_mask, mask=mask)))
        scores[name] = total

    color = max(scores, key=scores.get)
    return color if scores[color] > 0 else "unknown"


def detect_objects(image: np.ndarray, min_area: int) -> list[tuple[Detection, np.ndarray]]:
    mask = preprocess(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects: list[tuple[Detection, np.ndarray]] = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        label, confidence = classify_shape(contour)
        color = dominant_color(image, contour)
        detection = Detection(
            id=len(objects) + 1,
            label=label,
            color=color,
            confidence=round(confidence, 3),
            area=round(float(area), 2),
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
        )
        objects.append((detection, contour))

    return objects


def annotate(image: np.ndarray, objects: list[tuple[Detection, np.ndarray]]) -> np.ndarray:
    output = image.copy()
    for detection, contour in objects:
        cv2.drawContours(output, [contour], -1, (0, 0, 0), 3)
        cv2.rectangle(
            output,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            (0, 0, 0),
            2,
        )
        text = f"{detection.id}: {detection.color} {detection.label}"
        y = max(30, detection.y - 12)
        cv2.putText(output, text, (detection.x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return output


def save_results(objects: list[tuple[Detection, np.ndarray]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(detection) for detection, _ in objects]

    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(Detection.__dataclass_fields__.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(input_path: Path, output_dir: Path, min_area: int) -> list[Detection]:
    image = read_image(input_path)
    objects = detect_objects(image, min_area)
    annotated = annotate(image, objects)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "annotated.png"), annotated)
    save_results(objects, output_dir)
    return [detection for detection, _ in objects]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/demo.png"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--min-area", type=int, default=1200)
    parser.add_argument("--create-demo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.create_demo or not args.input.exists():
        create_demo_image(args.input)

    detections = run(args.input, args.output, args.min_area)
    for detection in detections:
        print(
            f"{detection.id}. {detection.color} {detection.label} "
            f"confidence={detection.confidence} bbox=({detection.x},{detection.y},{detection.width},{detection.height})"
        )


if __name__ == "__main__":
    main()
