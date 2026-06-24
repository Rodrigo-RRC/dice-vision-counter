"""
Gera imagens anotadas mostrando os dados detectados e seus pips marcados.
Salva os resultados em assets/ na raiz do projeto.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from detector import (
    create_purple_mask,
    find_dice_contours,
    load_image,
    preprocess_image,
)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

GREEN = (0, 220, 0)
RED   = (0, 0, 220)
WHITE = (255, 255, 255)


def _detect_pips_with_keypoints(image: np.ndarray, contour: np.ndarray):
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y + h, x:x + w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 1)

    die_area = w * h
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = die_area * 0.004
    params.maxArea = die_area * 0.06
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByInertia = True
    params.minInertiaRatio = 0.4
    params.filterByConvexity = True
    params.minConvexity = 0.7

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(roi_gray)
    return x, y, w, h, keypoints


def annotate(image_path: str) -> np.ndarray:
    image = load_image(image_path)
    if image is None:
        return None

    hsv  = preprocess_image(image)
    mask = create_purple_mask(hsv)
    contours = find_dice_contours(mask)

    out = image.copy()

    for i, contour in enumerate(contours):
        x, y, w, h, keypoints = _detect_pips_with_keypoints(image, contour)

        # bounding box verde
        cv2.rectangle(out, (x, y), (x + w, y + h), GREEN, 2)

        # círculo vermelho em cada pip
        for kp in keypoints:
            cx = int(kp.pt[0]) + x
            cy = int(kp.pt[1]) + y
            r  = max(int(kp.size / 2), 6)
            cv2.circle(out, (cx, cy), r, RED, 2)

        # label com contagem
        label = str(len(keypoints))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(out, (x, y - th - 10), (x + tw + 8, y), GREEN, -1)
        cv2.putText(out, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)

    return out


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)

    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    images   = sorted(p for p in data_dir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    for img_path in images:
        result = annotate(str(img_path))
        if result is None:
            continue
        out_path = ASSETS_DIR / f"{img_path.stem}_detected{img_path.suffix}"
        cv2.imwrite(str(out_path), result)
        print(f"Salvo: {out_path}")


if __name__ == "__main__":
    main()
