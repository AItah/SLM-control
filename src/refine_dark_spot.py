import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _to_u8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    gmin = float(np.min(gray))
    gmax = float(np.max(gray))
    if gmax <= gmin:
        return np.zeros_like(gray, dtype=np.uint8)
    return ((gray - gmin) / (gmax - gmin) * 255.0).astype(np.uint8)


def _refine_once(
    img: np.ndarray,
    click_x: float,
    click_y: float,
    roi_size: int,
    sensitivity: float,
    min_area: int,
) -> Tuple[Optional[Tuple[float, float]], Optional[float], dict]:
    if img is None or img.size == 0:
        return None, None, {"stage": "empty_image"}

    gray = _to_u8(_to_gray(img))
    gray = cv2.medianBlur(gray, 3)
    h, w = gray.shape[:2]
    cx = int(round(float(click_x)))
    cy = int(round(float(click_y)))

    y1 = max(0, cy - roi_size)
    y2 = min(h, cy + roi_size)
    x1 = max(0, cx - roi_size)
    x2 = min(w, cx + roi_size)
    if x2 <= x1 or y2 <= y1:
        return None, None, {"stage": "invalid_roi", "roi": (x1, y1, x2, y2)}

    roi = gray[y1:y2, x1:x2]
    min_dim = min(roi.shape[0], roi.shape[1])
    if min_dim < 3:
        return None, None, {"stage": "roi_too_small", "roi_shape": roi.shape}

    if min_dim % 2 == 0:
        min_dim -= 1
    block = max(3, min(51, min_dim))
    if block % 2 == 0:
        block -= 1
    c = max(1, int(round(10 * float(sensitivity))))
    bin_mask = cv2.adaptiveThreshold(
        roi,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        c,
    )
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )
    if num_labels <= 1:
        pct = max(5.0, min(40.0, 30.0 * float(sensitivity)))
        thr = np.percentile(roi, pct)
        bin_mask = (roi <= thr).astype(np.uint8) * 255
        bin_mask = cv2.morphologyEx(
            bin_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8
        )
        if num_labels <= 1:
            return None, None, {"stage": "no_components", "roi_shape": roi.shape}

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= int(min_area)
    if not np.any(keep):
        return None, None, {"stage": "small_components", "max_area": float(areas.max())}

    idx = int(np.where(keep)[0][np.argmax(areas[keep])] + 1)
    area = float(stats[idx, cv2.CC_STAT_AREA])
    if area <= 0:
        return None, None, {"stage": "area_zero", "area": area}

    mask = labels == idx
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None, None, {"stage": "mask_empty"}

    weights = roi[mask].astype(np.float64)
    wsum = float(np.sum(weights))
    if wsum > 0:
        local_cx = float(np.sum(xs * weights) / wsum)
        local_cy = float(np.sum(ys * weights) / wsum)
    else:
        local_cx = float(xs.mean())
        local_cy = float(ys.mean())

    global_cx = float(x1 + local_cx)
    global_cy = float(y1 + local_cy)
    diameter = float(np.sqrt(4.0 * area / np.pi))
    return (
        (global_cx, global_cy),
        diameter,
        {"stage": "adaptive_cc", "area": area, "roi": (x1, y1, x2, y2)},
    )


def refine_dark_spot(
    img: np.ndarray,
    click_x: float,
    click_y: float,
    roi_size: int = 75,
    sensitivity: float = 0.5,
    min_area: int = 20,
) -> Tuple[Optional[Tuple[float, float]], Optional[float], dict]:
    center1, diameter1, dbg1 = _refine_once(
        img, click_x, click_y, roi_size, sensitivity, min_area
    )
    if center1 is None:
        return None, None, {"pass1": dbg1}
    center2, diameter2, dbg2 = _refine_once(
        img, center1[0], center1[1], roi_size, sensitivity, min_area
    )
    if center2 is None:
        return center1, diameter1, {"pass1": dbg1, "pass2": dbg2}
    shift = float(
        np.hypot(center2[0] - center1[0], center2[1] - center1[1])
    )
    diameter = diameter2 if diameter2 is not None else diameter1
    return center2, diameter, {"pass1": dbg1, "pass2": dbg2, "shift_px": shift}


def _select_file() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
    )
    root.destroy()
    return file_path if file_path else None


def _run_demo(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        print("Error: Could not load the image.")
        return

    display = img.copy()

    def on_click(event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        center, diameter, dbg = refine_dark_spot(img, x, y)
        out = display.copy()
        cv2.drawMarker(
            out, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 15, 2
        )
        if center is None:
            print(f"Refine failed dbg={dbg}")
        else:
            cx, cy = int(round(center[0])), int(round(center[1]))
            cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)
            if diameter is not None:
                cv2.circle(out, (cx, cy), int(round(diameter / 2.0)), (0, 255, 255), 2)
            print(f"Center: ({center[0]:.2f}, {center[1]:.2f}) | Diameter: {diameter:.2f}px")
        cv2.imshow("Dark Spot Finder", out)

    cv2.namedWindow("Dark Spot Finder")
    cv2.setMouseCallback("Dark Spot Finder", on_click)
    print(f"Loaded: {path}")
    print("Click on a dark spot. Press any key to exit.")
    cv2.imshow("Dark Spot Finder", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else _select_file()
    if not img_path:
        print("No file selected. Exiting.")
        raise SystemExit(0)
    _run_demo(img_path)
