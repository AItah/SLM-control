from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import cv2
except Exception:
    cv2 = None

from camera_window import CameraWindow
from vortex_window import VortexWindow
from slm_control_window import SlmControlWindow


@dataclass
class ScanSettings:
    x_range_mm: float
    x_step_mm: float
    y_range_mm: float
    y_step_mm: float
    refine_enabled: bool
    refine_x_range_mm: float
    refine_x_step_mm: float
    refine_y_range_mm: float
    refine_y_step_mm: float
    settle_ms: int
    slot: int
    debug_enabled: bool
    angles_count: int
    manual_center: Tuple[float, float]
    manual_radius: float


class OffsetScanWorker(QtCore.QObject):
    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(float, float)
    failed = QtCore.Signal(str)
    debug_data = QtCore.Signal(object, object, object, object, object)
    _ANGLES_FAST = 4
    _MAX_WORSE_STREAK = 2

    def __init__(
        self,
        vortex: VortexWindow,
        slm: SlmControlWindow,
        camera: CameraWindow,
        roi: Tuple[int, int, int, int],
        settings: ScanSettings,
    ) -> None:
        super().__init__()
        self._vortex = vortex
        self._slm = slm
        self._camera = camera
        self._roi = roi
        self._settings = settings
        self._running = True

    @QtCore.Slot()
    def run(self) -> None:
        try:
            best_x, best_y = self._run_scan()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(best_x, best_y)

    def stop(self) -> None:
        self._running = False

    def _run_scan(self) -> Tuple[float, float]:
        center_x, center_y = self._vortex.get_offsets_mm()
        self.log.emit(f"Starting scan at center X={center_x:.3f} mm, Y={center_y:.3f} mm")

        best_x = self._scan_axis(
            axis="x",
            center=center_x,
            fixed=center_y,
            rng=self._settings.x_range_mm,
            step=self._settings.x_step_mm,
        )
        best_y = self._scan_axis(
            axis="y",
            center=center_y,
            fixed=best_x,
            rng=self._settings.y_range_mm,
            step=self._settings.y_step_mm,
        )

        if self._settings.refine_enabled:
            self.log.emit("Starting refinement pass...")
            best_x = self._scan_axis(
                axis="x",
                center=best_x,
                fixed=best_y,
                rng=self._settings.refine_x_range_mm,
                step=self._settings.refine_x_step_mm,
            )
            best_y = self._scan_axis(
                axis="y",
                center=best_y,
                fixed=best_x,
                rng=self._settings.refine_y_range_mm,
                step=self._settings.refine_y_step_mm,
            )

        return best_x, best_y

    def _scan_axis(self, axis: str, center: float, fixed: float, rng: float, step: float) -> float:
        if step <= 0:
            raise ValueError("Step must be > 0.")
        if rng <= 0:
            return center

        angles_full = max(self._ANGLES_FAST, int(self._settings.angles_count))
        bound_lo = center - rng
        bound_hi = center + rng
        max_steps = int(math.floor(rng / step))
        total_est = max(1, 3 + max_steps)
        eval_count = 0

        def eval_offset(offset: float, angles: int) -> float:
            nonlocal eval_count
            if not self._running:
                raise RuntimeError("Scan canceled.")
            if axis == "x":
                off_x, off_y = offset, fixed
            else:
                off_x, off_y = fixed, offset
            score = self._evaluate_offset(off_x, off_y, angles)
            eval_count += 1
            self.progress.emit(min(eval_count, total_est), total_est)
            self.log.emit(
                f"{axis.upper()} {offset:.3f} mm -> score {score:.3f} (angles {angles})"
            )
            return score

        best_offset = center
        center_score = eval_offset(center, self._ANGLES_FAST)
        best_score = center_score

        candidates = []
        for direction in (-1.0, 1.0):
            cand = center + direction * step
            if cand < bound_lo or cand > bound_hi:
                continue
            score = eval_offset(cand, self._ANGLES_FAST)
            candidates.append((score, direction, cand))
            if score < best_score:
                best_score = score
                best_offset = cand

        if not candidates:
            self.log.emit(
                f"Best {axis.upper()} = {best_offset:.3f} mm (score {best_score:.3f})"
            )
            return best_offset

        candidates.sort(key=lambda item: item[0])
        if candidates[0][0] >= center_score:
            self.log.emit(
                f"No improvement along {axis.upper()} axis; keeping center."
            )
            self.log.emit(
                f"Best {axis.upper()} = {best_offset:.3f} mm (score {best_score:.3f})"
            )
            return best_offset

        direction = candidates[0][1]
        current = candidates[0][2]
        current_score = candidates[0][0]
        worse_streak = 0 if current_score <= best_score else 1
        steps_taken = 1

        while steps_taken < max_steps:
            cand = current + direction * step
            if cand < bound_lo or cand > bound_hi:
                break
            angles = angles_full if steps_taken >= 2 else self._ANGLES_FAST
            score = eval_offset(cand, angles)
            steps_taken += 1
            if score < best_score:
                best_score = score
                best_offset = cand
                worse_streak = 0
            else:
                worse_streak += 1
                if worse_streak >= self._MAX_WORSE_STREAK:
                    self.log.emit(
                        f"{axis.upper()} worsening twice; stopping early and backtracking."
                    )
                    break
            current = cand

        self.log.emit(
            f"Best {axis.upper()} = {best_offset:.3f} mm (score {best_score:.3f})"
        )
        return best_offset

    @staticmethod
    def _build_offsets(center: float, rng: float, step: float) -> list[float]:
        if rng <= 0:
            return [center]
        start = center - rng
        end = center + rng
        count = int(round((end - start) / step)) + 1
        return [start + i * step for i in range(count)]

    def _evaluate_offset(
        self, offset_x_mm: float, offset_y_mm: float, angles_count: int
    ) -> float:
        prev_time = self._camera.get_last_frame_time()
        mask_u8 = self._vortex.build_mask(offset_x_mm, offset_y_mm)
        ok = self._slm.send_mask_to_slot(mask_u8, self._settings.slot)
        if not ok:
            raise RuntimeError("Failed to send mask to SLM.")

        self.log.emit("Waiting for SLM settle...")
        time.sleep(max(2.0, self._settings.settle_ms / 1000.0))

        # Wait briefly for a fresh frame
        timeout = time.monotonic() + 1.0
        while time.monotonic() < timeout:
            if self._camera.get_last_frame_time() > prev_time:
                break
            time.sleep(0.05)

        frame = self._camera.get_last_frame()
        if frame is None:
            raise RuntimeError("No camera frame available.")

        x, y, w, h = self._roi
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            raise RuntimeError("ROI is empty.")

        roi_gray = self._to_gray(roi)
        cx_full, cy_full = self._settings.manual_center
        radius = float(self._settings.manual_radius)
        cx = float(cx_full) - float(x)
        cy = float(cy_full) - float(y)
        if not (0 <= cx < w and 0 <= cy < h):
            raise RuntimeError("Manual center is outside the ROI.")

        score, polar, peaks, valid, meta = self._score_manual_concentricity(
            roi_gray, (cx, cy), radius, angles_count
        )

        if self._settings.debug_enabled:
            debug_roi = self._draw_manual_overlay(roi_gray, (cx, cy), radius, meta)
            self.debug_data.emit(debug_roi, polar, peaks, valid, meta)

        self.log.emit(
            f"manual concentricity score={score:.3f} px (valid {int(np.count_nonzero(valid))}/{len(valid)})"
        )
        return float(score)

    @staticmethod
    def _to_gray(roi: np.ndarray) -> np.ndarray:
        if roi.ndim == 2:
            return roi.astype(np.float32)
        return (
            0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]
        ).astype(np.float32)

    def _score_center_distance(self, roi_gray: np.ndarray) -> float:
        hole_center, ring_center = self._find_centers(roi_gray)
        dist = math.hypot(ring_center[0] - hole_center[0], ring_center[1] - hole_center[1])
        self.log.emit(
            f"hole=({hole_center[0]:.1f},{hole_center[1]:.1f}) "
            f"ring=({ring_center[0]:.1f},{ring_center[1]:.1f}) "
            f"dist={dist:.2f}"
        )
        return float(dist)

    def _find_centers(self, roi_gray: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        if cv2 is not None:
            return self._find_centers_cv2(roi_gray)
        return self._find_centers_numpy(roi_gray)

    def _find_centers_cv2(self, roi_gray: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        hole_center = self._find_hole_center_cv2(roi_gray)
        rx, ry, _ = self._find_ring_cv2(roi_gray)
        return hole_center, (rx, ry)

    @staticmethod
    def _find_centers_numpy(roi_gray: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        # Fallback: dark-core centroid and bright-ring centroid
        thr_dark = np.percentile(roi_gray, 10.0)
        mask_dark = roi_gray <= thr_dark
        if not np.any(mask_dark):
            raise RuntimeError("Dark core not found.")
        ys, xs = np.where(mask_dark)
        hx = float(xs.mean())
        hy = float(ys.mean())

        thr_bright = np.percentile(roi_gray, 90.0)
        mask_bright = roi_gray >= thr_bright
        if not np.any(mask_bright):
            raise RuntimeError("Ring not found.")
        ys2, xs2 = np.where(mask_bright)
        rx = float(xs2.mean())
        ry = float(ys2.mean())
        return (hx, hy), (rx, ry)

    def _score_manual_concentricity(
        self,
        roi_gray: np.ndarray,
        center: Tuple[float, float],
        radius: float,
        angles_count: int,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict]:
        angles_count = max(self._ANGLES_FAST, int(angles_count))
        img = roi_gray.astype(np.float32)
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_max <= img_min:
            raise RuntimeError("Flat image.")
        img = (img - img_min) / (img_max - img_min)
        img = (img * 255.0).astype(np.float32)
        if cv2 is not None:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        h, w = img.shape
        cx, cy = center
        if not (0 <= cx < w and 0 <= cy < h):
            raise RuntimeError("Center outside ROI.")
        max_r = min(cx, cy, (w - 1) - cx, (h - 1) - cy)
        if max_r < 5:
            raise RuntimeError("ROI too small.")

        # Robustness: if manual center/radius are a bit off, the ring peak can land on the
        # sampling band's edge for many angles. Try progressively wider bands / looser
        # validity rules before giving up.
        angles = np.linspace(0.0, 2.0 * math.pi, angles_count, endpoint=False)
        attempts = [
            # (band_scale, threshold_percentile, require_non_edge_peak)
            (0.35, 25.0, True),
            (0.55, 20.0, True),
            (0.80, 15.0, False),
            (1.10, 10.0, False),
        ]

        last_diag: dict | None = None
        for band_scale, thr_pct, require_non_edge in attempts:
            band = max(3.0, float(radius) * float(band_scale))
            r_min = max(1.0, float(radius) - band)
            r_max = min(float(max_r) - 1.0, float(radius) + band)
            if r_max <= r_min + 1.0:
                last_diag = {"reason": "invalid_band", "r_min": r_min, "r_max": r_max}
                continue

            polar = self._sample_polar(img, center, r_min, r_max, angles)
            if polar.size == 0:
                last_diag = {"reason": "polar_failed", "r_min": r_min, "r_max": r_max}
                continue

            peaks_idx = np.argmax(polar, axis=1)
            max_per_angle = polar.max(axis=1)
            step = (r_max - r_min) / max(1, (polar.shape[1] - 1))
            peaks_r = r_min + peaks_idx * step

            threshold = np.percentile(max_per_angle, float(thr_pct))
            valid = max_per_angle >= threshold
            if require_non_edge:
                valid = valid & (peaks_idx > 0) & (peaks_idx < (polar.shape[1] - 1))

            n_valid = int(np.count_nonzero(valid))
            last_diag = {
                "reason": "too_few_valid" if n_valid < self._ANGLES_FAST else "ok",
                "band_scale": float(band_scale),
                "thr_pct": float(thr_pct),
                "require_non_edge": bool(require_non_edge),
                "n_valid": n_valid,
                "angles_count": int(angles_count),
                "radius": float(radius),
                "r_min": float(r_min),
                "r_max": float(r_max),
            }
            if n_valid >= self._ANGLES_FAST:
                break

        if last_diag is None or last_diag.get("reason") != "ok":
            diag = last_diag or {}
            raise RuntimeError(
                "Not enough valid angles for manual scoring. "
                "Check `manual_center` / `manual_radius` and ROI cropping. "
                f"diag={diag}"
            )

        peaks_valid = peaks_r[valid].astype(np.float32)
        score = float(np.std(peaks_valid))

        fit_r, fit_dx, fit_dy = self._fit_radius_offset(angles[valid], peaks_valid)
        fit_center = (float(cx + fit_dx), float(cy + fit_dy))

        meta = {
            "angles_deg": np.degrees(angles),
            "max_per_angle": max_per_angle,
            "fit_center": fit_center,
            "fit_offset": (float(fit_dx), float(fit_dy)),
            "fit_radius": float(fit_r),
            "center": (float(cx), float(cy)),
            "radius": float(radius),
            "r_min": float(r_min),
            "r_max": float(r_max),
        }
        return score, polar, peaks_r, valid, meta

    def _sample_polar(
        self,
        img: np.ndarray,
        center: Tuple[float, float],
        r_min: float,
        r_max: float,
        angles: np.ndarray,
    ) -> np.ndarray:
        span = max(1.0, float(r_max - r_min))
        samples = int(max(50, min(300, span * 3.0)))
        radii = np.linspace(r_min, r_max, samples)
        cos_t = np.cos(angles)[:, None]
        sin_t = np.sin(angles)[:, None]
        cx, cy = center
        xs = cx + cos_t * radii[None, :]
        ys = cy + sin_t * radii[None, :]

        if cv2 is not None:
            map_x = xs.astype(np.float32)
            map_y = ys.astype(np.float32)
            return cv2.remap(
                img.astype(np.float32),
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return self._bilinear_sample(img.astype(np.float32), xs, ys)

    @staticmethod
    def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        h, w = img.shape
        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y1, 0, h - 1)

        wa = (x1 - xs) * (y1 - ys)
        wb = (xs - x0) * (y1 - ys)
        wc = (x1 - xs) * (ys - y0)
        wd = (xs - x0) * (ys - y0)

        Ia = img[y0c, x0c]
        Ib = img[y0c, x1c]
        Ic = img[y1c, x0c]
        Id = img[y1c, x1c]

        out = wa * Ia + wb * Ib + wc * Ic + wd * Id
        mask = (xs >= 0) & (xs <= (w - 1)) & (ys >= 0) & (ys <= (h - 1))
        out = np.where(mask, out, 0.0)
        return out.astype(np.float32)

    @staticmethod
    def _fit_radius_offset(angles: np.ndarray, radii: np.ndarray) -> tuple[float, float, float]:
        if angles.size < 3:
            return float(np.mean(radii)), 0.0, 0.0
        a = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
        sol, _, _, _ = np.linalg.lstsq(a, radii, rcond=None)
        r0, dx, dy = sol
        return float(r0), float(dx), float(dy)

    def _score_warp_polar(
        self, roi_gray: np.ndarray, center: Tuple[float, float]
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        img = roi_gray.astype(np.float32)
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_max <= img_min:
            raise RuntimeError("Flat image.")
        img = (img - img_min) / (img_max - img_min)
        img = (img * 255.0).astype(np.uint8)
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        h, w = blur.shape
        cx, cy = center
        if not (0 <= cx < w and 0 <= cy < h):
            cx, cy = w / 2.0, h / 2.0
        max_r = min(cx, cy, (w - 1) - cx, (h - 1) - cy)
        if max_r < 5:
            raise RuntimeError("ROI too small.")

        polar = cv2.warpPolar(
            blur,
            (int(max_r), 360),
            (float(cx), float(cy)),
            max_r,
            cv2.WARP_POLAR_LINEAR,
        )

        # peaks per angle row
        peaks = np.argmax(polar, axis=1)
        max_per_angle = polar.max(axis=1)
        threshold = np.percentile(max_per_angle, 25.0)
        valid = max_per_angle >= threshold
        if np.count_nonzero(valid) < 30:
            raise RuntimeError("Not enough valid angles for warpPolar.")

        peaks_valid = peaks[valid].astype(np.float32)
        return float(np.std(peaks_valid)), polar, peaks, valid

    @staticmethod
    def _normalize_to_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_max <= img_min:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

    def _find_hole_center_cv2(self, roi_gray: np.ndarray) -> tuple[float, float]:
        img = self._normalize_to_u8(roi_gray)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        thr = np.percentile(blur, 10.0)
        dark_mask = (blur <= thr).astype(np.uint8) * 255
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("Dark core not found.")
        contour = max(contours, key=cv2.contourArea)
        m = cv2.moments(contour)
        if m["m00"] == 0:
            raise RuntimeError("Dark core moment is zero.")
        hx = m["m10"] / m["m00"]
        hy = m["m01"] / m["m00"]
        return float(hx), float(hy)

    def _find_ring_cv2(self, roi_gray: np.ndarray) -> tuple[float, float, float]:
        rx, ry, r, _ = self._find_ring_cv2_with_edges(roi_gray)
        return rx, ry, r

    def _find_ring_cv2_with_edges(
        self, roi_gray: np.ndarray
    ) -> tuple[float, float, float, Optional[Tuple[np.ndarray, np.ndarray]]]:
        img = self._normalize_to_u8(roi_gray)
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        hole_center = self._safe_find_hole_center(blur)

        xs, ys = self._ring_edges_from_gradient(blur)
        if len(xs) >= 30:
            if hole_center is not None:
                filt = self._filter_edges_by_radius(xs, ys, blur, hole_center)
                if filt is not None:
                    xs_f, ys_f, r0 = filt
                    if len(xs_f) >= 30:
                        rx, ry, r = self._fit_circle(xs_f, ys_f)
                        return rx, ry, r, (xs_f, ys_f)
            rx, ry, r = self._fit_circle(xs, ys)
            return rx, ry, r, (xs, ys)

        xs, ys = self._ring_edges_from_canny(blur)
        if len(xs) >= 30:
            if hole_center is not None:
                filt = self._filter_edges_by_radius(xs, ys, blur, hole_center)
                if filt is not None:
                    xs_f, ys_f, r0 = filt
                    if len(xs_f) >= 30:
                        rx, ry, r = self._fit_circle(xs_f, ys_f)
                        return rx, ry, r, (xs_f, ys_f)
            rx, ry, r = self._fit_circle(xs, ys)
            return rx, ry, r, (xs, ys)

        rx, ry, r = self._ring_from_bright_mask(blur)
        return rx, ry, r, None

    @staticmethod
    def _ring_edges_from_gradient(blur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        if not np.isfinite(mag).all():
            mag = np.nan_to_num(mag)

        thresholds = [97.0, 95.0, 92.0, 90.0, 85.0, 80.0]
        for pct in thresholds:
            thr = np.percentile(mag, pct)
            edges = mag >= thr
            ys, xs = np.where(edges)
            if len(xs) >= 30:
                return xs, ys
        return np.array([]), np.array([])

    @staticmethod
    def _ring_edges_from_canny(blur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.median(blur)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blur, lower, upper)
        ys, xs = np.where(edges > 0)
        return xs, ys

    @staticmethod
    def _ring_from_bright_mask(blur: np.ndarray) -> tuple[float, float, float]:
        thr = np.percentile(blur, 85.0)
        mask = blur >= thr
        ys, xs = np.where(mask)
        if len(xs) < 50:
            raise RuntimeError("Not enough bright ring pixels.")
        cx = float(xs.mean())
        cy = float(ys.mean())
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        r = float(np.median(d))
        return cx, cy, r

    def _filter_edges_by_radius(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        blur: np.ndarray,
        center: Tuple[float, float],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        if len(xs) == 0:
            return None
        h, w = blur.shape
        cx, cy = center
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        try:
            r0, band = self._estimate_ring_radius(blur, center)
        except Exception:
            return None
        dx = xs.astype(np.float32) - float(cx)
        dy = ys.astype(np.float32) - float(cy)
        r = np.sqrt(dx * dx + dy * dy)
        mask = (r >= (r0 - band)) & (r <= (r0 + band))
        if np.count_nonzero(mask) < 30:
            return None
        return xs[mask], ys[mask], r0

    def _estimate_ring_radius(
        self, img: np.ndarray, center: Tuple[float, float]
    ) -> Tuple[float, float]:
        # Returns (r0, band) where band is half-width for filtering
        h, w = img.shape
        cx, cy = center
        if not (0 <= cx < w and 0 <= cy < h):
            raise RuntimeError("Center outside ROI.")
        y, x = np.indices(img.shape)
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_int = r.astype(np.int32)
        max_r = int(r_int.max())
        if max_r < 5:
            raise RuntimeError("ROI too small for radial profile.")
        tbin = np.bincount(r_int.ravel(), img.ravel().astype(np.float64))
        nr = np.bincount(r_int.ravel())
        radial = tbin / (nr + 1e-6)
        min_r = max(3, int(max_r * 0.05))
        peak_r = int(min_r + np.argmax(radial[min_r:]))
        peak_val = radial[peak_r]
        if peak_val <= 0:
            raise RuntimeError("Invalid radial profile.")
        thr = peak_val * 0.6
        left = peak_r
        while left > min_r and radial[left] > thr:
            left -= 1
        right = peak_r
        while right < len(radial) - 1 and radial[right] > thr:
            right += 1
        band = max(3.0, (right - left) * 0.6)
        if band <= 0:
            band = max(3.0, peak_r * 0.1)
        return float(peak_r), float(band)

    def _safe_find_hole_center(self, roi_gray: np.ndarray) -> Optional[Tuple[float, float]]:
        if cv2 is None:
            return None
        try:
            return self._find_hole_center_cv2(roi_gray)
        except Exception:
            return None

    def _draw_manual_overlay(
        self,
        roi_gray: np.ndarray,
        center: Tuple[float, float],
        radius: float,
        meta: Optional[dict] = None,
    ) -> np.ndarray:
        base = self._normalize_to_u8(roi_gray)
        if cv2 is None:
            return base
        color = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        cx, cy = int(round(center[0])), int(round(center[1]))
        cv2.circle(color, (cx, cy), int(round(radius)), (0, 200, 0), 1)
        cv2.drawMarker(color, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS)
        if meta is not None and "fit_center" in meta:
            fx, fy = meta["fit_center"]
            fx_i, fy_i = int(round(fx)), int(round(fy))
            cv2.drawMarker(
                color, (fx_i, fy_i), (0, 255, 255), markerType=cv2.MARKER_CROSS
            )
        return color

    def _draw_overlay(
        self,
        roi_gray: np.ndarray,
        ring_center: Optional[Tuple[float, float]],
        ring_r: Optional[float],
        hole_center: Optional[Tuple[float, float]],
        edge_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        base = self._normalize_to_u8(roi_gray)
        if cv2 is None:
            return base
        color = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        if edge_points is not None:
            xs, ys = edge_points
            if len(xs) > 0:
                if len(xs) > 2000:
                    idx = np.linspace(0, len(xs) - 1, 2000).astype(int)
                    xs = xs[idx]
                    ys = ys[idx]
                xs = xs.astype(int)
                ys = ys.astype(int)
                h, w = color.shape[:2]
                mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                color[ys[mask], xs[mask]] = (0, 255, 255)
        if ring_center is not None and ring_r is not None:
            cx, cy = int(round(ring_center[0])), int(round(ring_center[1]))
            cv2.circle(color, (cx, cy), int(round(ring_r)), (0, 255, 0), 1)
            cv2.drawMarker(color, (cx, cy), (255, 0, 0), markerType=cv2.MARKER_CROSS)
        if hole_center is not None:
            hx, hy = int(round(hole_center[0])), int(round(hole_center[1]))
            cv2.drawMarker(color, (hx, hy), (0, 0, 255), markerType=cv2.MARKER_CROSS)
        return color

    @staticmethod
    def _fit_circle(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
        # Algebraic least squares circle fit (Kasa)
        xs = xs.astype(np.float64)
        ys = ys.astype(np.float64)
        a = np.c_[2 * xs, 2 * ys, np.ones_like(xs)]
        b = xs * xs + ys * ys
        sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy, c = sol
        r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
        return float(cx), float(cy), float(r)


class DebugWindow(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Donut Optimization Debug")
        self.resize(900, 600)

        self._roi_pixmap: Optional[QtGui.QPixmap] = None
        self._polar_pixmap: Optional[QtGui.QPixmap] = None
        self._peaks_pixmap: Optional[QtGui.QPixmap] = None

        layout = QtWidgets.QGridLayout(self)

        self.lbl_roi_title = QtWidgets.QLabel("ROI (grayscale)")
        self.lbl_polar_title = QtWidgets.QLabel("Polar unwrap")
        self.lbl_peaks_title = QtWidgets.QLabel("Peaks (radius vs angle)")
        self.lbl_data_title = QtWidgets.QLabel("Angle data")

        self.lbl_roi = QtWidgets.QLabel()
        self.lbl_polar = QtWidgets.QLabel()
        self.lbl_peaks = QtWidgets.QLabel()
        self.txt_data = QtWidgets.QPlainTextEdit(readOnly=True)
        self.txt_data.setMaximumBlockCount(10000)
        for lbl in (self.lbl_roi, self.lbl_polar, self.lbl_peaks):
            lbl.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setMinimumSize(240, 180)

        self.lbl_stats = QtWidgets.QLabel("")

        layout.addWidget(self.lbl_roi_title, 0, 0)
        layout.addWidget(self.lbl_polar_title, 0, 1)
        layout.addWidget(self.lbl_roi, 1, 0)
        layout.addWidget(self.lbl_polar, 1, 1)
        layout.addWidget(self.lbl_peaks_title, 2, 0, 1, 2)
        layout.addWidget(self.lbl_peaks, 3, 0, 1, 2)
        layout.addWidget(self.lbl_stats, 4, 0, 1, 2)
        layout.addWidget(self.lbl_data_title, 5, 0, 1, 2)
        layout.addWidget(self.txt_data, 6, 0, 1, 2)

    def update_views(
        self,
        roi_gray: np.ndarray,
        polar: np.ndarray,
        peaks: np.ndarray,
        valid: np.ndarray,
        meta: dict,
    ) -> None:
        self._roi_pixmap = self._gray_to_pixmap(roi_gray) if roi_gray is not None else None
        self._polar_pixmap = self._gray_to_pixmap(polar) if polar is not None else None
        self._peaks_pixmap = self._plot_peaks(peaks, valid)
        self._apply_scaled()

        if peaks is not None and len(peaks) > 0:
            valid_count = int(np.count_nonzero(valid)) if valid is not None else len(peaks)
            std = float(np.std(peaks[valid])) if valid is not None else float(np.std(peaks))
            extra = ""
            if meta:
                fit = meta.get("fit_offset")
                if fit:
                    extra = f" | Fit dx,dy: ({fit[0]:.2f}, {fit[1]:.2f})"
            self.lbl_stats.setText(
                f"Angles: {len(peaks)} | Valid: {valid_count} | Peaks std: {std:.3f}{extra}"
            )
        else:
            self.lbl_stats.setText("No peaks data.")

        self._update_data_table(peaks, valid, meta)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_scaled()

    def _apply_scaled(self) -> None:
        if self._roi_pixmap is not None:
            self.lbl_roi.setPixmap(self._scaled(self._roi_pixmap, self.lbl_roi))
        else:
            self.lbl_roi.clear()
        if self._polar_pixmap is not None:
            self.lbl_polar.setPixmap(self._scaled(self._polar_pixmap, self.lbl_polar))
        else:
            self.lbl_polar.clear()
        if self._peaks_pixmap is not None:
            self.lbl_peaks.setPixmap(self._scaled(self._peaks_pixmap, self.lbl_peaks))
        else:
            self.lbl_peaks.clear()

    def _update_data_table(
        self, peaks: np.ndarray, valid: np.ndarray, meta: Optional[dict]
    ) -> None:
        if peaks is None or meta is None or "angles_deg" not in meta:
            self.txt_data.setPlainText("No data.")
            return
        angles = meta.get("angles_deg")
        max_per = meta.get("max_per_angle")
        if angles is None:
            self.txt_data.setPlainText("No data.")
            return
        if valid is None or len(valid) != len(peaks):
            valid = np.ones(len(peaks), dtype=bool)

        lines = []
        if "center" in meta and "radius" in meta:
            cx, cy = meta["center"]
            radius = meta["radius"]
            r_min = meta.get("r_min")
            r_max = meta.get("r_max")
            lines.append(
                f"center=({cx:.2f}, {cy:.2f}) radius={radius:.2f} "
                f"r_min={r_min:.2f} r_max={r_max:.2f}"
            )
        if "fit_center" in meta:
            fx, fy = meta["fit_center"]
            lines.append(f"fit_center=({fx:.2f}, {fy:.2f})")
        for i, angle in enumerate(angles):
            peak_r = peaks[i] if i < len(peaks) else float("nan")
            max_val = (
                max_per[i] if max_per is not None and i < len(max_per) else float("nan")
            )
            val = 1 if bool(valid[i]) else 0
            lines.append(
                f"{i:03d} angle={angle:6.1f} deg peak_r={peak_r:7.2f} "
                f"max={max_val:7.1f} valid={val}"
            )
        self.txt_data.setPlainText("\n".join(lines))

    @staticmethod
    def _scaled(pix: QtGui.QPixmap, label: QtWidgets.QLabel) -> QtGui.QPixmap:
        return pix.scaled(
            label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

    @staticmethod
    def _gray_to_pixmap(arr: np.ndarray) -> QtGui.QPixmap:
        if arr.ndim == 3 and arr.shape[2] == 3:
            if arr.dtype != np.uint8:
                arr_min = float(np.min(arr))
                arr_max = float(np.max(arr))
                if arr_max <= arr_min:
                    arr = np.zeros_like(arr, dtype=np.uint8)
                else:
                    arr = ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
            arr = np.ascontiguousarray(arr)
            h, w, _ = arr.shape
            fmt = QtGui.QImage.Format_BGR888
            qimg = QtGui.QImage(arr.data, w, h, w * 3, fmt)
            return QtGui.QPixmap.fromImage(qimg.copy())

        if arr.ndim != 2:
            arr = arr[:, :, 0]
        if arr.dtype != np.uint8:
            arr_min = float(np.min(arr))
            arr_max = float(np.max(arr))
            if arr_max <= arr_min:
                arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                arr = ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
        arr = np.ascontiguousarray(arr)
        h, w = arr.shape
        qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        return QtGui.QPixmap.fromImage(qimg.copy())

    @staticmethod
    def _plot_peaks(peaks: np.ndarray, valid: np.ndarray) -> QtGui.QPixmap:
        if peaks is None or len(peaks) == 0:
            pix = QtGui.QPixmap(400, 200)
            pix.fill(QtGui.QColor(40, 40, 40))
            return pix

        peaks = peaks.astype(np.float32)
        width = max(200, int(len(peaks)))
        height = 200
        pix = QtGui.QPixmap(width, height)
        pix.fill(QtGui.QColor(30, 30, 30))

        max_p = float(np.max(peaks)) if float(np.max(peaks)) > 0 else 1.0
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)

        if valid is None or len(valid) != len(peaks):
            valid = np.ones(len(peaks), dtype=bool)

        pen_valid = QtGui.QPen(QtGui.QColor(0, 200, 0), 1)
        pen_invalid = QtGui.QPen(QtGui.QColor(180, 60, 60), 1)

        for i in range(1, len(peaks)):
            y1 = height - 1 - int((peaks[i - 1] / max_p) * (height - 1))
            y2 = height - 1 - int((peaks[i] / max_p) * (height - 1))
            painter.setPen(pen_valid if valid[i] and valid[i - 1] else pen_invalid)
            painter.drawLine(i - 1, y1, i, y2)

        painter.end()
        return pix


class DonutOptimizationWindow(QtWidgets.QDialog):
    def __init__(
        self,
        vortex: VortexWindow,
        slm: SlmControlWindow,
        camera: CameraWindow,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._vortex = vortex
        self._slm = slm
        self._camera = camera
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[OffsetScanWorker] = None
        self._debug_window: Optional[DebugWindow] = None
        self._manual_center: Optional[Tuple[float, float]] = None
        self._manual_radius: Optional[float] = None

        self.setWindowTitle("Donut Optimization Wizard")
        self.resize(600, 520)

        self._build_ui()
        self._camera.roi_changed.connect(self._on_roi_changed)
        self._camera.point_selected.connect(self._on_point_selected)
        self._camera.circle_selected.connect(self._on_circle_selected)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        roi_group = QtWidgets.QGroupBox("1) Select ROI")
        roi_layout = QtWidgets.QHBoxLayout(roi_group)
        self.lbl_roi = QtWidgets.QLabel("ROI: not set")
        self.btn_select_roi = QtWidgets.QPushButton("Select ROI")
        self.btn_select_roi.clicked.connect(self._select_roi)
        self.btn_clear_roi = QtWidgets.QPushButton("Clear ROI")
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        roi_layout.addWidget(self.lbl_roi)
        roi_layout.addStretch(1)
        roi_layout.addWidget(self.btn_select_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        layout.addWidget(roi_group)

        manual_group = QtWidgets.QGroupBox("2) Manual Target")
        manual_layout = QtWidgets.QGridLayout(manual_group)
        self.lbl_manual_center = QtWidgets.QLabel("Dark spot: not set")
        self.lbl_manual_radius = QtWidgets.QLabel("Circle: not set")
        self.btn_pick_center = QtWidgets.QPushButton("Pick dark spot")
        self.btn_pick_center.clicked.connect(self._pick_dark_spot)
        self.btn_pick_circle = QtWidgets.QPushButton("Draw donut circle")
        self.btn_pick_circle.clicked.connect(self._pick_donut_circle)
        self.btn_clear_manual = QtWidgets.QPushButton("Clear")
        self.btn_clear_manual.clicked.connect(self._clear_manual_target)

        manual_layout.addWidget(self.lbl_manual_center, 0, 0, 1, 2)
        manual_layout.addWidget(self.lbl_manual_radius, 1, 0, 1, 2)
        manual_layout.addWidget(self.btn_pick_center, 0, 2)
        manual_layout.addWidget(self.btn_pick_circle, 1, 2)
        manual_layout.addWidget(self.btn_clear_manual, 0, 3, 2, 1)
        layout.addWidget(manual_group)

        scan_group = QtWidgets.QGroupBox("3) Scan Settings (mm)")
        scan_layout = QtWidgets.QGridLayout(scan_group)
        r = 0
        self.dsb_x_range = QtWidgets.QDoubleSpinBox()
        self.dsb_x_range.setRange(0.0, 10.0)
        self.dsb_x_range.setDecimals(3)
        self.dsb_x_range.setValue(0.3)
        self.dsb_x_step = QtWidgets.QDoubleSpinBox()
        self.dsb_x_step.setRange(0.001, 10.0)
        self.dsb_x_step.setDecimals(3)
        self.dsb_x_step.setValue(0.05)
        self.dsb_y_range = QtWidgets.QDoubleSpinBox()
        self.dsb_y_range.setRange(0.0, 10.0)
        self.dsb_y_range.setDecimals(3)
        self.dsb_y_range.setValue(0.3)
        self.dsb_y_step = QtWidgets.QDoubleSpinBox()
        self.dsb_y_step.setRange(0.001, 10.0)
        self.dsb_y_step.setDecimals(3)
        self.dsb_y_step.setValue(0.05)

        scan_layout.addWidget(QtWidgets.QLabel("X range ±"), r, 0)
        scan_layout.addWidget(self.dsb_x_range, r, 1)
        scan_layout.addWidget(QtWidgets.QLabel("X step"), r, 2)
        scan_layout.addWidget(self.dsb_x_step, r, 3)
        r += 1
        scan_layout.addWidget(QtWidgets.QLabel("Y range ±"), r, 0)
        scan_layout.addWidget(self.dsb_y_range, r, 1)
        scan_layout.addWidget(QtWidgets.QLabel("Y step"), r, 2)
        scan_layout.addWidget(self.dsb_y_step, r, 3)
        r += 1

        self.chk_refine = QtWidgets.QCheckBox("Enable refinement")
        self.chk_refine.setChecked(True)
        scan_layout.addWidget(self.chk_refine, r, 0, 1, 2)
        r += 1

        self.dsb_ref_x_range = QtWidgets.QDoubleSpinBox()
        self.dsb_ref_x_range.setRange(0.0, 10.0)
        self.dsb_ref_x_range.setDecimals(3)
        self.dsb_ref_x_range.setValue(0.1)
        self.dsb_ref_x_step = QtWidgets.QDoubleSpinBox()
        self.dsb_ref_x_step.setRange(0.001, 10.0)
        self.dsb_ref_x_step.setDecimals(3)
        self.dsb_ref_x_step.setValue(0.01)
        self.dsb_ref_y_range = QtWidgets.QDoubleSpinBox()
        self.dsb_ref_y_range.setRange(0.0, 10.0)
        self.dsb_ref_y_range.setDecimals(3)
        self.dsb_ref_y_range.setValue(0.1)
        self.dsb_ref_y_step = QtWidgets.QDoubleSpinBox()
        self.dsb_ref_y_step.setRange(0.001, 10.0)
        self.dsb_ref_y_step.setDecimals(3)
        self.dsb_ref_y_step.setValue(0.01)

        scan_layout.addWidget(QtWidgets.QLabel("Ref X range ±"), r, 0)
        scan_layout.addWidget(self.dsb_ref_x_range, r, 1)
        scan_layout.addWidget(QtWidgets.QLabel("Ref X step"), r, 2)
        scan_layout.addWidget(self.dsb_ref_x_step, r, 3)
        r += 1
        scan_layout.addWidget(QtWidgets.QLabel("Ref Y range ±"), r, 0)
        scan_layout.addWidget(self.dsb_ref_y_range, r, 1)
        scan_layout.addWidget(QtWidgets.QLabel("Ref Y step"), r, 2)
        scan_layout.addWidget(self.dsb_ref_y_step, r, 3)
        r += 1

        self.spin_settle = QtWidgets.QSpinBox()
        self.spin_settle.setRange(2000, 10000)
        self.spin_settle.setValue(2000)
        self.spin_settle.setSuffix(" ms")
        self.spin_slot = QtWidgets.QSpinBox()
        self.spin_slot.setRange(0, 15)
        self.spin_slot.setValue(0)

        scan_layout.addWidget(QtWidgets.QLabel("Settle time"), r, 0)
        scan_layout.addWidget(self.spin_settle, r, 1)
        scan_layout.addWidget(QtWidgets.QLabel("SLM slot"), r, 2)
        scan_layout.addWidget(self.spin_slot, r, 3)
        r += 1

        self.spin_angles = QtWidgets.QSpinBox()
        self.spin_angles.setRange(4, 360)
        self.spin_angles.setValue(10)
        scan_layout.addWidget(QtWidgets.QLabel("Angles"), r, 0)
        scan_layout.addWidget(self.spin_angles, r, 1)

        layout.addWidget(scan_group)

        btns = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addStretch(1)
        layout.addLayout(btns)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log, 1)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)

        self.chk_debug = QtWidgets.QCheckBox("Enable debug view")
        self.chk_debug.setChecked(False)
        self.btn_debug = QtWidgets.QPushButton("Open Debug Window")
        self.btn_debug.clicked.connect(self._open_debug)
        dbg = QtWidgets.QHBoxLayout()
        dbg.addWidget(self.chk_debug)
        dbg.addWidget(self.btn_debug)
        dbg.addStretch(1)
        layout.addLayout(dbg)

    def _select_roi(self) -> None:
        if not self._camera.is_running():
            self._append_error("Camera must be running to select ROI.")
            return
        self._camera.begin_roi_selection()

    def _on_roi_changed(self, roi: Tuple[int, int, int, int]) -> None:
        if roi is None:
            self.lbl_roi.setText("ROI: not set")
        else:
            self.lbl_roi.setText(f"ROI: {roi}")

    def _clear_roi(self) -> None:
        self._camera.clear_roi()

    def _pick_dark_spot(self) -> None:
        if not self._camera.is_running():
            self._append_error("Camera must be running.")
            return
        self._camera.begin_point_selection()

    def _pick_donut_circle(self) -> None:
        if not self._camera.is_running():
            self._append_error("Camera must be running.")
            return
        if self._manual_center is None:
            self._append_error("Pick the dark spot center first.")
            return
        # Circle selection should be independent from the picked dark-spot center.
        self._camera.begin_circle_selection()

    def _clear_manual_target(self) -> None:
        self._manual_center = None
        self._manual_radius = None
        self._update_manual_labels()
        self._camera.clear_manual_marks()

    def _on_point_selected(self, point: Tuple[float, float]) -> None:
        self._manual_center = (float(point[0]), float(point[1]))
        self._update_manual_labels()
        self._append_log(
            f"Dark spot set: ({self._manual_center[0]:.1f}, {self._manual_center[1]:.1f})"
        )

    def _on_circle_selected(self, circle: Tuple[float, float, float]) -> None:
        cx, cy, r = circle
        self._manual_radius = float(r)
        self._update_manual_labels()
        self._append_log(
            f"Donut circle set: radius={r:.1f} px (drawn at ({cx:.1f}, {cy:.1f}))"
        )

    def _update_manual_labels(self) -> None:
        if self._manual_center is None:
            self.lbl_manual_center.setText("Dark spot: not set")
        else:
            self.lbl_manual_center.setText(
                f"Dark spot: ({self._manual_center[0]:.1f}, {self._manual_center[1]:.1f})"
            )
        if self._manual_radius is None:
            self.lbl_manual_radius.setText("Circle: not set")
        else:
            self.lbl_manual_radius.setText(f"Circle radius: {self._manual_radius:.1f} px")

    def _start(self) -> None:
        if not self._camera.is_running():
            self._append_error("Camera must be running.")
            return
        roi = self._camera.get_roi()
        if roi is None:
            self._append_error("Please select ROI first.")
            return
        if self._manual_center is None or self._manual_radius is None:
            self._append_error("Please pick the dark spot and draw the donut circle.")
            return
        cx, cy = self._manual_center
        x, y, w, h = roi
        if not (x <= cx < x + w and y <= cy < y + h):
            self._append_error("Manual center must be inside the ROI.")
            return
        if self._manual_radius <= 0:
            self._append_error("Manual circle radius must be > 0.")
            return
        if self._thread is not None:
            return

        settings = ScanSettings(
            x_range_mm=float(self.dsb_x_range.value()),
            x_step_mm=float(self.dsb_x_step.value()),
            y_range_mm=float(self.dsb_y_range.value()),
            y_step_mm=float(self.dsb_y_step.value()),
            refine_enabled=self.chk_refine.isChecked(),
            refine_x_range_mm=float(self.dsb_ref_x_range.value()),
            refine_x_step_mm=float(self.dsb_ref_x_step.value()),
            refine_y_range_mm=float(self.dsb_ref_y_range.value()),
            refine_y_step_mm=float(self.dsb_ref_y_step.value()),
            settle_ms=int(self.spin_settle.value()),
            slot=int(self.spin_slot.value()),
            debug_enabled=self.chk_debug.isChecked(),
            angles_count=int(self.spin_angles.value()),
            manual_center=(float(cx), float(cy)),
            manual_radius=float(self._manual_radius),
        )

        self._thread = QtCore.QThread(self)
        self._worker = OffsetScanWorker(self._vortex, self._slm, self._camera, roi, settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.debug_data.connect(self._on_debug_data)
        self._thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop(self) -> None:
        if self._worker:
            self._worker.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(1000)
        self._thread = None
        self._worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.progress.setValue(int((current / total) * 100))

    def _on_finished(self, best_x: float, best_y: float) -> None:
        self._append_log(f"Optimization complete. Best X={best_x:.3f} mm, Y={best_y:.3f} mm")
        self._vortex.set_offsets_mm(best_x, best_y)
        self._stop()

    def _on_failed(self, msg: str) -> None:
        self._append_error(msg)
        self._stop()

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _append_error(self, text: str) -> None:
        self.log.appendPlainText("ERROR: " + text)

    def _open_debug(self) -> None:
        if self._debug_window is None:
            self._debug_window = DebugWindow(self)
        self._debug_window.show()
        self._debug_window.raise_()
        self._debug_window.activateWindow()

    def _on_debug_data(
        self,
        roi_gray: np.ndarray,
        polar: np.ndarray,
        peaks: np.ndarray,
        valid: np.ndarray,
        meta: dict,
    ) -> None:
        if self._debug_window is None:
            self._debug_window = DebugWindow(self)
        if self.chk_debug.isChecked():
            self._debug_window.update_views(roi_gray, polar, peaks, valid, meta)
