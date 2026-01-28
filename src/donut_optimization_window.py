from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from refine_dark_spot import refine_dark_spot
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
    settle_ms: int
    slot: int
    debug_enabled: bool
    angles_count: int
    circle_center: Tuple[float, float]
    circle_radius: float
    dark_hint: Tuple[float, float]
    pixel_size_mm: float
    threshold_px: float
    filter_enabled: bool
    filter_threshold: float
    fast_search: bool
    fast_min_step: float
    fast_multi_pass: bool
    fast_shrink_factor: float
    scan_mode: str


class OffsetScanWorker(QtCore.QObject):
    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(float, float)
    failed = QtCore.Signal(str)
    dark_center = QtCore.Signal(object)
    debug_data = QtCore.Signal(object, object, object, object, object, object)
    _ANGLES_FAST = 4
    _MAX_WORSE_STREAK = 2
    _TARGET_DIST_PX = 2.0
    _MIN_STEP_FRACTION = 0.1
    _DARK_SEARCH_FRAC = 0.6
    _DARK_ROI_PX = 75

    def __init__(
        self,
        vortex: VortexWindow,
        slm: SlmControlWindow,
        camera: CameraWindow,
        settings: ScanSettings,
    ) -> None:
        super().__init__()
        self._vortex = vortex
        self._slm = slm
        self._camera = camera
        self._settings = settings
        self._running = True
        self._current_dark = (
            float(self._settings.dark_hint[0]),
            float(self._settings.dark_hint[1]),
        )
        self._circle_center = (
            float(self._settings.circle_center[0]),
            float(self._settings.circle_center[1]),
        )

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

    def _sleep_with_cancel(self, duration_s: float) -> None:
        end_time = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < end_time:
            if not self._running:
                raise RuntimeError("Scan canceled.")
            time.sleep(0.05)

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

        return best_x, best_y

    def _scan_axis(self, axis: str, center: float, fixed: float, rng: float, step: float) -> float:
        if step <= 0:
            raise ValueError("Step must be > 0.")
        if rng <= 0:
            return center
        bound_lo = center - rng
        bound_hi = center + rng
        min_step = max(step * self._MIN_STEP_FRACTION, 0.001)
        total_est = max(1, int(math.ceil(rng / max(min_step, 1e-6))))
        eval_count = 0

        def eval_offset(offset: float) -> float:
            nonlocal eval_count
            if not self._running:
                raise RuntimeError("Scan canceled.")
            if axis == "x":
                off_x, off_y = offset, fixed
            else:
                off_x, off_y = fixed, offset
            score = self._evaluate_offset(off_x, off_y)
            eval_count += 1
            self.progress.emit(min(eval_count, total_est), total_est)
            self.log.emit(
                f"{axis.upper()} {offset:.3f} mm -> dist {score:.3f} px"
            )
            return score

        current = center
        current_score = eval_offset(current)
        if current_score <= self._settings.threshold_px:
            self.log.emit(
                f"{axis.upper()} within threshold at {current:.3f} mm (dist {current_score:.3f} px)"
            )
            return current

        direction = 1.0
        step_size = step
        best_offset = current
        best_score = current_score

        while step_size >= min_step:
            cand = current + direction * step_size
            if cand < bound_lo or cand > bound_hi:
                direction *= -1.0
                step_size *= 0.5
                continue
            score = eval_offset(cand)
            if score <= self._settings.threshold_px:
                self.log.emit(
                    f"{axis.upper()} reached threshold at {cand:.3f} mm (dist {score:.3f} px)"
                )
                return cand
            if score < current_score:
                current = cand
                current_score = score
                if score < best_score:
                    best_score = score
                    best_offset = cand
            else:
                direction *= -1.0
                step_size *= 0.5

        self.log.emit(
            f"Best {axis.upper()} = {best_offset:.3f} mm (dist {best_score:.3f} px)"
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

    def _evaluate_offset(self, offset_x_mm: float, offset_y_mm: float) -> float:
        if not self._running:
            raise RuntimeError("Scan canceled.")
        prev_time = self._camera.get_last_frame_time()
        mask_u8 = self._vortex.build_mask(offset_x_mm, offset_y_mm)
        ok = self._slm.send_mask_to_slot(mask_u8, self._settings.slot)
        if not ok:
            raise RuntimeError("Failed to send mask to SLM.")

        self.log.emit("Waiting for SLM settle...")
        self._sleep_with_cancel(max(2.0, self._settings.settle_ms / 1000.0))

        # Wait briefly for a fresh frame
        timeout = time.monotonic() + 1.0
        while time.monotonic() < timeout:
            if not self._running:
                raise RuntimeError("Scan canceled.")
            if self._camera.get_last_frame_time() > prev_time:
                break
            time.sleep(0.05)

        frame = self._camera.get_last_frame()
        if frame is None:
            raise RuntimeError("No camera frame available.")

        gray = self._to_gray(frame)
        if self._settings.filter_enabled:
            thr = float(self._settings.filter_threshold)
            gray = gray.copy()
            gray[gray < thr] = 0.0
        cx, cy = self._circle_center
        radius = float(self._settings.circle_radius)

        hx, hy = self._current_dark
        dark_center, dark_diam, dark_dbg = OffsetScanWorker._find_dark_spot_cv2(
            gray, (hx, hy), self._DARK_ROI_PX
        )
        if dark_center is None:
            dx = float(hx) - float(cx)
            dy = float(hy) - float(cy)
            base_angle = math.atan2(dy, dx) if (dx != 0.0 or dy != 0.0) else 0.0
        else:
            dx = float(dark_center[0]) - float(cx)
            dy = float(dark_center[1]) - float(cy)
            base_angle = math.atan2(dy, dx) if (dx != 0.0 or dy != 0.0) else 0.0

        t_px, vals = self._sample_line_profile_px(gray, (cx, cy), radius, base_angle)
        if t_px.size == 0:
            raise RuntimeError("Failed to compute cross-section.")
        search_win = float(self._DARK_SEARCH_FRAC) * radius
        mask = np.abs(t_px) <= search_win
        if np.any(mask):
            idx_local = int(np.argmin(vals[mask]))
            idx_full = int(np.where(mask)[0][idx_local])
            t_min = float(t_px[idx_full])
            v_min = float(vals[idx_full])
        else:
            idx_full = int(np.argmin(vals))
            t_min = float(t_px[idx_full])
            v_min = float(vals[idx_full])

        if dark_center is None:
            dark_width = self._estimate_dark_width(t_px, vals, idx_full, radius)
            span = max(6.0, float(dark_width))
            t1, v1 = self._sample_line_profile_span(gray, (hx, hy), span, base_angle)
            t2, v2 = self._sample_line_profile_span(
                gray, (hx, hy), span, base_angle + math.pi / 2.0
            )
            t1_min = float(t1[int(np.argmin(v1))]) if t1.size > 0 else 0.0
            t2_min = float(t2[int(np.argmin(v2))]) if t2.size > 0 else 0.0
            cos_t = math.cos(base_angle)
            sin_t = math.sin(base_angle)
            cos_o = math.cos(base_angle + math.pi / 2.0)
            sin_o = math.sin(base_angle + math.pi / 2.0)
            dark_center = (
                float(hx + cos_t * t1_min + cos_o * t2_min),
                float(hy + sin_t * t1_min + sin_o * t2_min),
            )
            dark_diam = float(dark_width)
        else:
            dark_width = float(dark_diam) if dark_diam is not None else 0.0
        score = float(
            math.hypot(dark_center[0] - float(cx), dark_center[1] - float(cy))
        )
        self._current_dark = dark_center
        self.dark_center.emit(dark_center)
        self._circle_center = (float(dark_center[0]), float(dark_center[1]))

        if self._settings.debug_enabled:
            crop = self._crop_circle_mask(gray, (cx, cy), radius)
            x0c, y0c, _, _ = self._circle_crop_bounds(
                gray.shape[:2], (cx, cy), radius
            )
            center_in_crop = (float(cx - x0c), float(cy - y0c))
            overlay = DonutOptimizationWindow._draw_angle_lines(
                crop,
                center_in_crop,
                radius,
                base_angle,
                max(1, int(self._settings.angles_count)),
            )
            profile_x_mm = t_px * float(self._settings.pixel_size_mm)
            meta = {
                "center": (float(cx), float(cy)),
                "radius": float(radius),
                "profile_x_mm": (float(profile_x_mm[0]), float(profile_x_mm[-1])),
                "dark_offset_px": float(t_min),
                "dark_value": float(v_min),
                "score_px": float(score),
                "dark_center": dark_center,
                "dark_width_px": float(dark_width),
                "dark_diam_px": float(dark_diam) if dark_diam is not None else None,
                "dark_dbg": dark_dbg,
            }
            if self._settings.filter_enabled:
                meta["filter_threshold"] = float(self._settings.filter_threshold)
            self.debug_data.emit(crop, overlay, None, None, meta, (profile_x_mm, vals))

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
            # Be robust to any NaNs/Infs that can appear from sampling/normalization.
            if not np.isfinite(max_per_angle).all():
                max_per_angle = np.nan_to_num(
                    max_per_angle, nan=0.0, posinf=0.0, neginf=0.0
                )
            step = (r_max - r_min) / max(1, (polar.shape[1] - 1))
            peaks_r = r_min + peaks_idx * step

            threshold = np.percentile(max_per_angle, float(thr_pct))
            valid = max_per_angle >= threshold
            if require_non_edge:
                valid = valid & (peaks_idx > 0) & (peaks_idx < (polar.shape[1] - 1))

            n_valid = int(np.count_nonzero(valid))
            min_required = max(3, min(int(angles_count), int(self._ANGLES_FAST)))
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
                "min_required": int(min_required),
            }
            if n_valid >= min_required:
                break

        if last_diag is None or last_diag.get("reason") != "ok":
            diag = last_diag or {}
            raise RuntimeError(
                "Not enough valid angles for manual scoring. "
                "Check `manual_center` / `manual_radius` and ROI cropping. "
                f"diag={diag}"
            )

        peaks_valid = peaks_r[valid].astype(np.float32)
        # Estimate ring center offset from the provided reference center using a simple
        # sinusoidal model: r(theta) = r0 + dx*cos(theta) + dy*sin(theta).
        # Here dx/dy are the center mismatch (in pixels). We score the *magnitude*
        # of that mismatch so the reference center is treated as a hint, not truth.
        fit_r, fit_dx, fit_dy = self._fit_radius_offset(angles[valid], peaks_valid)
        score = float(math.hypot(float(fit_dx), float(fit_dy)))

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
            "peaks_std_px": float(np.std(peaks_valid)),
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
    def _normalize_to_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_max <= img_min:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

    @staticmethod
    def _circle_crop_bounds(
        shape: Tuple[int, int], center: Tuple[float, float], radius: float
    ) -> Tuple[int, int, int, int]:
        h, w = shape
        cx, cy = center
        r = max(1.0, float(radius))
        x0 = int(math.floor(cx - r))
        x1 = int(math.ceil(cx + r))
        y0 = int(math.floor(cy - r))
        y1 = int(math.ceil(cy + r))
        x0c = max(0, x0)
        y0c = max(0, y0)
        x1c = min(w, x1)
        y1c = min(h, y1)
        return x0c, y0c, x1c, y1c

    @staticmethod
    def _crop_circle_mask(
        img: np.ndarray, center: Tuple[float, float], radius: float
    ) -> np.ndarray:
        if img.size == 0:
            return img
        h, w = img.shape[:2]
        x0c, y0c, x1c, y1c = OffsetScanWorker._circle_crop_bounds(
            (h, w), center, radius
        )
        if x1c <= x0c or y1c <= y0c:
            return np.zeros((0, 0), dtype=img.dtype)
        crop = img[y0c:y1c, x0c:x1c].copy()
        yy, xx = np.indices(crop.shape[:2])
        cx, cy = center
        dx = (x0c + xx) - cx
        dy = (y0c + yy) - cy
        r = max(1.0, float(radius))
        mask = (dx * dx + dy * dy) <= (r * r)
        masked = np.zeros_like(crop)
        masked[mask] = crop[mask]
        return masked

    @staticmethod
    def _fit_radius_offset(angles: np.ndarray, radii: np.ndarray) -> tuple[float, float, float]:
        if angles.size < 3:
            return float(np.mean(radii)), 0.0, 0.0
        a = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
        sol, _, _, _ = np.linalg.lstsq(a, radii, rcond=None)
        r0, dx, dy = sol
        return float(r0), float(dx), float(dy)

    @staticmethod
    def _sample_line_profile_px(
        img: np.ndarray,
        center: Tuple[float, float],
        radius: float,
        angle: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if img is None or img.size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        r = max(1.0, float(radius))
        samples = int(max(50, round(2.0 * r)))
        t = np.linspace(-r, r, samples, dtype=np.float32)
        cx, cy = center
        cos_t = math.cos(angle)
        sin_t = math.sin(angle)
        xs = cx + cos_t * t
        ys = cy + sin_t * t
        vals = OffsetScanWorker._bilinear_sample(img.astype(np.float32), xs, ys).ravel()
        return t.astype(np.float32), vals

    @staticmethod
    def _sample_line_profile_span(
        img: np.ndarray,
        center: Tuple[float, float],
        span_px: float,
        angle: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if img is None or img.size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        half = max(1.0, float(span_px) / 2.0)
        samples = int(max(30, round(2.0 * half)))
        t = np.linspace(-half, half, samples, dtype=np.float32)
        cx, cy = center
        cos_t = math.cos(angle)
        sin_t = math.sin(angle)
        xs = cx + cos_t * t
        ys = cy + sin_t * t
        vals = OffsetScanWorker._bilinear_sample(img.astype(np.float32), xs, ys).ravel()
        return t.astype(np.float32), vals

    @staticmethod
    def _estimate_dark_width(
        t_px: np.ndarray,
        vals: np.ndarray,
        idx_min: int,
        radius: float,
    ) -> float:
        if t_px.size < 5 or vals.size != t_px.size:
            return max(10.0, 0.2 * float(radius))
        min_v = float(vals[idx_min])
        baseline = float(np.median(vals))
        if baseline <= min_v:
            return max(10.0, 0.2 * float(radius))
        thr = min_v + 0.3 * (baseline - min_v)
        mask = vals <= thr
        if not np.any(mask):
            return max(10.0, 0.2 * float(radius))
        left = idx_min
        right = idx_min
        while left > 0 and mask[left]:
            left -= 1
        while right < len(mask) - 1 and mask[right]:
            right += 1
        if not mask[left]:
            left = min(len(mask) - 1, left + 1)
        if not mask[right]:
            right = max(0, right - 1)
        width = float(abs(t_px[right] - t_px[left]))
        if width <= 1.0:
            width = max(10.0, 0.2 * float(radius))
        return width

    @staticmethod
    def _find_dark_spot_cv2(
        img: np.ndarray, click: Tuple[float, float], roi_size: int
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float], dict]:
        if cv2 is None:
            return None, None, {"stage": "cv2_missing"}
        return refine_dark_spot(img, click[0], click[1], roi_size=roi_size)


class CostScanWorker(QtCore.QObject):
    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(float, float, str)
    failed = QtCore.Signal(str)
    dark_center = QtCore.Signal(object)
    debug_data = QtCore.Signal(object, object, object, object, object, object)

    def __init__(
        self,
        vortex: VortexWindow,
        slm: SlmControlWindow,
        camera: CameraWindow,
        settings: ScanSettings,
    ) -> None:
        super().__init__()
        self._vortex = vortex
        self._slm = slm
        self._camera = camera
        self._settings = settings
        self._running = True
        self._current_dark = (
            float(self._settings.dark_hint[0]),
            float(self._settings.dark_hint[1]),
        )
        self._circle_center = (
            float(self._settings.circle_center[0]),
            float(self._settings.circle_center[1]),
        )

    @QtCore.Slot()
    def run(self) -> None:
        try:
            best_x, best_y, csv_path = self._run_scan()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(best_x, best_y, csv_path)

    def stop(self) -> None:
        self._running = False

    def _sleep_with_cancel(self, duration_s: float) -> None:
        end_time = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < end_time:
            if not self._running:
                raise RuntimeError("Scan canceled.")
            time.sleep(0.05)

    def _run_scan(self) -> Tuple[float, float, str]:
        return self._run_fast_search()

    def _run_snake_scan(self) -> Tuple[float, float, str]:
        xs, ys, labels, base = self._scan_grid()
        self.log.emit(
            "Snake scan steps "
            f"x_step={self._settings.x_step_mm:.4f} y_step={self._settings.y_step_mm:.4f} "
            f"x_range={self._settings.x_range_mm:.4f} y_range={self._settings.y_range_mm:.4f}"
        )
        total = max(1, len(xs) * len(ys))
        count = 0

        results: list[tuple[float, float, float, float, float]] = []
        best_cost = float("inf")
        best_a = xs[0] if xs else 0.0
        best_b = ys[0] if ys else 0.0

        for row, b_val in enumerate(ys):
            if not self._running:
                raise RuntimeError("Scan canceled.")
            row_xs = xs if (row % 2 == 0) else list(reversed(xs))
            for a_val in row_xs:
                if not self._running:
                    raise RuntimeError("Scan canceled.")
                cost = self._evaluate_cost(a_val, b_val, base)
                results.append(
                    (
                        a_val,
                        b_val,
                        cost,
                        float(self._settings.x_step_mm),
                        float(self._settings.y_step_mm),
                    )
                )
                if cost < best_cost:
                    best_cost = cost
                    best_a, best_b = a_val, b_val
                count += 1
                self.progress.emit(min(count, total), total)
                self.log.emit(
                    f"Scan {labels[0]}={a_val:.3f} {labels[1]}={b_val:.3f} cost={cost:.4f}"
                )

        csv_path = self._write_csv(results, labels)
        self.log.emit(
            f"Scan complete. Best cost={best_cost:.4f} at {labels[0]}={best_a:.3f} {labels[1]}={best_b:.3f}"
        )
        return best_a, best_b, csv_path

    def _run_fast_search(self) -> Tuple[float, float, str]:
        xs, ys, labels, base = self._scan_grid()
        best_x = xs[len(xs) // 2] if xs else 0.0
        best_y = ys[len(ys) // 2] if ys else 0.0
        mode = self._settings.scan_mode

        step_x0 = float(self._settings.x_step_mm)
        step_y0 = float(self._settings.y_step_mm)
        if mode == "spher":
            step_y0 = 0.0
        min_step_user = max(float(self._settings.fast_min_step), 1e-6)
        shrink_factor = max(2.0, float(self._settings.fast_shrink_factor))
        max_step = max(step_x0, step_y0, 1e-9)
        min_scale = min(1.0, min_step_user / max_step)

        results: list[tuple[float, float, float, float, float]] = []
        best_cost = self._evaluate_cost(best_x, best_y, base)
        results.append((best_x, best_y, best_cost, step_x0, step_y0))
        self.log.emit(
            f"Fast search start {labels[0]}={best_x:.3f} {labels[1]}={best_y:.3f} cost={best_cost:.4f}"
        )
        self.log.emit(
            "Fast search steps "
            f"step_x={step_x0:.4f} step_y={step_y0:.4f} "
            f"min_step={min_step_user:.4f} shrink={shrink_factor:.2f}"
        )
        if mode == "spher":
            self.log.emit("Cost metric: (max - noise) / (dark - noise) from crop + dark spot.")
        else:
            self.log.emit("Cost metric: donut radial symmetry + center leakage.")

        directions_count = max(1, int(self._settings.angles_count))
        progress_est = max(1, directions_count * 10)
        count = 1

        scale = 1.0
        last_pass = False
        while scale >= min_scale:
            if not self._running:
                raise RuntimeError("Scan canceled.")
            step_x = step_x0 * scale
            step_y = step_y0 * scale

            cx, cy = self._circle_center
            dx0 = self._current_dark[0] - cx
            dy0 = self._current_dark[1] - cy
            base_angle = math.atan2(dy0, dx0) if (dx0 != 0.0 or dy0 != 0.0) else 0.0

            if mode == "spher":
                angles = [0.0, math.pi]
            else:
                angles = [
                    base_angle + (2.0 * math.pi * i / directions_count)
                    for i in range(directions_count)
                ]

            improved = False
            for angle in angles:
                if not self._running:
                    raise RuntimeError("Scan canceled.")
                dir_x = math.cos(angle)
                dir_y = math.sin(angle)
                cand_x = best_x + dir_x * step_x
                cand_y = best_y + dir_y * step_y
                cost = self._evaluate_cost(cand_x, cand_y, base)
                results.append((cand_x, cand_y, cost, step_x, step_y))
                count += 1
                self.progress.emit(min(count, progress_est), progress_est)
                self.log.emit(
                    f"Scan {labels[0]}={cand_x:.3f} {labels[1]}={cand_y:.3f} "
                    f"cost={cost:.4f} step_x={step_x:.4f} step_y={step_y:.4f} "
                    f"dir={math.degrees(angle):.1f}"
                )
                if cost < best_cost:
                    best_cost = cost
                    best_x, best_y = cand_x, cand_y
                    improved = True

                    # Keep moving in the same direction while improving
                    while True:
                        if not self._running:
                            raise RuntimeError("Scan canceled.")
                        cand_x = best_x + dir_x * step_x
                        cand_y = best_y + dir_y * step_y
                        cost = self._evaluate_cost(cand_x, cand_y, base)
                        results.append((cand_x, cand_y, cost, step_x, step_y))
                        count += 1
                        self.progress.emit(min(count, progress_est), progress_est)
                        self.log.emit(
                            f"Scan {labels[0]}={cand_x:.3f} {labels[1]}={cand_y:.3f} "
                            f"cost={cost:.4f} step_x={step_x:.4f} step_y={step_y:.4f} "
                            f"dir={math.degrees(angle):.1f}"
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_x, best_y = cand_x, cand_y
                            improved = True
                            continue
                        break

                    break

            if improved:
                if last_pass:
                    break
                continue
            scale /= shrink_factor
            if scale < min_scale:
                scale = min_scale
                last_pass = True
            self.log.emit(
                f"Fast search reduce step scale={scale:.4f} "
                f"step_x={step_x0 * scale:.4f} step_y={step_y0 * scale:.4f}"
            )

        csv_path = self._write_csv(results, labels)
        self.log.emit(
            f"Fast search complete. Best cost={best_cost:.4f} at {labels[0]}={best_x:.3f} {labels[1]}={best_y:.3f}"
        )
        return best_x, best_y, csv_path

    def _fast_search_attempt(
        self,
        center: Tuple[float, float],
        step_x: float,
        step_y: float,
        min_step_x: float,
        min_step_y: float,
        bounds_x: Tuple[float, float],
        bounds_y: Tuple[float, float],
        best_cost: float,
        results: list[tuple[float, float, float, float, float]],
        count: int,
        total: int,
        base: dict,
    ) -> Tuple[float, float, float, float, float, int]:
        best_x, best_y = center
        while step_x >= min_step_x or step_y >= min_step_y:
            if not self._running:
                raise RuntimeError("Scan canceled.")
            improved = False

            best_x, best_y, best_cost, step_x, count, improved_x = self._walk_axis(
                axis="x",
                center=(best_x, best_y),
                step=step_x,
                min_step=min_step_x,
                bounds=bounds_x,
                best_cost=best_cost,
                results=results,
                count=count,
                total=total,
                base=base,
                step_pair=(step_x, step_y),
            )
            if step_x < min_step_x:
                step_x = min_step_x

            best_x, best_y, best_cost, step_y, count, improved_y = self._walk_axis(
                axis="y",
                center=(best_x, best_y),
                step=step_y,
                min_step=min_step_y,
                bounds=bounds_y,
                best_cost=best_cost,
                results=results,
                count=count,
                total=total,
                base=base,
                step_pair=(step_x, step_y),
            )
            if step_y < min_step_y:
                step_y = min_step_y

            improved = improved_x or improved_y
            if improved:
                continue

            new_step_x = max(min_step_x, step_x * 0.5) if step_x > min_step_x else step_x
            new_step_y = max(min_step_y, step_y * 0.5) if step_y > min_step_y else step_y
            if new_step_x == step_x and new_step_y == step_y:
                break
            step_x, step_y = new_step_x, new_step_y

        return best_x, best_y, best_cost, step_x, step_y, count

    def _walk_axis(
        self,
        axis: str,
        center: Tuple[float, float],
        step: float,
        min_step: float,
        bounds: Tuple[float, float],
        best_cost: float,
        results: list[tuple[float, float, float, float, float]],
        count: int,
        total: int,
        base: dict,
        step_pair: Tuple[float, float],
    ) -> Tuple[float, float, float, float, int, bool]:
        if not self._running:
            raise RuntimeError("Scan canceled.")
        if step < min_step:
            return center[0], center[1], best_cost, step, count, False

        x, y = center
        improved = False
        direction = 1.0
        for _ in range(2):
            if not self._running:
                raise RuntimeError("Scan canceled.")
            cand = x + direction * step if axis == "x" else y + direction * step
            if cand < bounds[0] or cand > bounds[1]:
                direction *= -1.0
                continue
            cx = cand if axis == "x" else x
            cy = cand if axis == "y" else y
            cost = self._evaluate_cost(cx, cy, base)
            results.append((cx, cy, cost, step_pair[0], step_pair[1]))
            count += 1
            self.progress.emit(min(count, total), total)
            self.log.emit(
                f"Fast {axis.upper()} x={cx:.3f} y={cy:.3f} cost={cost:.4f} step={step:.4f}"
            )
            if cost < best_cost:
                best_cost = cost
                x, y = cx, cy
                improved = True
                break
            direction *= -1.0

        if not improved:
            step *= 0.5
            return x, y, best_cost, step, count, False

        # Continue in the improving direction until it stops improving
        while step >= min_step:
            if not self._running:
                raise RuntimeError("Scan canceled.")
            cand = x + direction * step if axis == "x" else y + direction * step
            if cand < bounds[0] or cand > bounds[1]:
                break
            cx = cand if axis == "x" else x
            cy = cand if axis == "y" else y
            cost = self._evaluate_cost(cx, cy, base)
            results.append((cx, cy, cost, step_pair[0], step_pair[1]))
            count += 1
            self.progress.emit(min(count, total), total)
            self.log.emit(
                f"Fast {axis.upper()} x={cx:.3f} y={cy:.3f} cost={cost:.4f} step={step:.4f}"
            )
            if cost < best_cost:
                best_cost = cost
                x, y = cx, cy
                improved = True
            else:
                direction *= -1.0
                step *= 0.5
                break

        return x, y, best_cost, step, count, improved

    def _evaluate_cost(self, a_val: float, b_val: float, base: dict) -> float:
        if not self._running:
            raise RuntimeError("Scan canceled.")
        prev_time = self._camera.get_last_frame_time()
        mode = self._settings.scan_mode
        if mode == "shift":
            mask_u8 = self._vortex.build_mask_with_params(
                offset_x_mm=a_val, offset_y_mm=b_val
            )
        elif mode == "astig":
            mask_u8 = self._vortex.build_mask_with_params(
                offset_x_mm=base["offset_x_mm"],
                offset_y_mm=base["offset_y_mm"],
                c_astig_v=a_val,
                c_astig_o=b_val,
                c_coma_y=base["c_coma_y"],
                c_coma_x=base["c_coma_x"],
                c_spher=base["c_spher"],
            )
        elif mode == "coma":
            mask_u8 = self._vortex.build_mask_with_params(
                offset_x_mm=base["offset_x_mm"],
                offset_y_mm=base["offset_y_mm"],
                c_astig_v=base["c_astig_v"],
                c_astig_o=base["c_astig_o"],
                c_coma_y=b_val,
                c_coma_x=a_val,
                c_spher=base["c_spher"],
            )
        else:
            mask_u8 = self._vortex.build_mask_with_params(
                offset_x_mm=base["offset_x_mm"],
                offset_y_mm=base["offset_y_mm"],
                c_astig_v=base["c_astig_v"],
                c_astig_o=base["c_astig_o"],
                c_coma_y=base["c_coma_y"],
                c_coma_x=base["c_coma_x"],
                c_spher=a_val,
            )
        ok = self._slm.send_mask_to_slot(mask_u8, self._settings.slot)
        if not ok:
            raise RuntimeError("Failed to send mask to SLM.")

        self._sleep_with_cancel(max(2.0, self._settings.settle_ms / 1000.0))

        timeout = time.monotonic() + 1.0
        while time.monotonic() < timeout:
            if not self._running:
                raise RuntimeError("Scan canceled.")
            if self._camera.get_last_frame_time() > prev_time:
                break
            time.sleep(0.05)

        frame = self._camera.get_last_frame()
        if frame is None:
            raise RuntimeError("No camera frame available.")

        gray = OffsetScanWorker._to_gray(frame)
        raw_gray = gray
        if self._settings.filter_enabled:
            thr = float(self._settings.filter_threshold)
            gray = gray.copy()
            gray[gray < thr] = 0.0

        cx, cy = self._circle_center
        radius = float(self._settings.circle_radius)
        crop = OffsetScanWorker._crop_circle_mask(gray, (cx, cy), radius)
        x0c, y0c, _, _ = OffsetScanWorker._circle_crop_bounds(
            gray.shape[:2], (cx, cy), radius
        )
        dark_center, _, _ = OffsetScanWorker._find_dark_spot_cv2(
            gray, self._current_dark, OffsetScanWorker._DARK_ROI_PX
        )
        if dark_center is not None:
            self._current_dark = (float(dark_center[0]), float(dark_center[1]))
            self.dark_center.emit(self._current_dark)
            self._circle_center = self._current_dark

        dark_x, dark_y = self._current_dark
        center_in_crop = (float(dark_x - x0c), float(dark_y - y0c))

        dark_noise = None
        max_intensity = None
        dark_spot_intensity = None
        ratio = None
        if mode == "spher":
            dark_noise = self._estimate_dark_noise(raw_gray)
            max_intensity = float(np.max(crop)) if crop is not None and crop.size else 0.0
            dark_spot_intensity = self._sample_dark_spot_min(gray, self._current_dark)
            numerator = max_intensity - dark_noise
            denom = dark_spot_intensity - dark_noise
            if numerator <= 0.0:
                cost = float("inf")
            else:
                denom = max(denom, 1e-6)
                ratio = numerator / denom
                cost = 1.0 / max(ratio, 1e-9)
        else:
            cost = self._donut_cost(
                crop,
                center_in_crop,
                max_r=radius,
                num_angles=max(8, int(self._settings.angles_count)),
                num_pts=100,
            )

        if self._settings.debug_enabled:
            base_angle = math.atan2(dark_y - cy, dark_x - cx)
            overlay = DonutOptimizationWindow._draw_angle_lines(
                crop,
                (float(cx - x0c), float(cy - y0c)),
                radius,
                base_angle,
                max(1, int(self._settings.angles_count)),
            )
            pixel_size_mm = float(self._settings.pixel_size_mm)
            profile_x_mm, profile_y = DonutOptimizationWindow._sample_line_profile(
                gray, (float(cx), float(cy)), radius, base_angle, pixel_size_mm
            )
            meta = {
                "center": (float(cx), float(cy)),
                "radius": float(radius),
                "profile_x_mm": (float(profile_x_mm[0]), float(profile_x_mm[-1])),
                "cost": float(cost),
            }
            if mode == "spher":
                meta["dark_noise"] = float(dark_noise) if dark_noise is not None else None
                meta["max_intensity"] = (
                    float(max_intensity) if max_intensity is not None else None
                )
                meta["dark_spot_intensity"] = (
                    float(dark_spot_intensity) if dark_spot_intensity is not None else None
                )
                meta["spher_ratio"] = float(ratio) if ratio is not None else None
            if self._settings.filter_enabled:
                meta["filter_threshold"] = float(self._settings.filter_threshold)
            self.debug_data.emit(crop, overlay, None, None, meta, (profile_x_mm, profile_y))

        return float(cost)

    @staticmethod
    def _estimate_dark_noise(gray: np.ndarray) -> float:
        if gray is None or gray.size == 0:
            return 0.0
        h, w = gray.shape[:2]
        patch = max(5, int(min(h, w) * 0.05))
        x1 = min(w, patch)
        y1 = min(h, patch)
        patches = [
            gray[0:y1, 0:x1],
            gray[0:y1, max(0, w - x1) : w],
            gray[max(0, h - y1) : h, 0:x1],
            gray[max(0, h - y1) : h, max(0, w - x1) : w],
        ]
        vals = [float(np.median(p)) for p in patches if p is not None and p.size]
        if not vals:
            return float(np.median(gray))
        return float(np.median(np.array(vals, dtype=np.float32)))

    @staticmethod
    def _sample_dark_spot_min(
        gray: np.ndarray, center: Tuple[float, float], radius_px: int = 4
    ) -> float:
        if gray is None or gray.size == 0:
            return 0.0
        h, w = gray.shape[:2]
        cx = int(round(center[0]))
        cy = int(round(center[1]))
        r = max(1, int(radius_px))
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            if 0 <= cx < w and 0 <= cy < h:
                return float(gray[cy, cx])
            return float(np.min(gray))
        patch = gray[y0:y1, x0:x1]
        return float(np.min(patch)) if patch.size else float(gray[cy, cx])

    def _scan_grid(self) -> tuple[list[float], list[float], tuple[str, str], dict]:
        off_x, off_y = self._vortex.get_offsets_mm()
        astig_v, astig_o, coma_x, coma_y, spher = self._vortex.get_zernike_values()
        base = {
            "offset_x_mm": off_x,
            "offset_y_mm": off_y,
            "c_astig_v": astig_v,
            "c_astig_o": astig_o,
            "c_coma_x": coma_x,
            "c_coma_y": coma_y,
            "c_spher": spher,
        }

        mode = self._settings.scan_mode
        if mode == "shift":
            center_x, center_y = off_x, off_y
            labels = ("x_mm", "y_mm")
        elif mode == "astig":
            center_x, center_y = astig_v, astig_o
            labels = ("astig_v", "astig_o")
        elif mode == "coma":
            center_x, center_y = coma_x, coma_y
            labels = ("coma_x", "coma_y")
        else:
            center_x, center_y = spher, spher
            labels = ("spher", "fixed")

        xs = OffsetScanWorker._build_offsets(
            center_x, self._settings.x_range_mm, self._settings.x_step_mm
        )
        if mode == "spher":
            ys = [center_y]
        else:
            ys = OffsetScanWorker._build_offsets(
                center_y, self._settings.y_range_mm, self._settings.y_step_mm
            )
        return xs, ys, labels, base

    @staticmethod
    def _donut_cost(
        img_gray: np.ndarray,
        center: Tuple[float, float],
        max_r: float,
        num_angles: int,
        num_pts: int,
    ) -> float:
        if img_gray is None or img_gray.size == 0:
            return float("inf")
        cx, cy = center
        angles = np.linspace(0.0, 2.0 * math.pi, num_angles, endpoint=False)
        profiles = np.zeros((num_pts, num_angles), dtype=np.float32)
        radii = np.linspace(0.0, float(max_r), num_pts, dtype=np.float32)
        for i, theta in enumerate(angles):
            xs = cx + np.cos(theta) * radii
            ys = cy + np.sin(theta) * radii
            profiles[:, i] = OffsetScanWorker._bilinear_sample(
                img_gray.astype(np.float32), xs, ys
            ).ravel()
        max_val = float(np.max(profiles))
        if max_val > 0:
            profiles /= max_val
        radial_variance = np.std(profiles, axis=1)
        asymmetry_score = float(np.sum(radial_variance))
        hole_idx = max(1, int(num_pts * 0.1))
        center_leakage = float(np.mean(profiles[:hole_idx, :]))
        return asymmetry_score + (center_leakage * 10.0)

    @staticmethod
    def _write_csv(
        results: list[tuple[float, float, float, float, float]],
        labels: tuple[str, str],
    ) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = Path.cwd() / f"donut_scan_{ts}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([labels[0], labels[1], "cost", "step_x", "step_y"])
            writer.writerows(results)
        return str(path)

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

    @staticmethod
    def _crop_circle_mask(
        img: np.ndarray, center: Tuple[float, float], radius: float
    ) -> np.ndarray:
        if img.size == 0:
            return img
        h, w = img.shape[:2]
        x0c, y0c, x1c, y1c = OffsetScanWorker._circle_crop_bounds(
            (h, w), center, radius
        )
        if x1c <= x0c or y1c <= y0c:
            return np.zeros((0, 0), dtype=img.dtype)
        crop = img[y0c:y1c, x0c:x1c].copy()
        yy, xx = np.indices(crop.shape[:2])
        cx, cy = center
        dx = (x0c + xx) - cx
        dy = (y0c + yy) - cy
        r = max(1.0, float(radius))
        mask = (dx * dx + dy * dy) <= (r * r)
        masked = np.zeros_like(crop)
        masked[mask] = crop[mask]
        return masked

    @staticmethod
    def _circle_crop_bounds(
        shape: Tuple[int, int], center: Tuple[float, float], radius: float
    ) -> Tuple[int, int, int, int]:
        h, w = shape
        cx, cy = center
        r = max(1.0, float(radius))
        x0 = int(math.floor(cx - r))
        x1 = int(math.ceil(cx + r))
        y0 = int(math.floor(cy - r))
        y1 = int(math.ceil(cy + r))
        x0c = max(0, x0)
        y0c = max(0, y0)
        x1c = min(w, x1)
        y1c = min(h, y1)
        return x0c, y0c, x1c, y1c

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

        self._crop_pixmap: Optional[QtGui.QPixmap] = None
        self._lines_pixmap: Optional[QtGui.QPixmap] = None
        self._profile_pixmap: Optional[QtGui.QPixmap] = None

        layout = QtWidgets.QGridLayout(self)

        self.lbl_crop_title = QtWidgets.QLabel("Circle crop (masked)")
        self.lbl_lines_title = QtWidgets.QLabel("Angle lines")
        self.lbl_profile_title = QtWidgets.QLabel("Cross-section (first line)")
        self.lbl_data_title = QtWidgets.QLabel("Debug data")

        self.lbl_crop = QtWidgets.QLabel()
        self.lbl_lines = QtWidgets.QLabel()
        self.lbl_profile = QtWidgets.QLabel()
        self.txt_data = QtWidgets.QPlainTextEdit(readOnly=True)
        self.txt_data.setMaximumBlockCount(10000)
        self.lbl_crop.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.lbl_crop.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_crop.setMinimumSize(240, 180)
        self.lbl_lines.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.lbl_lines.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_lines.setMinimumSize(240, 180)
        self.lbl_profile.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.lbl_profile.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_profile.setMinimumSize(240, 120)

        self.lbl_stats = QtWidgets.QLabel("")

        layout.addWidget(self.lbl_crop_title, 0, 0)
        layout.addWidget(self.lbl_lines_title, 0, 1)
        layout.addWidget(self.lbl_crop, 1, 0)
        layout.addWidget(self.lbl_lines, 1, 1)
        layout.addWidget(self.lbl_profile_title, 2, 0, 1, 2)
        layout.addWidget(self.lbl_profile, 3, 0, 1, 2)
        layout.addWidget(self.lbl_stats, 4, 0, 1, 2)
        layout.addWidget(self.lbl_data_title, 5, 0, 1, 2)
        layout.addWidget(self.txt_data, 6, 0, 1, 2)

    def update_views(
        self,
        roi_gray: np.ndarray,
        overlay: np.ndarray,
        peaks: np.ndarray,
        valid: np.ndarray,
        meta: dict,
        profile: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        if roi_gray is not None and roi_gray.size > 0:
            self._crop_pixmap = self._gray_to_pixmap(roi_gray)
        else:
            self._crop_pixmap = None
        if overlay is not None and overlay.size > 0:
            self._lines_pixmap = self._gray_to_pixmap(overlay)
        else:
            self._lines_pixmap = None
        self._profile_pixmap = self._plot_profile(profile)
        self._apply_scaled()

        if meta:
            center = meta.get("center")
            radius = meta.get("radius")
            if center is not None and radius is not None:
                self.lbl_stats.setText(
                    f"center=({center[0]:.2f}, {center[1]:.2f}) radius={radius:.2f}"
                )
            else:
                self.lbl_stats.setText("No circle metadata.")
        else:
            self.lbl_stats.setText("No metadata.")

        self._update_data_table(peaks, valid, meta)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_scaled()

    def _apply_scaled(self) -> None:
        if self._crop_pixmap is not None:
            self.lbl_crop.setPixmap(self._scaled(self._crop_pixmap, self.lbl_crop))
        else:
            self.lbl_crop.clear()
        if self._lines_pixmap is not None:
            self.lbl_lines.setPixmap(self._scaled(self._lines_pixmap, self.lbl_lines))
        else:
            self.lbl_lines.clear()
        if self._profile_pixmap is not None:
            self.lbl_profile.setPixmap(self._scaled(self._profile_pixmap, self.lbl_profile))
        else:
            self.lbl_profile.clear()

    def _update_data_table(
        self, peaks: np.ndarray, valid: np.ndarray, meta: Optional[dict]
    ) -> None:
        if meta is None:
            self.txt_data.setPlainText("No data.")
            return
        lines = []
        if "center" in meta and "radius" in meta:
            cx, cy = meta["center"]
            radius = meta["radius"]
            r_min = meta.get("r_min")
            r_max = meta.get("r_max")
            if r_min is None or r_max is None:
                lines.append(
                    f"center=({cx:.2f}, {cy:.2f}) radius={radius:.2f}"
                )
            else:
                lines.append(
                    f"center=({cx:.2f}, {cy:.2f}) radius={radius:.2f} "
                    f"r_min={r_min:.2f} r_max={r_max:.2f}"
                )
        if "fit_center" in meta:
            fx, fy = meta["fit_center"]
            lines.append(f"fit_center=({fx:.2f}, {fy:.2f})")

        angles = meta.get("angles_deg")
        max_per = meta.get("max_per_angle")
        if peaks is not None and angles is not None:
            if valid is None or len(valid) != len(peaks):
                valid = np.ones(len(peaks), dtype=bool)
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

        if meta.get("profile_x_mm") is not None:
            x0, x1 = meta["profile_x_mm"]
            lines.append(f"profile_x_mm=({x0:.3f}, {x1:.3f})")
        if meta.get("dark_offset_px") is not None:
            lines.append(f"dark_offset_px={meta['dark_offset_px']:.3f}")
        if meta.get("dark_width_px") is not None:
            lines.append(f"dark_width_px={meta['dark_width_px']:.3f}")
        if meta.get("dark_diam_px") is not None:
            lines.append(f"dark_diam_px={meta['dark_diam_px']:.3f}")
        if meta.get("dark_dbg") is not None:
            lines.append(f"dark_dbg={meta['dark_dbg']}")
        if meta.get("filter_threshold") is not None:
            lines.append(f"filter_threshold={meta['filter_threshold']}")
        if meta.get("cost") is not None:
            lines.append(f"cost={meta['cost']:.4f}")
        if meta.get("dark_noise") is not None:
            lines.append(f"dark_noise={meta['dark_noise']:.2f}")
        if meta.get("max_intensity") is not None:
            lines.append(f"max_intensity={meta['max_intensity']:.2f}")
        if meta.get("dark_spot_intensity") is not None:
            lines.append(f"dark_spot_intensity={meta['dark_spot_intensity']:.2f}")
        if meta.get("spher_ratio") is not None:
            lines.append(f"spher_ratio={meta['spher_ratio']:.4f}")
        if meta.get("score_px") is not None:
            lines.append(f"score_px={meta['score_px']:.3f}")

        self.txt_data.setPlainText("\n".join(lines) if lines else "No data.")

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
    def _plot_profile(
        profile: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[QtGui.QPixmap]:
        if profile is None:
            return None
        xs, ys = profile
        xs = np.asarray(xs, dtype=np.float32).ravel()
        vals = np.asarray(ys, dtype=np.float32).ravel()
        if vals.size < 2 or xs.size != vals.size:
            return None
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        xmin = float(xs[0])
        xmax = float(xs[-1])
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

        w = max(360, int(vals.size))
        h = 180
        left = 40
        right = 10
        top = 10
        bottom = 25
        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)

        img = np.full((h, w, 3), 255, dtype=np.uint8)
        # Axes
        DebugWindow._draw_line(img, left, top, left, top + plot_h, (0, 0, 0))
        DebugWindow._draw_line(
            img, left, top + plot_h, left + plot_w, top + plot_h, (0, 0, 0)
        )
        if xmax > xmin:
            x0 = int(round(left + plot_w * (-xmin / (xmax - xmin))))
            if left <= x0 <= left + plot_w:
                DebugWindow._draw_line(img, x0, top, x0, top + plot_h, (220, 220, 220))

        if vmax <= vmin:
            y = int(top + plot_h / 2)
            DebugWindow._draw_line(img, left, y, left + plot_w, y, (0, 0, 0))
            return DebugWindow._gray_to_pixmap(img)

        xs_plot = np.linspace(0, vals.size - 1, plot_w)
        vals_plot = np.interp(xs_plot, np.arange(vals.size), vals)
        y_plot = (plot_h - 1) - (vals_plot - vmin) / (vmax - vmin) * (plot_h - 1)
        for i in range(1, plot_w):
            DebugWindow._draw_line(
                img,
                left + i - 1,
                top + y_plot[i - 1],
                left + i,
                top + y_plot[i],
                (0, 0, 0),
            )
        return DebugWindow._gray_to_pixmap(img)

    @staticmethod
    def _draw_line(
        img: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: Tuple[int, int, int],
    ) -> None:
        h, w = img.shape[:2]
        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy))) + 1
        if steps <= 0:
            return
        xs = np.linspace(x0, x1, steps).astype(np.int32)
        ys = np.linspace(y0, y1, steps).astype(np.int32)
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        img[ys[mask], xs[mask]] = color

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
        # User hints (not assumed exact)
        self._manual_center: Optional[Tuple[float, float]] = None  # "dark spot" hint
        self._circle_center: Optional[Tuple[float, float]] = None  # circle center hint
        self._manual_radius: Optional[float] = None  # circle radius

        self.setWindowTitle("Donut Optimization Wizard")
        self.resize(600, 520)

        self._build_ui()
        self._camera.point_selected.connect(self._on_point_selected)
        self._camera.circle_selected.connect(self._on_circle_selected)
        self._restore_settings()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        manual_group = QtWidgets.QGroupBox("1) Manual Target")
        manual_layout = QtWidgets.QGridLayout(manual_group)
        self.lbl_manual_center = QtWidgets.QLabel("Hint (dark spot): not set")
        self.lbl_manual_radius = QtWidgets.QLabel("Circle: not set")
        self.btn_pick_center = QtWidgets.QPushButton("Pick dark spot")
        self.btn_pick_center.setToolTip("Click the dark spot center in the camera view.")
        self.btn_pick_center.clicked.connect(self._pick_dark_spot)
        self.btn_pick_circle = QtWidgets.QPushButton("Draw donut circle")
        self.btn_pick_circle.setToolTip("Draw the donut circle on the camera view.")
        self.btn_pick_circle.clicked.connect(self._pick_donut_circle)
        self.btn_analyze = QtWidgets.QPushButton("Donut analysis")
        self.btn_analyze.setToolTip("Analyze a single frame and update the debug plots.")
        self.btn_analyze.clicked.connect(self._run_donut_analysis)
        self.btn_refine_dark = QtWidgets.QPushButton("Refine dark spot")
        self.btn_refine_dark.setToolTip("Refine the dark spot center using CV.")
        self.btn_refine_dark.clicked.connect(self._refine_dark_spot)
        self.dsb_px_um = QtWidgets.QDoubleSpinBox()
        self.dsb_px_um.setRange(0.01, 1000.0)
        self.dsb_px_um.setDecimals(4)
        self.dsb_px_um.setValue(3.45)
        self.dsb_px_um.setSuffix(" um/px")
        self.chk_filter = QtWidgets.QCheckBox("Zero below threshold")
        self.chk_filter.setChecked(False)
        self.chk_filter.setToolTip("Zero pixels below the threshold before analysis.")
        self.spin_filter = QtWidgets.QSpinBox()
        self.spin_filter.setRange(0, 255)
        self.spin_filter.setValue(10)
        self.btn_clear_manual = QtWidgets.QPushButton("Clear")
        self.btn_clear_manual.setToolTip("Clear the selected dark spot and circle.")
        self.btn_clear_manual.clicked.connect(self._clear_manual_target)
        self.dsb_auto_circle_mm = QtWidgets.QDoubleSpinBox()
        self.dsb_auto_circle_mm.setRange(0.000001, 1000.0)
        self.dsb_auto_circle_mm.setDecimals(6)
        self.dsb_auto_circle_mm.setValue(1.000000)
        self.dsb_auto_circle_mm.setSuffix(" mm")

        manual_layout.addWidget(self.lbl_manual_center, 0, 0, 1, 2)
        manual_layout.addWidget(self.lbl_manual_radius, 1, 0, 1, 2)
        manual_layout.addWidget(self.btn_pick_center, 0, 2)
        manual_layout.addWidget(self.btn_pick_circle, 1, 2)
        manual_layout.addWidget(QtWidgets.QLabel("Camera pixel size"), 2, 0)
        manual_layout.addWidget(self.dsb_px_um, 2, 1)
        manual_layout.addWidget(self.btn_clear_manual, 0, 3, 2, 1)
        manual_layout.addWidget(self.chk_filter, 3, 0)
        manual_layout.addWidget(self.spin_filter, 3, 1)
        manual_layout.addWidget(QtWidgets.QLabel("Auto circle radius"), 3, 2)
        manual_layout.addWidget(self.dsb_auto_circle_mm, 3, 3)
        manual_layout.addWidget(self.btn_refine_dark, 2, 2, 1, 2)
        manual_layout.addWidget(self.btn_analyze, 4, 0, 1, 4)
        layout.addWidget(manual_group)

        scan_group = QtWidgets.QGroupBox("2) Scan Settings (mm)")
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

        self.spin_settle = QtWidgets.QSpinBox()
        self.spin_settle.setRange(100, 10000)
        self.spin_settle.setValue(500)
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
        self.spin_angles.setRange(2, 360)
        self.spin_angles.setValue(10)
        scan_layout.addWidget(QtWidgets.QLabel("Angles"), r, 0)
        scan_layout.addWidget(self.spin_angles, r, 1)
        r += 1
        scan_layout.addWidget(QtWidgets.QLabel("Fast min step"), r, 0)
        self.dsb_fast_min_step = QtWidgets.QDoubleSpinBox()
        self.dsb_fast_min_step.setRange(0.0001, 5.0)
        self.dsb_fast_min_step.setDecimals(4)
        self.dsb_fast_min_step.setValue(0.01)
        self.dsb_fast_min_step.setSuffix(" mm")
        scan_layout.addWidget(self.dsb_fast_min_step, r, 1)
        scan_layout.addWidget(QtWidgets.QLabel("Shrink factor"), r, 2)
        self.dsb_fast_shrink = QtWidgets.QDoubleSpinBox()
        self.dsb_fast_shrink.setRange(2.0, 10.0)
        self.dsb_fast_shrink.setDecimals(2)
        self.dsb_fast_shrink.setValue(2.0)
        self.dsb_fast_shrink.setToolTip("Step size is divided by this factor each reduction.")
        scan_layout.addWidget(self.dsb_fast_shrink, r, 3)
        r += 1
        self.rb_scan_shift = QtWidgets.QRadioButton("Shift")
        self.rb_scan_astig = QtWidgets.QRadioButton("Astigmatism")
        self.rb_scan_coma = QtWidgets.QRadioButton("Coma")
        self.rb_scan_spher = QtWidgets.QRadioButton("Spherical")
        self.rb_scan_shift.setToolTip("Optimize X/Y shift offsets.")
        self.rb_scan_astig.setToolTip("Optimize astigmatism coefficients.")
        self.rb_scan_coma.setToolTip("Optimize coma coefficients.")
        self.rb_scan_spher.setToolTip("Optimize spherical coefficient.")
        self.rb_scan_shift.setChecked(True)
        self.scan_group = QtWidgets.QButtonGroup(self)
        for rb in (
            self.rb_scan_shift,
            self.rb_scan_astig,
            self.rb_scan_coma,
            self.rb_scan_spher,
        ):
            self.scan_group.addButton(rb)
        scan_layout.addWidget(QtWidgets.QLabel("Scan mode"), r, 0)
        scan_layout.addWidget(self.rb_scan_shift, r, 1)
        scan_layout.addWidget(self.rb_scan_astig, r, 2)
        scan_layout.addWidget(self.rb_scan_coma, r, 3)
        r += 1
        scan_layout.addWidget(self.rb_scan_spher, r, 1)

        layout.addWidget(scan_group)

        btns = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_start.setToolTip("Start the optimization scan.")
        self.btn_stop.setToolTip("Stop the current scan.")
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
        self.chk_debug.setToolTip("Enable debug plots during scanning.")
        self.btn_debug.setToolTip("Open the donut optimization debug window.")
        self.btn_debug.clicked.connect(self._open_debug)
        dbg = QtWidgets.QHBoxLayout()
        dbg.addWidget(self.chk_debug)
        dbg.addWidget(self.btn_debug)
        dbg.addStretch(1)
        layout.addLayout(dbg)

    def _pick_dark_spot(self) -> None:
        if not self._camera.is_running():
            self._append_error("Camera must be running.")
            return
        self._camera.begin_point_selection()

    def _pick_donut_circle(self) -> None:
        if not self._camera.is_running():
            self._append_error("Camera must be running.")
            return
        # Circle selection should be independent from the picked dark-spot center.
        self._camera.begin_circle_selection()

    def _clear_manual_target(self) -> None:
        self._manual_center = None
        self._circle_center = None
        self._manual_radius = None
        self._update_manual_labels()
        self._camera.clear_manual_marks()
        self._save_settings()

    def _on_point_selected(self, point: Tuple[float, float]) -> None:
        self._manual_center = (float(point[0]), float(point[1]))
        self._update_manual_labels()
        self._append_log(
            f"Dark spot set: ({self._manual_center[0]:.1f}, {self._manual_center[1]:.1f})"
        )
        self._refine_dark_spot()
        self._save_settings()

    def _on_circle_selected(self, circle: Tuple[float, float, float]) -> None:
        cx, cy, r = circle
        self._circle_center = (float(cx), float(cy))
        self._manual_radius = float(r)
        self._update_manual_labels()
        self._append_log(
            f"Donut circle set: radius={r:.1f} px (drawn at ({cx:.1f}, {cy:.1f}))"
        )
        self._save_settings()

    def _update_manual_labels(self) -> None:
        if self._manual_center is None:
            self.lbl_manual_center.setText("Hint (dark spot): not set")
        else:
            self.lbl_manual_center.setText(
                f"Hint (dark spot): ({self._manual_center[0]:.1f}, {self._manual_center[1]:.1f})"
            )
        if self._manual_radius is None:
            self.lbl_manual_radius.setText("Circle: not set")
        else:
            if self._circle_center is None:
                self.lbl_manual_radius.setText(f"Circle radius: {self._manual_radius:.1f} px")
            else:
                self.lbl_manual_radius.setText(
                    f"Circle: center=({self._circle_center[0]:.1f},{self._circle_center[1]:.1f}), "
                    f"r={self._manual_radius:.1f} px"
                )

    def _run_donut_analysis(self) -> None:
        if not self._camera.is_running():
            self._append_error("Camera must be running.")
            return
        if self._manual_center is None:
            QtWidgets.QMessageBox.information(
                self,
                "Select dark spot",
                "Please click the dark spot center in the camera view, then run Donut analysis again.",
            )
            self._camera.begin_point_selection()
            return
        if self._circle_center is None or self._manual_radius is None:
            self._append_error("Please draw the donut circle.")
            return
        frame = self._camera.get_last_frame()
        if frame is None:
            self._append_error("No camera frame available.")
            return
        gray = OffsetScanWorker._to_gray(frame)
        gray = self._apply_filter(gray)
        cx, cy = self._circle_center
        radius = float(self._manual_radius)
        crop = OffsetScanWorker._crop_circle_mask(gray, (cx, cy), radius)
        x0c, y0c, _, _ = OffsetScanWorker._circle_crop_bounds(
            gray.shape[:2], (cx, cy), radius
        )
        center_in_crop = (float(cx - x0c), float(cy - y0c))
        dx = float(self._manual_center[0]) - float(cx)
        dy = float(self._manual_center[1]) - float(cy)
        base_angle = math.atan2(dy, dx) if (dx != 0.0 or dy != 0.0) else 0.0
        angles_count = max(1, int(self.spin_angles.value()))
        overlay = self._draw_angle_lines(
            crop, center_in_crop, radius, base_angle, angles_count
        )
        pixel_size_mm = float(self.dsb_px_um.value()) * 1e-3
        profile_x_mm, profile_y = self._sample_line_profile(
            gray, (float(cx), float(cy)), radius, base_angle, pixel_size_mm
        )
        meta = {
            "center": (float(cx), float(cy)),
            "radius": float(radius),
            "r_min": 0.0,
            "r_max": float(radius),
            "profile_x_mm": (float(profile_x_mm[0]), float(profile_x_mm[-1])),
        }
        if self.chk_filter.isChecked():
            meta["filter_threshold"] = int(self.spin_filter.value())
        if self._debug_window is None:
            self._debug_window = DebugWindow(self)
        self._debug_window.update_views(
            crop, overlay, None, None, meta, (profile_x_mm, profile_y)
        )
        self._debug_window.show()
        self._debug_window.raise_()
        self._debug_window.activateWindow()
        self._append_log("Donut analysis: plotted circle crop.")
        self._save_settings()

    def _refine_dark_spot(self) -> None:
        if not self._camera.is_running():
            self._append_error("Camera must be running.")
            return
        if self._manual_center is None:
            self._append_error("Please pick the dark spot center first.")
            return
        frame = self._camera.get_last_frame()
        if frame is None:
            self._append_error("No camera frame available.")
            return
        gray = OffsetScanWorker._to_gray(frame)
        gray = self._apply_filter(gray)
        dark_center, _, dark_dbg = OffsetScanWorker._find_dark_spot_cv2(
            gray, self._manual_center, OffsetScanWorker._DARK_ROI_PX
        )
        if dark_center is None:
            self._append_error(f"Dark spot refinement failed. dbg={dark_dbg}")
            return
        self._manual_center = (float(dark_center[0]), float(dark_center[1]))
        self._update_manual_labels()
        self._camera.set_selected_point(self._manual_center, emit=False)
        self._append_log(
            f"Dark spot refined: ({self._manual_center[0]:.1f}, {self._manual_center[1]:.1f})"
        )
        self._save_settings()

    def _apply_filter(self, img: np.ndarray) -> np.ndarray:
        if not self.chk_filter.isChecked():
            return img
        thr = float(self.spin_filter.value())
        out = img.copy()
        out[out < thr] = 0.0
        return out

    def _auto_set_circle_from_dark(self) -> bool:
        if self._manual_center is None:
            self._append_error("Please pick the dark spot center.")
            return False
        radius_mm = float(self.dsb_auto_circle_mm.value())
        if radius_mm <= 0:
            self._append_error("Auto circle radius must be > 0.")
            return False
        pixel_size_mm = float(self.dsb_px_um.value()) * 1e-3
        if pixel_size_mm <= 0:
            self._append_error("Camera pixel size must be > 0.")
            return False
        radius_px = radius_mm / pixel_size_mm
        self._circle_center = (float(self._manual_center[0]), float(self._manual_center[1]))
        self._manual_radius = float(radius_px)
        self._update_manual_labels()
        if hasattr(self._camera, "set_selected_circle"):
            self._camera.set_selected_circle(
                self._circle_center, self._manual_radius, emit=False
            )
        self._append_log(
            f"Auto donut circle set: radius={radius_px:.1f} px ({radius_mm:.6f} mm)"
        )
        self._save_settings()
        return True

    @staticmethod
    def _draw_angle_lines(
        crop: np.ndarray,
        center: Tuple[float, float],
        radius: float,
        base_angle: float,
        angles_count: int,
    ) -> np.ndarray:
        if crop is None or crop.size == 0:
            return crop
        base = OffsetScanWorker._normalize_to_u8(crop)
        color = np.repeat(base[:, :, None], 3, axis=2)
        cx, cy = center
        r = max(1.0, float(radius))
        for i in range(angles_count):
            # Use half-turn spacing so each diameter is unique (theta and theta+pi are the same line).
            angle = base_angle + (math.pi * i / angles_count)
            dx = math.cos(angle) * r
            dy = math.sin(angle) * r
            x0 = cx - dx
            y0 = cy - dy
            x1 = cx + dx
            y1 = cy + dy
            line_color = (0, 0, 255) if i == 0 else (0, 255, 255)
            DonutOptimizationWindow._draw_line(color, x0, y0, x1, y1, line_color)
        return color

    @staticmethod
    def _sample_line_profile(
        img: np.ndarray,
        center: Tuple[float, float],
        radius: float,
        angle: float,
        pixel_size_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if img is None or img.size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        r = max(1.0, float(radius))
        samples = int(max(50, round(2.0 * r)))
        t = np.linspace(-r, r, samples, dtype=np.float32)
        cx, cy = center
        cos_t = math.cos(angle)
        sin_t = math.sin(angle)
        xs = cx + cos_t * t
        ys = cy + sin_t * t
        vals = OffsetScanWorker._bilinear_sample(img.astype(np.float32), xs, ys).ravel()
        x_mm = t * float(pixel_size_mm)
        return x_mm.astype(np.float32), vals

    @staticmethod
    def _draw_line(
        img: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: Tuple[int, int, int],
    ) -> None:
        h, w = img.shape[:2]
        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy))) + 1
        if steps <= 0:
            return
        xs = np.linspace(x0, x1, steps).astype(np.int32)
        ys = np.linspace(y0, y1, steps).astype(np.int32)
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        img[ys[mask], xs[mask]] = color

    def _start(self) -> None:
        if not self._camera.is_running():
            self._append_error("Camera must be running.")
            return
        if self._manual_center is None:
            self._append_error("Please pick the dark spot center.")
            return
        if self._circle_center is None or self._manual_radius is None:
            if not self._auto_set_circle_from_dark():
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
            settle_ms=int(self.spin_settle.value()),
            slot=int(self.spin_slot.value()),
            debug_enabled=self.chk_debug.isChecked(),
            angles_count=int(self.spin_angles.value()),
            circle_center=(float(self._circle_center[0]), float(self._circle_center[1])),
            circle_radius=float(self._manual_radius),
            dark_hint=(float(self._manual_center[0]), float(self._manual_center[1])),
            pixel_size_mm=float(self.dsb_px_um.value()) * 1e-3,
            threshold_px=OffsetScanWorker._TARGET_DIST_PX,
            filter_enabled=self.chk_filter.isChecked(),
            filter_threshold=float(self.spin_filter.value()),
            fast_search=True,
            fast_min_step=float(self.dsb_fast_min_step.value()),
            fast_shrink_factor=float(self.dsb_fast_shrink.value()),
            fast_multi_pass=True,
            scan_mode=self._get_scan_mode(),
        )

        self._thread = QtCore.QThread(self)
        self._worker = CostScanWorker(self._vortex, self._slm, self._camera, settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.debug_data.connect(self._on_debug_data)
        self._worker.dark_center.connect(self._on_dark_center_update)
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

    def _on_scan_finished(self, best_x: float, best_y: float, csv_path: str) -> None:
        mode = self._get_scan_mode()
        if mode == "shift":
            msg = f"Scan complete. Best X={best_x:.3f} mm, Y={best_y:.3f} mm"
        elif mode == "astig":
            msg = f"Scan complete. Best astig_v={best_x:.3f}, astig_o={best_y:.3f}"
        elif mode == "coma":
            msg = f"Scan complete. Best coma_x={best_x:.3f}, coma_y={best_y:.3f}"
        else:
            msg = f"Scan complete. Best spherical={best_x:.3f}"
        self._append_log(msg)
        self._append_log(f"Saved scan CSV: {csv_path}")
        if mode == "shift":
            self._vortex.set_offsets_mm(best_x, best_y)
        elif mode == "astig":
            self._vortex.set_zernike_values(astig_v=best_x, astig_o=best_y)
        elif mode == "coma":
            self._vortex.set_zernike_values(coma_x=best_x, coma_y=best_y)
        else:
            self._vortex.set_zernike_values(spher=best_x)
        self._stop()

    def _on_failed(self, msg: str) -> None:
        self._append_error(msg)
        self._stop()

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _append_error(self, text: str) -> None:
        self.log.appendPlainText("ERROR: " + text)

    def _restore_settings(self) -> None:
        settings = QtCore.QSettings()
        settings.beginGroup("donut_opt_window")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        self.dsb_px_um.setValue(float(settings.value("pixel_um", self.dsb_px_um.value())))
        self.chk_filter.setChecked(bool(settings.value("filter_enabled", False, bool)))
        self.spin_filter.setValue(int(settings.value("filter_threshold", self.spin_filter.value())))
        self.dsb_auto_circle_mm.setValue(
            float(settings.value("auto_circle_radius_mm", self.dsb_auto_circle_mm.value()))
        )

        self.dsb_x_range.setValue(float(settings.value("x_range_mm", self.dsb_x_range.value())))
        self.dsb_x_step.setValue(float(settings.value("x_step_mm", self.dsb_x_step.value())))
        self.dsb_y_range.setValue(float(settings.value("y_range_mm", self.dsb_y_range.value())))
        self.dsb_y_step.setValue(float(settings.value("y_step_mm", self.dsb_y_step.value())))
        self.spin_settle.setValue(int(settings.value("settle_ms", self.spin_settle.value())))
        self.spin_slot.setValue(int(settings.value("slot", self.spin_slot.value())))
        self.spin_angles.setValue(int(settings.value("angles_count", self.spin_angles.value())))
        self.dsb_fast_min_step.setValue(
            float(settings.value("fast_min_step", self.dsb_fast_min_step.value()))
        )
        self.dsb_fast_shrink.setValue(
            float(settings.value("fast_shrink", self.dsb_fast_shrink.value()))
        )
        self.chk_debug.setChecked(bool(settings.value("debug_enabled", False, bool)))

        mode = settings.value("scan_mode", "shift")
        if mode == "astig":
            self.rb_scan_astig.setChecked(True)
        elif mode == "coma":
            self.rb_scan_coma.setChecked(True)
        elif mode == "spher":
            self.rb_scan_spher.setChecked(True)
        else:
            self.rb_scan_shift.setChecked(True)

        if settings.contains("manual_center_x") and settings.contains("manual_center_y"):
            self._manual_center = (
                float(settings.value("manual_center_x")),
                float(settings.value("manual_center_y")),
            )
        if settings.contains("circle_center_x") and settings.contains("circle_center_y"):
            self._circle_center = (
                float(settings.value("circle_center_x")),
                float(settings.value("circle_center_y")),
            )
        if settings.contains("circle_radius_px"):
            self._manual_radius = float(settings.value("circle_radius_px"))

        self._update_manual_labels()
        if self._manual_center is not None:
            self._camera.set_selected_point(self._manual_center, emit=False)
        if (
            self._circle_center is not None
            and self._manual_radius is not None
            and hasattr(self._camera, "set_selected_circle")
        ):
            self._camera.set_selected_circle(
                self._circle_center, self._manual_radius, emit=False
            )

        debug_visible = settings.value("debug_visible", False, bool)
        debug_geometry = settings.value("debug_geometry")
        settings.endGroup()

        if debug_visible:
            if self._debug_window is None:
                self._debug_window = DebugWindow(self)
            if debug_geometry:
                self._debug_window.restoreGeometry(debug_geometry)
            self._debug_window.show()
            self._debug_window.raise_()
            self._debug_window.activateWindow()

    def _save_settings(self) -> None:
        settings = QtCore.QSettings()
        settings.beginGroup("donut_opt_window")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("pixel_um", float(self.dsb_px_um.value()))
        settings.setValue("filter_enabled", self.chk_filter.isChecked())
        settings.setValue("filter_threshold", int(self.spin_filter.value()))
        settings.setValue("auto_circle_radius_mm", float(self.dsb_auto_circle_mm.value()))
        settings.setValue("x_range_mm", float(self.dsb_x_range.value()))
        settings.setValue("x_step_mm", float(self.dsb_x_step.value()))
        settings.setValue("y_range_mm", float(self.dsb_y_range.value()))
        settings.setValue("y_step_mm", float(self.dsb_y_step.value()))
        settings.setValue("settle_ms", int(self.spin_settle.value()))
        settings.setValue("slot", int(self.spin_slot.value()))
        settings.setValue("angles_count", int(self.spin_angles.value()))
        settings.setValue("fast_min_step", float(self.dsb_fast_min_step.value()))
        settings.setValue("fast_shrink", float(self.dsb_fast_shrink.value()))
        settings.setValue("debug_enabled", self.chk_debug.isChecked())
        settings.setValue("scan_mode", self._get_scan_mode())
        settings.setValue("visible", self.isVisible())

        if self._manual_center is None:
            settings.remove("manual_center_x")
            settings.remove("manual_center_y")
        else:
            settings.setValue("manual_center_x", float(self._manual_center[0]))
            settings.setValue("manual_center_y", float(self._manual_center[1]))

        if self._circle_center is None:
            settings.remove("circle_center_x")
            settings.remove("circle_center_y")
        else:
            settings.setValue("circle_center_x", float(self._circle_center[0]))
            settings.setValue("circle_center_y", float(self._circle_center[1]))

        if self._manual_radius is None:
            settings.remove("circle_radius_px")
        else:
            settings.setValue("circle_radius_px", float(self._manual_radius))

        if self._debug_window is not None:
            settings.setValue("debug_visible", self._debug_window.isVisible())
            settings.setValue("debug_geometry", self._debug_window.saveGeometry())
        settings.endGroup()

    def _get_scan_mode(self) -> str:
        if self.rb_scan_astig.isChecked():
            return "astig"
        if self.rb_scan_coma.isChecked():
            return "coma"
        if self.rb_scan_spher.isChecked():
            return "spher"
        return "shift"

    def _open_debug(self) -> None:
        if self._debug_window is None:
            self._debug_window = DebugWindow(self)
        self._debug_window.show()
        self._debug_window.raise_()
        self._debug_window.activateWindow()
        self._save_settings()

    def _on_debug_data(
        self,
        roi_gray: np.ndarray,
        polar: np.ndarray,
        peaks: np.ndarray,
        valid: np.ndarray,
        meta: dict,
        profile: object,
    ) -> None:
        if self._debug_window is None:
            self._debug_window = DebugWindow(self)
        if self.chk_debug.isChecked():
            self._debug_window.update_views(roi_gray, polar, peaks, valid, meta, profile)

    def _on_dark_center_update(self, point: Tuple[float, float]) -> None:
        self._manual_center = (float(point[0]), float(point[1]))
        if self._manual_radius is not None:
            self._circle_center = (float(point[0]), float(point[1]))
        self._update_manual_labels()
        self._camera.set_selected_point(self._manual_center, emit=False)
        if self._manual_radius is not None and hasattr(self._camera, "set_selected_circle"):
            self._camera.set_selected_circle(self._circle_center, self._manual_radius, emit=False)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._save_settings()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        self._save_settings()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_settings()
        if QtCore.QCoreApplication.closingDown():
            event.accept()
            return
        event.ignore()
        self.hide()
