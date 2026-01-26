from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

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


class OffsetScanWorker(QtCore.QObject):
    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(float, float)
    failed = QtCore.Signal(str)
    debug_data = QtCore.Signal(object, object, object, object)

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
        offsets = self._build_offsets(center, rng, step)
        total = len(offsets)
        best_offset = center
        best_score = float("inf")

        for idx, offset in enumerate(offsets, start=1):
            if not self._running:
                raise RuntimeError("Scan canceled.")
            if axis == "x":
                off_x, off_y = offset, fixed
            else:
                off_x, off_y = fixed, offset
            score = self._evaluate_offset(off_x, off_y)
            self.log.emit(f"{axis.upper()} {offset:.3f} mm -> score {score:.3f}")
            if score < best_score:
                best_score = score
                best_offset = offset
            self.progress.emit(idx, total)

        self.log.emit(f"Best {axis.upper()} = {best_offset:.3f} mm (score {best_score:.3f})")
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

        if cv2 is not None:
            try:
                score, polar, peaks, valid = self._score_warp_polar(roi_gray)
                if self._settings.debug_enabled:
                    self.debug_data.emit(roi_gray, polar, peaks, valid)
                self.log.emit(f"warpPolar score={score:.3f}")
                return float(score)
            except Exception as exc:
                self.log.emit(f"warpPolar failed: {exc}; falling back to circle fit.")

        score = self._score_center_distance(roi_gray)
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
        img = roi_gray.astype(np.uint8)
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Dark core (hole) detection via percentile threshold
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

        # Ring center via edge points and circle fit
        v = np.median(blur)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blur, lower, upper)
        ys, xs = np.where(edges > 0)
        if len(xs) < 30:
            raise RuntimeError("Not enough edge points for ring fit.")
        rx, ry, _ = self._fit_circle(xs, ys)

        return (hx, hy), (rx, ry)

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

    def _score_warp_polar(
        self, roi_gray: np.ndarray
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
        center = (w / 2.0, h / 2.0)
        max_r = min(center[0], center[1])
        if max_r < 5:
            raise RuntimeError("ROI too small.")

        polar = cv2.warpPolar(
            blur,
            (int(max_r), 360),
            center,
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

        self.setWindowTitle("Donut Optimization Wizard")
        self.resize(600, 520)

        self._build_ui()
        self._camera.roi_changed.connect(self._on_roi_changed)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        roi_group = QtWidgets.QGroupBox("1) Select ROI")
        roi_layout = QtWidgets.QHBoxLayout(roi_group)
        self.lbl_roi = QtWidgets.QLabel("ROI: not set")
        self.btn_select_roi = QtWidgets.QPushButton("Select ROI")
        self.btn_select_roi.clicked.connect(self._select_roi)
        roi_layout.addWidget(self.lbl_roi)
        roi_layout.addStretch(1)
        roi_layout.addWidget(self.btn_select_roi)
        layout.addWidget(roi_group)

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
        self.lbl_roi.setText(f"ROI: {roi}")

    def _start(self) -> None:
        if not self._camera.is_running():
            self._append_error("Camera must be running.")
            return
        roi = self._camera.get_roi()
        if roi is None:
            self._append_error("Please select ROI first.")
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
        self, roi_gray: np.ndarray, polar: np.ndarray, peaks: np.ndarray, valid: np.ndarray
    ) -> None:
        if self._debug_window is None:
            self._debug_window = DebugWindow(self)
        if self.chk_debug.isChecked():
            self._debug_window.update_views(roi_gray, polar, peaks, valid)
