from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

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


class OffsetScanWorker(QtCore.QObject):
    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(float, float)
    failed = QtCore.Signal(str)

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

        if roi.ndim == 3:
            roi_gray = (
                0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]
            )
        else:
            roi_gray = roi.astype(np.float32)

        min_idx = np.argmin(roi_gray)
        min_y, min_x = np.unravel_index(min_idx, roi_gray.shape)
        cx = roi_gray.shape[1] / 2.0
        cy = roi_gray.shape[0] / 2.0
        dist = math.hypot(min_x - cx, min_y - cy)
        return float(dist)


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
        )

        self._thread = QtCore.QThread(self)
        self._worker = OffsetScanWorker(self._vortex, self._slm, self._camera, roi, settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
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
