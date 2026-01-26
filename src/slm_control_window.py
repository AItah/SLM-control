from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from image_utils import exception_to_text, image_path_to_array, to_grayscale_qpixmap
from slm_cls import SLM, SLMError
from slm_params import SlmParams
from slm_store import SlmParamsStore
from slm_params_window import SlmParamsWindow
from vortex_window import VortexWindow


class ImageViewer(QtWidgets.QDialog):
    def __init__(self, title: str, arr: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(700, 550)
        layout = QtWidgets.QVBoxLayout(self)
        lbl = QtWidgets.QLabel()
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setPixmap(
            to_grayscale_qpixmap(arr).scaled(
                self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        )
        layout.addWidget(lbl)


class SLMWorker(QtCore.QObject):
    connected = QtCore.Signal(str)
    disconnected = QtCore.Signal()
    status = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    head_temp = QtCore.Signal(float)
    display_image = QtCore.Signal(object)
    fmem_image = QtCore.Signal(object, int)

    def __init__(self) -> None:
        super().__init__()
        self.slm: Optional[SLM] = None

    def _guard(self) -> None:
        if self.slm is None:
            raise SLMError("SLM is not connected.")

    @QtCore.Slot()
    def open(self) -> None:
        try:
            if self.slm is not None:
                self.status.emit("SLM already open.")
                return
            self.slm = SLM()
            self.slm.open()
            serial = self.slm.get_head_serial(0)
            self.connected.emit(
                f"Connected ({serial}); {self.slm._num_devices} device(s)"
            )
        except Exception as exc:
            self.slm = None
            self.failed.emit(exception_to_text(exc))

    @QtCore.Slot()
    def close(self) -> None:
        try:
            if self.slm is not None:
                self.slm.close()
                self.slm = None
            self.disconnected.emit()
        except Exception as exc:
            self.failed.emit(exception_to_text(exc))

    @QtCore.Slot(str, int, int, int)
    def send_bmp_to_slot(self, bmp_path: str, slot: int, xpix: int, ypix: int) -> None:
        try:
            self._guard()
            arr = image_path_to_array(bmp_path, size=(xpix, ypix)).flatten()
            self.slm.write_frame_array(arr, xpix, ypix, slot)
            self.status.emit(f"Wrote BMP to frame memory slot {slot}.")
        except Exception as exc:
            self.failed.emit(exception_to_text(exc))

    @QtCore.Slot(int)
    def change_display_slot(self, slot: int) -> None:
        try:
            self._guard()
            self.slm.change_display_slot(slot)
            self.status.emit(f"Display switched to slot {slot}.")
        except Exception as exc:
            self.failed.emit(exception_to_text(exc))

    @QtCore.Slot(object, int, int, int)
    def send_array_to_slot(self, arr: object, slot: int, xpix: int, ypix: int) -> None:
        try:
            self._guard()
            data = np.asarray(arr, dtype=np.uint8).reshape(ypix, xpix).flatten()
            self.slm.write_frame_array(data, xpix, ypix, slot)
            self.status.emit(f"Wrote array to frame memory slot {slot}.")
        except Exception as exc:
            self.failed.emit(exception_to_text(exc))

    @QtCore.Slot()
    def read_temperature(self) -> None:
        try:
            self._guard()
            head, cb = self.slm.get_temperature()
            self.head_temp.emit(head)
            self.status.emit(
                f"Temperature - Head: {head:.1f} C | Control Box: {cb:.1f} C"
            )
        except Exception as exc:
            self.failed.emit(exception_to_text(exc))

    @QtCore.Slot(int, int, int)
    def read_frame_memory(self, slot: int, xpix: int, ypix: int) -> None:
        try:
            self._guard()
            data = self.slm.read_frame_memory(slot, xpix, ypix)
            arr = np.array(data, dtype=np.uint8).reshape(ypix, xpix)
            self.fmem_image.emit(arr, slot)
            self.status.emit(f"Frame memory read from slot {slot}.")
        except Exception as exc:
            self.failed.emit(exception_to_text(exc))

    @QtCore.Slot(int, int)
    def read_display_image(self, xpix: int, ypix: int) -> None:
        try:
            self._guard()
            data = self.slm.read_display_image(xpix, ypix)
            arr = np.array(data, dtype=np.uint8).reshape(ypix, xpix)
            self.display_image.emit(arr)
            self.status.emit("Displayed image read.")
        except Exception as exc:
            self.failed.emit(exception_to_text(exc))


class SlmControlWindow(QtWidgets.QWidget):
    def __init__(
        self,
        store: SlmParamsStore,
        params_window: SlmParamsWindow,
        vortex_window: VortexWindow,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._params_window = params_window
        self._vortex_window = vortex_window
        self._slm_params: SlmParams | None = None
        self.temp_threshold_c = 35.0
        self.temp_timer: Optional[QtCore.QTimer] = None
        self._syncing_visibility = False

        self.setWindowTitle("SLM Control")
        self.resize(720, 520)

        self._build_ui()
        self._setup_worker_thread()

        self._store.changed.connect(self._on_params_changed)
        self._params_window.visibility_changed.connect(
            self._on_params_visibility_changed
        )
        self._vortex_window.visibility_changed.connect(
            self._on_vortex_visibility_changed
        )
        self._restore_settings()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.btn_connect = QtWidgets.QPushButton("Connect SLM")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)
        self.lbl_params = QtWidgets.QLabel("SLM params: not loaded")
        self.lbl_params.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        top.addWidget(self.btn_connect)
        top.addWidget(self.btn_disconnect)
        top.addStretch(1)
        top.addWidget(self.lbl_params)
        layout.addLayout(top)

        windows_box = QtWidgets.QGroupBox("Windows")
        windows_layout = QtWidgets.QHBoxLayout(windows_box)
        self.chk_window_params = QtWidgets.QCheckBox("SLM Parameters")
        self.chk_window_params.setChecked(True)
        self.chk_window_vortex = QtWidgets.QCheckBox("Vortex Generator")
        self.chk_window_vortex.setChecked(True)
        windows_layout.addWidget(self.chk_window_params)
        windows_layout.addWidget(self.chk_window_vortex)
        windows_layout.addStretch(1)
        layout.addWidget(windows_box)

        grid = QtWidgets.QGridLayout()
        r = 0

        self.ed_current_bmp = QtWidgets.QLineEdit()
        self.btn_pick_current_bmp = QtWidgets.QPushButton("Browse...")
        self.btn_pick_current_bmp.clicked.connect(self._pick_bmp)

        grid.addWidget(QtWidgets.QLabel("Current BMP:"), r, 0)
        grid.addWidget(self.ed_current_bmp, r, 1, 1, 3)
        grid.addWidget(self.btn_pick_current_bmp, r, 4)
        r += 1

        self.spin_slot = QtWidgets.QSpinBox()
        self.spin_slot.setRange(0, 818)
        self.spin_slot.setValue(0)

        grid.addWidget(QtWidgets.QLabel("Slot:"), r, 0)
        grid.addWidget(self.spin_slot, r, 1)
        r += 1

        self.btn_send_to_slot = QtWidgets.QPushButton("Send BMP to slot")
        self.btn_change_display = QtWidgets.QPushButton("Change display slot")
        grid.addWidget(self.btn_send_to_slot, r, 0, 1, 2)
        grid.addWidget(self.btn_change_display, r, 2, 1, 2)
        r += 1

        self.btn_temp = QtWidgets.QPushButton("Read temperature")
        self.btn_check_display = QtWidgets.QPushButton("Check DISPLAYED image")
        self.btn_check_fmem = QtWidgets.QPushButton("Check FMEM image")
        grid.addWidget(self.btn_temp, r, 0, 1, 2)
        grid.addWidget(self.btn_check_display, r, 2, 1, 2)
        grid.addWidget(self.btn_check_fmem, r, 4, 1, 1)
        r += 1

        self.chk_watchdog = QtWidgets.QCheckBox("Temperature watchdog")
        self.spin_interval = QtWidgets.QSpinBox()
        self.spin_interval.setRange(1, 3600)
        self.spin_interval.setValue(10)
        self.spin_interval.setSuffix(" s")
        grid.addWidget(self.chk_watchdog, r, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Interval:"), r, 2)
        grid.addWidget(self.spin_interval, r, 3)

        layout.addLayout(grid)

        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("Status and errors appear here...")
        layout.addWidget(self.log, 1)

        self.btn_connect.clicked.connect(self._on_connect_clicked)
        self.btn_disconnect.clicked.connect(self._on_disconnect_clicked)
        self.btn_send_to_slot.clicked.connect(self._on_send_to_slot)
        self.btn_change_display.clicked.connect(self._on_change_display)
        self.btn_temp.clicked.connect(self._on_temp_clicked)
        self.btn_check_display.clicked.connect(self._on_check_display)
        self.btn_check_fmem.clicked.connect(self._on_check_fmem)
        self.chk_watchdog.stateChanged.connect(self._on_watchdog_toggled)
        self.chk_window_params.toggled.connect(self._on_params_window_toggled)
        self.chk_window_vortex.toggled.connect(self._on_vortex_window_toggled)

    def _setup_worker_thread(self) -> None:
        self.slm_thread = QtCore.QThread(self)
        self.slm_worker = SLMWorker()
        self.slm_worker.moveToThread(self.slm_thread)
        self.slm_thread.start()

        self.slm_worker.connected.connect(self._on_slm_connected)
        self.slm_worker.disconnected.connect(self._on_slm_disconnected)
        self.slm_worker.status.connect(self._append_log)
        self.slm_worker.failed.connect(self._append_error)
        self.slm_worker.head_temp.connect(self._on_head_temp)
        self.slm_worker.display_image.connect(self._show_display_image)
        self.slm_worker.fmem_image.connect(self._show_fmem_image)

    def _on_params_changed(self, params: SlmParams) -> None:
        self._slm_params = params
        self.lbl_params.setText(
            f"SLM params: {params.nx} x {params.ny}, c2pi2unit {params.c2pi2unit}"
        )

    def _require_params(self) -> Optional[SlmParams]:
        if self._slm_params is None:
            self._params_window.ensure_loaded()
        if self._slm_params is None:
            self._append_error("Load SLM parameters first.")
            return None
        return self._slm_params

    def _restore_settings(self) -> None:
        settings = QtCore.QSettings()
        settings.beginGroup("slm_control_window")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        bmp_path = settings.value("bmp_path", "")
        if bmp_path:
            self.ed_current_bmp.setText(bmp_path)

        self.spin_slot.setValue(int(settings.value("slot", 0)))
        self.spin_interval.setValue(int(settings.value("watchdog_interval", 10)))
        self.chk_watchdog.setChecked(
            bool(settings.value("watchdog_enabled", False, bool))
        )

        params_visible = settings.value("params_window_visible", True, bool)
        vortex_visible = settings.value("vortex_window_visible", True, bool)
        settings.endGroup()

        self._set_params_window_visible(params_visible, update_checkbox=True)
        self._set_vortex_window_visible(vortex_visible, update_checkbox=True)

    def _save_settings(self) -> None:
        settings = QtCore.QSettings()
        settings.beginGroup("slm_control_window")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("bmp_path", self.ed_current_bmp.text().strip())
        settings.setValue("slot", int(self.spin_slot.value()))
        settings.setValue("watchdog_enabled", self.chk_watchdog.isChecked())
        settings.setValue("watchdog_interval", int(self.spin_interval.value()))
        settings.setValue("params_window_visible", self.chk_window_params.isChecked())
        settings.setValue("vortex_window_visible", self.chk_window_vortex.isChecked())
        settings.endGroup()

    def _set_params_window_visible(self, visible: bool, update_checkbox: bool) -> None:
        self._syncing_visibility = True
        try:
            if update_checkbox:
                self.chk_window_params.blockSignals(True)
                self.chk_window_params.setChecked(visible)
                self.chk_window_params.blockSignals(False)
            if visible:
                self._params_window.show()
                self._params_window.raise_()
                self._params_window.activateWindow()
            else:
                self._params_window.hide()
        finally:
            self._syncing_visibility = False

    def _on_params_window_toggled(self, checked: bool) -> None:
        self._set_params_window_visible(checked, update_checkbox=False)
        self._save_settings()

    def _set_vortex_window_visible(self, visible: bool, update_checkbox: bool) -> None:
        self._syncing_visibility = True
        try:
            if update_checkbox:
                self.chk_window_vortex.blockSignals(True)
                self.chk_window_vortex.setChecked(visible)
                self.chk_window_vortex.blockSignals(False)
            if visible:
                self._vortex_window.show()
                self._vortex_window.raise_()
                self._vortex_window.activateWindow()
            else:
                self._vortex_window.hide()
        finally:
            self._syncing_visibility = False

    def _on_vortex_window_toggled(self, checked: bool) -> None:
        self._set_vortex_window_visible(checked, update_checkbox=False)
        self._save_settings()

    def _pick_bmp(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select BMP", str(Path.cwd()), "BMP (*.bmp)"
        )
        if path:
            self.ed_current_bmp.setText(path)

    def _invoke_in_slm_thread(self, func, *args) -> None:
        qargs = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                qargs.append(QtCore.Q_ARG(object, arg))
            else:
                qargs.append(QtCore.Q_ARG(type(arg), arg))
        QtCore.QMetaObject.invokeMethod(
            self.slm_worker,
            func.__name__,
            QtCore.Qt.QueuedConnection,
            *qargs,
        )

    def _on_connect_clicked(self) -> None:
        if self._slm_params is None:
            self._params_window.ensure_loaded()
        self._invoke_in_slm_thread(self.slm_worker.open)

    def _on_disconnect_clicked(self) -> None:
        self._invoke_in_slm_thread(self.slm_worker.close)

    def _on_send_to_slot(self) -> None:
        bmp = self.ed_current_bmp.text().strip()
        if not bmp or not Path(bmp).exists():
            self._append_error("No BMP selected or path does not exist.")
            return
        params = self._require_params()
        if params is None:
            return
        slot = int(self.spin_slot.value())
        self._invoke_in_slm_thread(
            self.slm_worker.send_bmp_to_slot, bmp, slot, params.nx, params.ny
        )

    def _on_change_display(self) -> None:
        slot = int(self.spin_slot.value())
        self._invoke_in_slm_thread(self.slm_worker.change_display_slot, slot)

    def _on_temp_clicked(self) -> None:
        self._invoke_in_slm_thread(self.slm_worker.read_temperature)

    def _on_check_display(self) -> None:
        params = self._require_params()
        if params is None:
            return
        self._invoke_in_slm_thread(
            self.slm_worker.read_display_image, params.nx, params.ny
        )

    def _on_check_fmem(self) -> None:
        params = self._require_params()
        if params is None:
            return
        slot = int(self.spin_slot.value())
        self._invoke_in_slm_thread(
            self.slm_worker.read_frame_memory, slot, params.nx, params.ny
        )

    def _on_slm_connected(self, msg: str) -> None:
        self._append_log(msg)
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
        if self._slm_params is None:
            self._params_window.ensure_loaded()

    def _on_slm_disconnected(self) -> None:
        self._append_log("SLM disconnected.")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self._stop_watchdog()

    def _on_params_visibility_changed(self, visible: bool) -> None:
        if self._syncing_visibility:
            return
        self._syncing_visibility = True
        try:
            self.chk_window_params.blockSignals(True)
            self.chk_window_params.setChecked(visible)
            self.chk_window_params.blockSignals(False)
        finally:
            self._syncing_visibility = False
        self._save_settings()

    def _on_vortex_visibility_changed(self, visible: bool) -> None:
        if self._syncing_visibility:
            return
        self._syncing_visibility = True
        try:
            self.chk_window_vortex.blockSignals(True)
            self.chk_window_vortex.setChecked(visible)
            self.chk_window_vortex.blockSignals(False)
        finally:
            self._syncing_visibility = False
        self._save_settings()

    def _on_watchdog_toggled(self, state: int) -> None:
        if self.chk_watchdog.isChecked():
            self._start_watchdog()
        else:
            self._stop_watchdog()

    def _start_watchdog(self) -> None:
        if self.temp_timer:
            self.temp_timer.stop()
        self.temp_timer = QtCore.QTimer(self)
        self.temp_timer.timeout.connect(
            lambda: self._invoke_in_slm_thread(self.slm_worker.read_temperature)
        )
        self.temp_timer.start(self.spin_interval.value() * 1000)
        self._append_log(
            "Temperature watchdog enabled: threshold "
            f"{self.temp_threshold_c:.1f} C every {self.spin_interval.value()} s."
        )

    def _stop_watchdog(self) -> None:
        if self.temp_timer:
            self.temp_timer.stop()
            self.temp_timer = None
            self._append_log("Temperature watchdog disabled.")

    def _on_head_temp(self, head_c: float) -> None:
        if self.chk_watchdog.isChecked() and head_c > self.temp_threshold_c:
            QtWidgets.QMessageBox.warning(
                self,
                "Temperature Watchdog",
                f"Head temperature {head_c:.1f} C exceeds {self.temp_threshold_c:.1f} C.",
            )

    def _show_display_image(self, arr: np.ndarray) -> None:
        ImageViewer("Displayed image (current SLM output)", arr, self).exec()

    def _show_fmem_image(self, arr: np.ndarray, slot: int) -> None:
        ImageViewer(f"Frame memory (slot {slot})", arr, self).exec()

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _append_error(self, text: str) -> None:
        self.log.appendPlainText("ERROR: " + text)

    def shutdown(self) -> None:
        self._save_settings()
        self._stop_watchdog()
        try:
            self._invoke_in_slm_thread(self.slm_worker.close)
        except Exception:
            pass
        try:
            self.slm_thread.quit()
            self.slm_thread.wait(1000)
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_settings()
        self.shutdown()
        try:
            self._params_window.force_close()
        except Exception:
            pass
        try:
            self._vortex_window.force_close()
        except Exception:
            pass
        QtCore.QCoreApplication.quit()
        event.accept()

    def send_mask_to_slot(self, mask_u8: np.ndarray, slot: int) -> bool:
        params = self._require_params()
        if params is None:
            return False
        if mask_u8.shape != (params.ny, params.nx):
            self._append_error(
                f"Mask size {mask_u8.shape} does not match SLM {params.nx}x{params.ny}."
            )
            return False
        self._invoke_in_slm_thread(
            self.slm_worker.send_array_to_slot, mask_u8, slot, params.nx, params.ny
        )
        self._invoke_in_slm_thread(self.slm_worker.change_display_slot, slot)
        return True
