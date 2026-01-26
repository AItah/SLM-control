from __future__ import annotations

from typing import Optional, Tuple

import threading
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from camera_subscriber import CameraSettings, CameraSubscriber, decode_frame
from image_utils import exception_to_text


def _rgb_to_qimage(img: np.ndarray) -> QtGui.QImage:
    h, w, _ = img.shape
    return QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format_RGB888).copy()


def _bgr_to_qimage(img: np.ndarray) -> QtGui.QImage:
    rgb = img[:, :, ::-1]
    return _rgb_to_qimage(rgb)


def _gray_to_qimage(img: np.ndarray) -> QtGui.QImage:
    h, w = img.shape
    return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()


class CameraWorker(QtCore.QObject):
    frame_ready = QtCore.Signal(object, str)
    status_message = QtCore.Signal(str)
    error_message = QtCore.Signal(str)

    def __init__(self, settings: CameraSettings) -> None:
        super().__init__()
        self._settings = settings
        self._subscriber: Optional[CameraSubscriber] = None
        self._running = False

    @QtCore.Slot()
    def run(self) -> None:
        self._running = True
        self._subscriber = CameraSubscriber(self._settings)
        if not self._subscriber.available:
            self.error_message.emit(
                f"pyzmq is required: {self._subscriber.import_error}"
            )
            return
        try:
            self._subscriber.start()
        except Exception as exc:
            self.error_message.emit(exception_to_text(exc))
            return

        while self._running:
            try:
                item = self._subscriber.poll()
            except Exception as exc:
                self.error_message.emit(exception_to_text(exc))
                continue
            if item is None:
                continue
            topic, metadata, payload = item
            if topic == self._settings.status_topic:
                if self._settings.show_status:
                    self.status_message.emit(str(metadata))
                continue
            if topic != self._settings.topic:
                continue
            try:
                frame, fmt = decode_frame(metadata, payload)
            except Exception as exc:
                self.error_message.emit(f"Frame decode error: {exc}")
                continue
            self.frame_ready.emit(frame, fmt)

        if self._subscriber:
            self._subscriber.close()

    @QtCore.Slot()
    def stop(self) -> None:
        self._running = False


class CameraWindow(QtWidgets.QWidget):
    visibility_changed = QtCore.Signal(bool)
    roi_changed = QtCore.Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[CameraWorker] = None
        self._force_close = False
        self._restart_timer = QtCore.QTimer(self)
        self._restart_timer.setSingleShot(True)
        self._restart_timer.setInterval(300)
        self._restart_timer.timeout.connect(self._restart)
        self._frame_lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_fmt: str = ""
        self._last_frame_time: float = 0.0
        self._roi_lock = threading.Lock()
        self._roi: Optional[Tuple[int, int, int, int]] = None
        self._roi_selecting = False
        self._roi_start = QtCore.QPoint()
        self._roi_band: Optional[QtWidgets.QRubberBand] = None

        self.setWindowTitle("Camera Viewer")
        self.resize(900, 700)

        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        conn = QtWidgets.QGridLayout()
        r = 0
        self.ed_endpoint = QtWidgets.QLineEdit("tcp://127.0.0.1:5555")
        self.ed_topic = QtWidgets.QLineEdit("camera")
        self.ed_status_topic = QtWidgets.QLineEdit("camera/status")
        self.chk_show_status = QtWidgets.QCheckBox("Show status")
        self.chk_show_status.setChecked(True)
        self.chk_bind = QtWidgets.QCheckBox("Bind")
        self.spin_hwm = QtWidgets.QSpinBox()
        self.spin_hwm.setRange(1, 1000)
        self.spin_hwm.setValue(2)
        self.spin_poll = QtWidgets.QSpinBox()
        self.spin_poll.setRange(1, 1000)
        self.spin_poll.setValue(50)
        self.spin_poll.setSuffix(" ms")

        conn.addWidget(QtWidgets.QLabel("Endpoint:"), r, 0)
        conn.addWidget(self.ed_endpoint, r, 1, 1, 3)
        r += 1
        conn.addWidget(QtWidgets.QLabel("Topic:"), r, 0)
        conn.addWidget(self.ed_topic, r, 1)
        conn.addWidget(QtWidgets.QLabel("Status topic:"), r, 2)
        conn.addWidget(self.ed_status_topic, r, 3)
        r += 1
        conn.addWidget(self.chk_show_status, r, 0)
        conn.addWidget(self.chk_bind, r, 1)
        conn.addWidget(QtWidgets.QLabel("RCV HWM:"), r, 2)
        conn.addWidget(self.spin_hwm, r, 3)
        r += 1
        conn.addWidget(QtWidgets.QLabel("Poll:"), r, 2)
        conn.addWidget(self.spin_poll, r, 3)
        layout.addLayout(conn)

        btns = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addStretch(1)
        layout.addLayout(btns)

        self.lbl_view = QtWidgets.QLabel()
        self.lbl_view.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_view.setMinimumSize(640, 480)
        self.lbl_view.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lbl_view.installEventFilter(self)
        layout.addWidget(self.lbl_view, 1)
        self._roi_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.lbl_view)

        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(1000)
        self.log.setPlaceholderText("Camera status and errors appear here...")
        layout.addWidget(self.log, 0)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)

        self.ed_endpoint.editingFinished.connect(self._schedule_restart)
        self.ed_topic.editingFinished.connect(self._schedule_restart)
        self.ed_status_topic.editingFinished.connect(self._schedule_restart)
        self.chk_show_status.stateChanged.connect(self._schedule_restart)
        self.chk_bind.stateChanged.connect(self._schedule_restart)
        self.spin_hwm.valueChanged.connect(self._schedule_restart)
        self.spin_poll.valueChanged.connect(self._schedule_restart)

    def _start(self) -> None:
        if self._thread is not None:
            return
        settings = CameraSettings(
            endpoint=self.ed_endpoint.text().strip(),
            topic=self.ed_topic.text().strip(),
            status_topic=self.ed_status_topic.text().strip(),
            show_status=self.chk_show_status.isChecked(),
            bind=self.chk_bind.isChecked(),
            rcv_hwm=int(self.spin_hwm.value()),
            poll_ms=int(self.spin_poll.value()),
        )
        self._thread = QtCore.QThread(self)
        self._worker = CameraWorker(settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.status_message.connect(self._append_log)
        self._worker.error_message.connect(self._append_error)
        self._thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._save_settings()

    def _restart(self) -> None:
        if self._thread is None:
            return
        self._append_log("Restarting subscriber with new settings...")
        self._stop()
        self._start()

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
        self._save_settings()

    def _on_frame(self, frame: np.ndarray, fmt: str) -> None:
        with self._frame_lock:
            self._last_frame = frame.copy()
            self._last_fmt = fmt
            self._last_frame_time = time.monotonic()
        if frame.ndim == 2:
            qimg = _gray_to_qimage(frame)
        else:
            if fmt == "rgb8":
                qimg = _rgb_to_qimage(frame)
            else:
                qimg = _bgr_to_qimage(frame)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.lbl_view.setPixmap(
            pix.scaled(
                self.lbl_view.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )
        self._update_roi_overlay()

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _append_error(self, text: str) -> None:
        self.log.appendPlainText("ERROR: " + text)

    def _schedule_restart(self) -> None:
        self._save_settings()
        if self._thread is None:
            return
        self._restart_timer.start()

    def _restore_settings(self) -> None:
        settings = QtCore.QSettings()
        settings.beginGroup("camera_window")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.ed_endpoint.setText(settings.value("endpoint", "tcp://127.0.0.1:5555"))
        self.ed_topic.setText(settings.value("topic", "camera"))
        self.ed_status_topic.setText(settings.value("status_topic", "camera/status"))
        self.chk_show_status.setChecked(bool(settings.value("show_status", True, bool)))
        self.chk_bind.setChecked(bool(settings.value("bind", False, bool)))
        self.spin_hwm.setValue(int(settings.value("rcv_hwm", 2)))
        self.spin_poll.setValue(int(settings.value("poll_ms", 50)))
        settings.endGroup()

    def _save_settings(self) -> None:
        settings = QtCore.QSettings()
        settings.beginGroup("camera_window")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("endpoint", self.ed_endpoint.text().strip())
        settings.setValue("topic", self.ed_topic.text().strip())
        settings.setValue("status_topic", self.ed_status_topic.text().strip())
        settings.setValue("show_status", self.chk_show_status.isChecked())
        settings.setValue("bind", self.chk_bind.isChecked())
        settings.setValue("rcv_hwm", int(self.spin_hwm.value()))
        settings.setValue("poll_ms", int(self.spin_poll.value()))
        settings.setValue("visible", self.isVisible())
        settings.endGroup()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.visibility_changed.emit(True)
        self._save_settings()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        self.visibility_changed.emit(False)
        self._save_settings()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_settings()
        if self._force_close or QtCore.QCoreApplication.closingDown():
            self._stop()
            event.accept()
            return
        event.ignore()
        self.hide()

    def force_close(self) -> None:
        self._force_close = True
        self.close()

    def is_running(self) -> bool:
        return self._thread is not None

    def get_last_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._last_frame is None:
                return None
            return self._last_frame.copy()

    def get_last_frame_time(self) -> float:
        with self._frame_lock:
            return float(self._last_frame_time)

    def begin_roi_selection(self) -> None:
        if self._thread is None:
            self._append_error("Camera is not running.")
            return
        self._roi_selecting = True
        self._append_log("Drag to select ROI on the image.")

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        with self._roi_lock:
            return self._roi

    def eventFilter(self, obj, event) -> bool:
        if obj is self.lbl_view and self._roi_selecting:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self._roi_start = event.position().toPoint()
                if self._roi_band is not None:
                    self._roi_band.setGeometry(QtCore.QRect(self._roi_start, QtCore.QSize()))
                    self._roi_band.show()
                return True
            if event.type() == QtCore.QEvent.MouseMove:
                if not self._roi_start.isNull():
                    rect = QtCore.QRect(self._roi_start, event.position().toPoint()).normalized()
                    if self._roi_band is not None:
                        self._roi_band.setGeometry(rect)
                return True
            if event.type() == QtCore.QEvent.MouseButtonRelease:
                rect = self._roi_band.geometry() if self._roi_band is not None else QtCore.QRect()
                self._roi_selecting = False
                self._set_roi_from_label_rect(rect)
                if self._roi_band is not None:
                    self._roi_band.hide()
                return True
        return super().eventFilter(obj, event)

    def _set_roi_from_label_rect(self, rect: QtCore.QRect) -> None:
        pix = self.lbl_view.pixmap()
        if pix is None or pix.isNull():
            return
        label_rect = self.lbl_view.contentsRect()
        pix_size = pix.size()
        scale = min(
            label_rect.width() / pix_size.width(),
            label_rect.height() / pix_size.height(),
        )
        disp_w = pix_size.width() * scale
        disp_h = pix_size.height() * scale
        x0 = label_rect.x() + (label_rect.width() - disp_w) / 2
        y0 = label_rect.y() + (label_rect.height() - disp_h) / 2

        rx = max(rect.x() - x0, 0)
        ry = max(rect.y() - y0, 0)
        rw = min(rect.width(), disp_w - rx)
        rh = min(rect.height(), disp_h - ry)
        if rw <= 0 or rh <= 0:
            return

        img_x = int(rx / scale)
        img_y = int(ry / scale)
        img_w = int(rw / scale)
        img_h = int(rh / scale)

        with self._roi_lock:
            self._roi = (img_x, img_y, img_w, img_h)
        self.roi_changed.emit(self._roi)
        self._append_log(f"ROI set: {self._roi}")

    def _update_roi_overlay(self) -> None:
        roi = self.get_roi()
        if roi is None:
            return
        pix = self.lbl_view.pixmap()
        if pix is None or pix.isNull():
            return
        if self._roi_band is None:
            return
        label_rect = self.lbl_view.contentsRect()
        pix_size = pix.size()
        scale = min(
            label_rect.width() / pix_size.width(),
            label_rect.height() / pix_size.height(),
        )
        disp_w = pix_size.width() * scale
        disp_h = pix_size.height() * scale
        x0 = label_rect.x() + (label_rect.width() - disp_w) / 2
        y0 = label_rect.y() + (label_rect.height() - disp_h) / 2

        img_x, img_y, img_w, img_h = roi
        rx = int(x0 + img_x * scale)
        ry = int(y0 + img_y * scale)
        rw = int(img_w * scale)
        rh = int(img_h * scale)
        self._roi_band.setGeometry(QtCore.QRect(rx, ry, rw, rh))
        self._roi_band.show()
