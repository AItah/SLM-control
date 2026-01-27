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
    point_selected = QtCore.Signal(object)
    circle_selected = QtCore.Signal(object)

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
        self._point_selecting = False
        self._circle_selecting = False
        self._circle_dragging = False
        self._circle_edit_mode: Optional[str] = None
        self._circle_drag_offset: Optional[Tuple[float, float]] = None
        self._selected_point: Optional[Tuple[float, float]] = None
        self._selected_circle_center: Optional[Tuple[float, float]] = None
        self._selected_circle_radius: Optional[float] = None

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
        self._render_frame(frame, fmt)

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
        self._clear_selection_modes()
        self._roi_selecting = True
        self._append_log("Drag to select ROI on the image.")

    def clear_roi(self) -> None:
        """Clear ROI selection (revert to full frame)."""
        with self._roi_lock:
            self._roi = None
        if self._roi_band is not None:
            self._roi_band.hide()
        self.roi_changed.emit(None)
        self._append_log("ROI cleared.")
        self._refresh_last_frame()

    def begin_point_selection(self) -> None:
        if self._thread is None:
            self._append_error("Camera is not running.")
            return
        self._clear_selection_modes()
        self._point_selecting = True
        self._append_log("Click to select the dark spot center.")

    def set_selected_point(
        self, point: Optional[Tuple[float, float]], emit: bool = False
    ) -> None:
        self._selected_point = (
            (float(point[0]), float(point[1])) if point is not None else None
        )
        if emit and self._selected_point is not None:
            self.point_selected.emit(self._selected_point)
        self._refresh_last_frame()

    def begin_circle_selection(self, center: Optional[Tuple[float, float]] = None) -> None:
        if self._thread is None:
            self._append_error("Camera is not running.")
            return
        self._clear_selection_modes()
        self._circle_selecting = True
        self._circle_dragging = False
        if center is not None:
            self._selected_circle_center = (float(center[0]), float(center[1]))
        self._selected_circle_radius = None
        self._append_log("Drag to set the donut circle radius.")
        self._refresh_last_frame()

    def clear_manual_marks(self) -> None:
        self._selected_point = None
        self._selected_circle_center = None
        self._selected_circle_radius = None
        self._refresh_last_frame()

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        with self._roi_lock:
            return self._roi

    def eventFilter(self, obj, event) -> bool:
        if obj is self.lbl_view:
            if (
                not self._roi_selecting
                and not self._point_selecting
                and not self._circle_selecting
                and self._selected_circle_center is not None
                and self._selected_circle_radius is not None
            ):
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    pos = self._label_pos_to_image(event.position())
                    if pos is None:
                        return False
                    cx, cy = self._selected_circle_center
                    dx = float(pos[0]) - float(cx)
                    dy = float(pos[1]) - float(cy)
                    dist = float((dx * dx + dy * dy) ** 0.5)
                    r = float(self._selected_circle_radius)
                    move_thresh = max(8.0, 0.03 * r)
                    resize_thresh = max(6.0, 0.02 * r)
                    if dist <= move_thresh:
                        self._circle_dragging = True
                        self._circle_edit_mode = "move"
                        self._circle_drag_offset = (dx, dy)
                        return True
                    if abs(dist - r) <= resize_thresh:
                        self._circle_dragging = True
                        self._circle_edit_mode = "resize"
                        self._circle_drag_offset = None
                        return True
                if (
                    event.type() == QtCore.QEvent.MouseMove
                    and self._circle_dragging
                    and self._circle_edit_mode is not None
                ):
                    pos = self._label_pos_to_image(event.position())
                    if pos is None:
                        return True
                    if self._circle_edit_mode == "move":
                        if self._circle_drag_offset is None:
                            self._circle_drag_offset = (0.0, 0.0)
                        ox, oy = self._circle_drag_offset
                        self._selected_circle_center = (
                            float(pos[0]) - float(ox),
                            float(pos[1]) - float(oy),
                        )
                    elif self._circle_edit_mode == "resize":
                        cx, cy = self._selected_circle_center
                        dx = float(pos[0]) - float(cx)
                        dy = float(pos[1]) - float(cy)
                        self._selected_circle_radius = float((dx * dx + dy * dy) ** 0.5)
                    self._refresh_last_frame()
                    return True
                if (
                    event.type() == QtCore.QEvent.MouseButtonRelease
                    and self._circle_dragging
                    and self._circle_edit_mode is not None
                ):
                    self._circle_dragging = False
                    self._circle_edit_mode = None
                    self._circle_drag_offset = None
                    if (
                        self._selected_circle_center is not None
                        and self._selected_circle_radius is not None
                        and self._selected_circle_radius >= 2.0
                    ):
                        payload = (
                            float(self._selected_circle_center[0]),
                            float(self._selected_circle_center[1]),
                            float(self._selected_circle_radius),
                        )
                        self.circle_selected.emit(payload)
                    self._refresh_last_frame()
                    return True
            if self._roi_selecting:
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    self._roi_start = event.position().toPoint()
                    if self._roi_band is not None:
                        self._roi_band.setGeometry(
                            QtCore.QRect(self._roi_start, QtCore.QSize())
                        )
                        self._roi_band.show()
                    return True
                if event.type() == QtCore.QEvent.MouseMove:
                    if not self._roi_start.isNull():
                        rect = QtCore.QRect(
                            self._roi_start, event.position().toPoint()
                        ).normalized()
                        if self._roi_band is not None:
                            self._roi_band.setGeometry(rect)
                    return True
                if event.type() == QtCore.QEvent.MouseButtonRelease:
                    rect = (
                        self._roi_band.geometry()
                        if self._roi_band is not None
                        else QtCore.QRect()
                    )
                    self._roi_selecting = False
                    self._set_roi_from_label_rect(rect)
                    if self._roi_band is not None:
                        self._roi_band.hide()
                    return True
            if self._point_selecting:
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    pos = self._label_pos_to_image(event.position())
                    if pos is not None:
                        self._selected_point = (float(pos[0]), float(pos[1]))
                        self._point_selecting = False
                        self.point_selected.emit(self._selected_point)
                        self._refresh_last_frame()
                    return True
            if self._circle_selecting:
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    pos = self._label_pos_to_image(event.position())
                    if pos is None:
                        return True
                    if self._selected_circle_center is None:
                        self._selected_circle_center = (float(pos[0]), float(pos[1]))
                    self._circle_dragging = True
                    self._update_circle_radius_from_pos(pos)
                    self._refresh_last_frame()
                    return True
                if event.type() == QtCore.QEvent.MouseMove:
                    if self._circle_dragging:
                        pos = self._label_pos_to_image(event.position())
                        if pos is not None:
                            self._update_circle_radius_from_pos(pos)
                            self._refresh_last_frame()
                    return True
                if event.type() == QtCore.QEvent.MouseButtonRelease:
                    if self._circle_dragging:
                        pos = self._label_pos_to_image(event.position())
                        if pos is not None:
                            self._update_circle_radius_from_pos(pos)
                        self._circle_dragging = False
                        self._circle_selecting = False
                        if (
                            self._selected_circle_center is not None
                            and self._selected_circle_radius is not None
                            and self._selected_circle_radius >= 2.0
                        ):
                            payload = (
                                float(self._selected_circle_center[0]),
                                float(self._selected_circle_center[1]),
                                float(self._selected_circle_radius),
                            )
                            self.circle_selected.emit(payload)
                        self._refresh_last_frame()
                    return True
        return super().eventFilter(obj, event)

    def _set_roi_from_label_rect(self, rect: QtCore.QRect) -> None:
        frame_size = self._get_frame_size()
        if frame_size is None:
            return
        img_w, img_h = frame_size
        label_rect = self.lbl_view.contentsRect()
        scale = min(
            label_rect.width() / img_w,
            label_rect.height() / img_h,
        )
        disp_w = img_w * scale
        disp_h = img_h * scale
        x0 = label_rect.x() + (label_rect.width() - disp_w) / 2
        y0 = label_rect.y() + (label_rect.height() - disp_h) / 2

        left = rect.x()
        top = rect.y()
        right = rect.x() + rect.width()
        bottom = rect.y() + rect.height()
        cx0 = max(left, x0)
        cy0 = max(top, y0)
        cx1 = min(right, x0 + disp_w)
        cy1 = min(bottom, y0 + disp_h)
        if cx1 <= cx0 or cy1 <= cy0:
            return

        rx = cx0 - x0
        ry = cy0 - y0
        rw = cx1 - cx0
        rh = cy1 - cy0

        img_x = int(round(rx / scale))
        img_y = int(round(ry / scale))
        img_w_roi = int(round(rw / scale))
        img_h_roi = int(round(rh / scale))

        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))
        img_w_roi = max(1, min(img_w_roi, img_w - img_x))
        img_h_roi = max(1, min(img_h_roi, img_h - img_y))

        with self._roi_lock:
            self._roi = (img_x, img_y, img_w_roi, img_h_roi)
        self.roi_changed.emit(self._roi)
        self._append_log(f"ROI set: {self._roi}")

    def _update_roi_overlay(self) -> None:
        roi = self.get_roi()
        if roi is None:
            return
        frame_size = self._get_frame_size()
        if frame_size is None:
            return
        if self._roi_band is None:
            return
        img_w, img_h = frame_size
        label_rect = self.lbl_view.contentsRect()
        scale = min(
            label_rect.width() / img_w,
            label_rect.height() / img_h,
        )
        disp_w = img_w * scale
        disp_h = img_h * scale
        x0 = label_rect.x() + (label_rect.width() - disp_w) / 2
        y0 = label_rect.y() + (label_rect.height() - disp_h) / 2

        img_x, img_y, img_w, img_h = roi
        rx = int(x0 + img_x * scale)
        ry = int(y0 + img_y * scale)
        rw = int(img_w * scale)
        rh = int(img_h * scale)
        self._roi_band.setGeometry(QtCore.QRect(rx, ry, rw, rh))
        self._roi_band.show()

    def _get_frame_size(self) -> Optional[Tuple[int, int]]:
        with self._frame_lock:
            if self._last_frame is None:
                return None
            h, w = self._last_frame.shape[:2]
            return int(w), int(h)

    def _render_frame(self, frame: np.ndarray, fmt: str) -> None:
        if frame.ndim == 2:
            qimg = _gray_to_qimage(frame)
        else:
            if fmt == "rgb8":
                qimg = _rgb_to_qimage(frame)
            else:
                qimg = _bgr_to_qimage(frame)
        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.lbl_view.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        pix = self._draw_overlay_on_pixmap(pix, frame.shape[1], frame.shape[0])
        self.lbl_view.setPixmap(pix)
        self._update_roi_overlay()

    def _refresh_last_frame(self) -> None:
        with self._frame_lock:
            if self._last_frame is None:
                return
            frame = self._last_frame.copy()
            fmt = self._last_fmt
        self._render_frame(frame, fmt)

    def _draw_overlay_on_pixmap(
        self, pix: QtGui.QPixmap, img_w: int, img_h: int
    ) -> QtGui.QPixmap:
        if (
            self._selected_point is None
            and self._selected_circle_center is None
            and self._selected_circle_radius is None
        ):
            return pix
        if img_w <= 0 or img_h <= 0:
            return pix
        scale = pix.width() / float(img_w)
        if scale <= 0:
            return pix
        out = pix.copy()
        painter = QtGui.QPainter(out)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        if self._selected_circle_center is not None:
            cx, cy = self._selected_circle_center
            pen = QtGui.QPen(QtGui.QColor(0, 200, 0), 2)
            painter.setPen(pen)
            if self._selected_circle_radius is not None:
                r = self._selected_circle_radius * scale
                painter.drawEllipse(
                    QtCore.QPointF(cx * scale, cy * scale), r, r
                )
            painter.drawLine(
                int(cx * scale) - 6,
                int(cy * scale),
                int(cx * scale) + 6,
                int(cy * scale),
            )
            painter.drawLine(
                int(cx * scale),
                int(cy * scale) - 6,
                int(cx * scale),
                int(cy * scale) + 6,
            )

        if self._selected_point is not None:
            x, y = self._selected_point
            pen = QtGui.QPen(QtGui.QColor(255, 60, 60), 2)
            painter.setPen(pen)
            painter.drawLine(
                int(x * scale) - 6,
                int(y * scale),
                int(x * scale) + 6,
                int(y * scale),
            )
            painter.drawLine(
                int(x * scale),
                int(y * scale) - 6,
                int(x * scale),
                int(y * scale) + 6,
            )

        painter.end()
        return out

    def _label_pos_to_image(self, pos: QtCore.QPointF) -> Optional[Tuple[float, float]]:
        frame_size = self._get_frame_size()
        if frame_size is None:
            return None
        img_w, img_h = frame_size
        label_rect = self.lbl_view.contentsRect()
        if img_w <= 0 or img_h <= 0:
            return None
        scale = min(label_rect.width() / img_w, label_rect.height() / img_h)
        if scale <= 0:
            return None
        disp_w = img_w * scale
        disp_h = img_h * scale
        x0 = label_rect.x() + (label_rect.width() - disp_w) / 2
        y0 = label_rect.y() + (label_rect.height() - disp_h) / 2
        x = float(pos.x()) - x0
        y = float(pos.y()) - y0
        if x < 0 or y < 0 or x > disp_w or y > disp_h:
            return None
        return (x / scale, y / scale)

    def _update_circle_radius_from_pos(self, pos: Tuple[float, float]) -> None:
        if self._selected_circle_center is None:
            return
        cx, cy = self._selected_circle_center
        dx = float(pos[0]) - float(cx)
        dy = float(pos[1]) - float(cy)
        self._selected_circle_radius = float((dx * dx + dy * dy) ** 0.5)

    def _clear_selection_modes(self) -> None:
        self._roi_selecting = False
        self._point_selecting = False
        self._circle_selecting = False
        self._circle_dragging = False
        self._circle_edit_mode = None
        self._circle_drag_offset = None
        if self._roi_band is not None:
            self._roi_band.hide()
