from __future__ import annotations

import sys
import json
import threading
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from slm_control_window import SlmControlWindow
from slm_params_window import SlmParamsWindow
from slm_store import SlmParamsStore
from vortex_window import VortexWindow

# ---- Qt / Matplotlib glue ----
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
QT_LIB = 'PySide6'

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SLM_PARAMS_JSON = Path("vendor") / "LCOS_SLM_X15213.json"
DEFAULT_BEAM_PARAMS_JSON = Path("GaussianBeam.json")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------- App Defaults ----------------------
# Use the provided defaults filename
DEFAULTS_JSON = HERE / 'app_defaults.json'

# ---------------------- Helpers ----------------------


def to_grayscale_qpixmap(arr_2d: np.ndarray) -> QtGui.QPixmap:
    arr = np.ascontiguousarray(arr_2d.astype(np.uint8))
    h, w = arr.shape
    qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qimg.copy())


def image_path_to_array(path: str, size) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(size, resample=Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def exception_to_text(e: BaseException) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))

# ---------------------- GS Progress Window ----------------------


class GSProgressWindow(QtWidgets.QDialog):
    """A simple live-plot window updated from the main thread via signals."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GS progress plots")
        self.resize(1000, 720)

        v = QtWidgets.QVBoxLayout(self)
        self.fig = Figure(figsize=(10, 7), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        v.addWidget(self.canvas)

        self.ax_recon = self.fig.add_subplot(2, 2, 1)
        self.ax_target = self.fig.add_subplot(2, 2, 2)
        self.ax_phase = self.fig.add_subplot(2, 2, 3)
        self.ax_err = self.fig.add_subplot(2, 2, 4)

        self.im_recon = self.ax_recon.imshow(np.zeros((10, 10)), cmap='gray')
        self.ax_recon.set_title("Reconstructed Amplitude")
        self.ax_recon.axis('off')

        self.im_target = self.ax_target.imshow(np.zeros((10, 10)), cmap='gray')
        self.ax_target.set_title("Target Amplitude")
        self.ax_target.axis('off')

        self.im_phase = self.ax_phase.imshow(
            np.zeros((10, 10)), cmap='hsv', vmin=-np.pi, vmax=np.pi)
        self.ax_phase.set_title("Current Phase Mask")
        self.ax_phase.axis('off')

        (self.err_line,) = self.ax_err.plot([], [], linewidth=2)
        self.ax_err.set_title("Error Progression")
        self.ax_err.set_xlabel("Iteration")
        self.ax_err.set_ylabel("MSE")
        self.ax_err.grid(True)
        self.err_x = []
        self.err_y = []

    def update_from_data(self, data: dict):
        # data: {'it': int, 'total': int, 'recon': 2D, 'phase': 2D, 'target': 2D, 'error': float}
        self.im_recon.set_data(data['recon'])
        self.im_target.set_data(data['target'])
        self.im_phase.set_data(data['phase'])

        # autoscale images
        self.im_recon.set_clim(vmin=np.min(
            data['recon']), vmax=np.max(data['recon']))
        self.im_target.set_clim(vmin=np.min(
            data['target']), vmax=np.max(data['target']))

        # update error trace
        self.err_x.append(data['it'])
        self.err_y.append(data['error'])
        self.err_line.set_data(self.err_x, self.err_y)
        if self.err_x:
            self.ax_err.set_xlim(0, max(self.err_x))
        if self.err_y:
            lo, hi = min(self.err_y), max(self.err_y)
            pad = 1e-9 + 0.1*(hi - lo)
            self.ax_err.set_ylim(lo - pad, hi + pad)

        self.ax_recon.set_title(
            f"Reconstructed Amplitude (it {data['it']} / {data['total']})")
        self.canvas.draw_idle()

# ---------------------- Worker Threads ----------------------


class CGHWorker(QtCore.QObject):
    finished = QtCore.Signal(
        str, np.ndarray) if QT_LIB == 'PySide6' else QtCore.pyqtSignal(str, np.ndarray)
    failed = QtCore.Signal(
        str) if QT_LIB == 'PySide6' else QtCore.pyqtSignal(str)
    progress = QtCore.Signal(
        int) if QT_LIB == 'PySide6' else QtCore.pyqtSignal(int)
    progress_data = QtCore.Signal(
        object) if QT_LIB == 'PySide6' else QtCore.pyqtSignal(object)

    def __init__(self, params: dict, cancel_event: threading.Event, show_plots: bool):
        super().__init__()
        self.params = params
        self.cancel_event = cancel_event
        self.show_plots = show_plots

    @QtCore.Slot() if QT_LIB == 'PySide6' else QtCore.pyqtSlot()
    def run(self):
        """
        Run the full pipeline with cooperative cancel + per-iteration progress callback.
        """
        try:
            num_iter = int(self.params.get('num_iter', 100))

            def _cb(it, total, u_source, u_target, target_canvas, mse):
                # iteration number is 1-based for UI smoothness
                if self.cancel_event.is_set():
                    raise RuntimeError("CANCELED")
                self.progress.emit(int(it))
                if self.show_plots and (it == 1 or it % 1 == 0 or it == total):
                    self.progress_data.emit({
                        'it': int(it),
                        'total': int(total),
                        'recon': np.abs(u_target).astype(np.float32),
                        'phase': np.angle(u_source).astype(np.float32),
                        'target': target_canvas.astype(np.float32),
                        'error': float(mse),
                    })

            # Run the trusted pipeline (now with wavelength/pixel/size + callbacks)
            run_gen_phase_map(progress_cb=_cb,
                              cancel_event=self.cancel_event,
                              **self.params)

            if self.cancel_event.is_set():
                raise RuntimeError("CANCELED")

            # Load produced BMP for preview using user-specified SLM size
            out_bmp = self.params["output_bmp_fp"]
            nx, ny = int(self.params['slm_nx']), int(self.params['slm_ny'])
            arr = image_path_to_array(out_bmp, size=(nx, ny))
            self.finished.emit(out_bmp, arr)

        except RuntimeError as e:
            if "CANCELED" in str(e):
                self.failed.emit("Generation canceled.")
            else:
                self.failed.emit(exception_to_text(e))
        except Exception as e:
            self.failed.emit(exception_to_text(e))


class SLMWorker(QtCore.QObject):
    connected = QtCore.Signal(
        str) if QT_LIB == 'PySide6' else QtCore.pyqtSignal(str)
    disconnected = QtCore.Signal() if QT_LIB == 'PySide6' else QtCore.pyqtSignal()
    status = QtCore.Signal(
        str) if QT_LIB == 'PySide6' else QtCore.pyqtSignal(str)
    failed = QtCore.Signal(
        str) if QT_LIB == 'PySide6' else QtCore.pyqtSignal(str)
    head_temp = QtCore.Signal(
        float) if QT_LIB == 'PySide6' else QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.slm: Optional[SLM] = None

    @QtCore.Slot() if QT_LIB == 'PySide6' else QtCore.pyqtSlot()
    def open(self):
        try:
            if self.slm is not None:
                self.status.emit("SLM already open.")
                return
            self.slm = SLM()
            self.slm.open()
            serial = self.slm.get_head_serial(0)
            self.connected.emit(
                f"Connected ({serial}); {self.slm._num_devices} device(s)")
        except Exception as e:
            self.slm = None
            self.failed.emit(exception_to_text(e))

    @QtCore.Slot() if QT_LIB == 'PySide6' else QtCore.pyqtSlot()
    def close(self):
        try:
            if self.slm is not None:
                self.slm.close()
                self.slm = None
            self.disconnected.emit()
        except Exception as e:
            self.failed.emit(exception_to_text(e))

    def _guard(self):
        if self.slm is None:
            raise SLMError("SLM is not connected.")

    @QtCore.Slot(str, int, int, int) if QT_LIB == 'PySide6' else QtCore.pyqtSlot(str, int, int, int)
    def send_bmp_to_slot(self, bmp_path: str, slot: int, xpix: int, ypix: int):
        try:
            self._guard()
            arr = image_path_to_array(bmp_path, size=(xpix, ypix)).flatten()
            self.slm.write_frame_array(arr, xpix, ypix, slot)
            self.status.emit(f"Wrote BMP to frame memory slot {slot}.")
        except Exception as e:
            self.failed.emit(exception_to_text(e))

    @QtCore.Slot(int) if QT_LIB == 'PySide6' else QtCore.pyqtSlot(int)
    def change_display_slot(self, slot: int):
        try:
            self._guard()
            self.slm.change_display_slot(slot)
            self.status.emit(f"Display switched to slot {slot}.")
        except Exception as e:
            self.failed.emit(exception_to_text(e))

    @QtCore.Slot() if QT_LIB == 'PySide6' else QtCore.pyqtSlot()
    def read_temperature(self):
        try:
            self._guard()
            head, cb = self.slm.get_temperature()
            self.head_temp.emit(head)
            self.status.emit(
                f"Temperature — Head: {head:.1f} °C | Control Box: {cb:.1f} °C")
        except Exception as e:
            self.failed.emit(exception_to_text(e))

    @QtCore.Slot(int, int, int) if QT_LIB == 'PySide6' else QtCore.pyqtSlot(int, int, int)
    def read_frame_memory(self, slot: int, xpix: int, ypix: int):
        try:
            self._guard()
            data = self.slm.read_frame_memory(slot, xpix, ypix)
            arr = np.array(data, dtype=np.uint8).reshape(ypix, xpix)
            np.save(HERE / 'tmp_fmem.npy', arr)
            self.status.emit("FMEM image saved (temporary).")
        except Exception as e:
            self.failed.emit(exception_to_text(e))

    @QtCore.Slot(int, int) if QT_LIB == 'PySide6' else QtCore.pyqtSlot(int, int)
    def read_display_image(self, xpix: int, ypix: int):
        try:
            self._guard()
            data = self.slm.read_display_image(xpix, ypix)
            arr = np.array(data, dtype=np.uint8).reshape(ypix, xpix)
            np.save(HERE / 'tmp_display.npy', arr)
            self.status.emit("Display image saved (temporary).")
        except Exception as e:
            self.failed.emit(exception_to_text(e))

# ---------------------- Viewer Dialog ----------------------


class ImageViewer(QtWidgets.QDialog):
    def __init__(self, title: str, arr: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(700, 550)
        lay = QtWidgets.QVBoxLayout(self)
        lbl = QtWidgets.QLabel()
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setPixmap(to_grayscale_qpixmap(arr).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        lay.addWidget(lbl)

# ---------------------- Main Window ----------------------


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SLM CGH Studio")
        self.resize(1320, 860)

        self.cancel_event = threading.Event()
        self.progress_dialog: Optional[QtWidgets.QProgressDialog] = None
        self.temp_timer: Optional[QtCore.QTimer] = None
        self.temp_threshold_c: float = 35.0
        self.gs_window: Optional[GSProgressWindow] = None

        self._setup_ui()
        self._setup_threads()
        self._load_defaults_json(DEFAULTS_JSON)  # auto-apply at startup

    # ---- Threads & signals ----
    def _setup_threads(self):
        self.slm_thread = QtCore.QThread(self)
        self.slm_worker = SLMWorker()
        self.slm_worker.moveToThread(self.slm_thread)
        self.slm_thread.start()

        self._connect(self.btn_connect.clicked, self.slm_worker.open)
        self._connect(self.btn_disconnect.clicked, self.slm_worker.close)
        self._connect(self.slm_worker.connected, self.on_slm_connected)
        self._connect(self.slm_worker.disconnected, self.on_slm_disconnected)
        self._connect(self.slm_worker.status, self.append_log)
        self._connect(self.slm_worker.failed, self.append_error)
        self._connect(self.slm_worker.head_temp, self.on_head_temp)

    def _connect(self, signal, slot):
        try:
            signal.connect(slot)
        except Exception:
            signal.connect(slot)

    # ---- UI ----
    def _setup_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Top bar: defaults JSON controls
        top = QtWidgets.QHBoxLayout()
        self.ed_defaults = QtWidgets.QLineEdit(str(DEFAULTS_JSON))
        self.btn_load_defaults = QtWidgets.QPushButton("Load Defaults JSON")
        self.btn_save_defaults = QtWidgets.QPushButton(
            "Save Current as Defaults")
        self.btn_load_defaults.clicked.connect(
            lambda: self._pick_file(self.ed_defaults, "JSON (*.json)"))
        self.btn_load_defaults.clicked.connect(
            lambda: self._load_defaults_json(Path(self.ed_defaults.text().strip())))
        self.btn_save_defaults.clicked.connect(
            lambda: self._save_defaults_json(Path(self.ed_defaults.text().strip())))
        top.addWidget(QtWidgets.QLabel("Defaults:"))
        top.addWidget(self.ed_defaults, 1)
        top.addWidget(self.btn_load_defaults)
        top.addWidget(self.btn_save_defaults)
        layout.addLayout(top)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        # --- Generate tab
        self.tab_gen = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_gen, "1) Generate CGH")
        self._build_tab_gen()

        # --- SLM parameters tab
        self.tab_slm_params = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_slm_params, "2) SLM Parameters")
        self._build_tab_slm_params()

        # --- Vortex mask tab
        self.tab_vortex = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_vortex, "3) Vortex Mask")
        self._build_tab_vortex()

        # --- SLM tab
        self.tab_slm = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_slm, "4) SLM Control")
        self._build_tab_slm()

        # Bottom log
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("Status and errors appear here...")
        layout.addWidget(self.log, 0)

    def _build_tab_gen(self):
        lay = QtWidgets.QGridLayout(self.tab_gen)

        # File pickers
        self.ed_target = QtWidgets.QLineEdit()
        self.ed_target.textChanged.connect(self._refresh_target_preview)
        self.btn_target = QtWidgets.QPushButton("Browse…")
        self.btn_target.clicked.connect(lambda: self._pick_file(
            self.ed_target, "Target image (*.bmp *.png *.jpg *.tif)"))

        self.ed_correction = QtWidgets.QLineEdit()  # optional now
        self.btn_correction = QtWidgets.QPushButton("Browse…")
        self.btn_correction.clicked.connect(lambda: self._pick_file(
            self.ed_correction, "Correction pattern (*.bmp *.png *.jpg *.tif)"))

        self.ed_source = QtWidgets.QLineEdit()
        self.btn_source = QtWidgets.QPushButton("Browse…")
        self.btn_source.clicked.connect(lambda: self._pick_file(
            self.ed_source, "Optional source amplitude (*.bmp *.png *.jpg *.tif)"))

        self.ed_output = QtWidgets.QLineEdit(
            str(HERE / "output" / "phase_mask.bmp"))
        self.btn_output = QtWidgets.QPushButton("Save As…")
        self.btn_output.clicked.connect(
            lambda: self._save_file(self.ed_output, "BMP (*.bmp)"))

        # Optics params
        self.dsb_wavelength_nm = QtWidgets.QDoubleSpinBox()
        self.dsb_wavelength_nm.setRange(200.0, 2000.0)
        self.dsb_wavelength_nm.setValue(775.0)
        self.dsb_wavelength_nm.setDecimals(3)
        self.dsb_wavelength_nm.setSuffix(" nm")

        # Params
        self.cmb_size_mode = QtWidgets.QComboBox()
        self.cmb_size_mode.addItems(["resized", "tiled"])

        self.cmb_method = QtWidgets.QComboBox()
        self.cmb_method.addItems(["far", "near"])

        self.dsb_z = QtWidgets.QDoubleSpinBox()
        self.dsb_z.setRange(-10.0, 10.0)
        self.dsb_z.setSingleStep(0.01)
        self.dsb_z.setValue(0.4)
        self.dsb_z.setSuffix(" m")

        self.dsb_M = QtWidgets.QDoubleSpinBox()
        self.dsb_M.setRange(0.01, 100.0)
        self.dsb_M.setSingleStep(0.01)
        self.dsb_M.setValue(1.0)

        self.spin_iter = QtWidgets.QSpinBox()
        self.spin_iter.setRange(1, 10000)
        self.spin_iter.setValue(100)

        self.dsb_tilt_x = QtWidgets.QDoubleSpinBox()
        self.dsb_tilt_x.setRange(-89.0, 89.0)
        self.dsb_tilt_x.setDecimals(3)
        self.dsb_tilt_x.setSuffix(" °")
        self.dsb_tilt_y = QtWidgets.QDoubleSpinBox()
        self.dsb_tilt_y.setRange(-89.0, 89.0)
        self.dsb_tilt_y.setDecimals(3)
        self.dsb_tilt_y.setSuffix(" °")

        # Options
        self.chk_plot = QtWidgets.QCheckBox(
            "GS progress plots (separate window)")
        self.chk_auto_load = QtWidgets.QCheckBox(
            "After generation: send BMP to slot")
        self.spin_auto_slot = QtWidgets.QSpinBox()
        self.spin_auto_slot.setRange(0, 818)
        self.spin_auto_slot.setValue(0)

        self.chk_auto_display = QtWidgets.QCheckBox(
            "Also switch display to that slot")

        # Actions
        self.btn_generate = QtWidgets.QPushButton("Generate Phase + Save BMP")
        self.btn_generate.clicked.connect(self.on_generate_clicked)

        # Previews
        self.lbl_preview_target = QtWidgets.QLabel()
        self.lbl_preview_target.setMinimumSize(300, 240)
        self.lbl_preview_target.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lbl_preview_target.installEventFilter(self)
        self.lbl_preview_phase = QtWidgets.QLabel()
        self.lbl_preview_phase.setMinimumSize(300, 240)
        self.lbl_preview_phase.setFrameShape(QtWidgets.QFrame.StyledPanel)

        # Layout
        r = 0
        lay.addWidget(QtWidgets.QLabel("Target image:"), r, 0)
        lay.addWidget(self.ed_target, r, 1, 1, 3)
        lay.addWidget(self.btn_target, r, 4)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Correction pattern (optional):"), r, 0)
        lay.addWidget(self.ed_correction, r, 1, 1, 3)
        lay.addWidget(self.btn_correction, r, 4)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Source amplitude (optional):"), r, 0)
        lay.addWidget(self.ed_source, r, 1, 1, 3)
        lay.addWidget(self.btn_source, r, 4)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Output BMP path:"), r, 0)
        lay.addWidget(self.ed_output, r, 1, 1, 3)
        lay.addWidget(self.btn_output, r, 4)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Wavelength:"), r, 0)
        lay.addWidget(self.dsb_wavelength_nm, r, 1)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Size mode:"), r, 0)
        lay.addWidget(self.cmb_size_mode, r, 1)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Propagation method:"), r, 0)
        lay.addWidget(self.cmb_method, r, 1)
        lay.addWidget(QtWidgets.QLabel("z (near-field):"), r, 2)
        lay.addWidget(self.dsb_z, r, 3)
        r += 1

        lay.addWidget(QtWidgets.QLabel("M (virtual 4f near-field):"), r, 0)
        lay.addWidget(self.dsb_M, r, 1)
        lay.addWidget(QtWidgets.QLabel("# Iterations:"), r, 2)
        lay.addWidget(self.spin_iter, r, 3)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Tilt X (deg):"), r, 0)
        lay.addWidget(self.dsb_tilt_x, r, 1)
        lay.addWidget(QtWidgets.QLabel("Tilt Y (deg):"), r, 2)
        lay.addWidget(self.dsb_tilt_y, r, 3)
        r += 1

        # keep the plots row
        lay.addWidget(self.chk_plot, r, 0, 1, 5)
        r += 1

        # >>> NEW: auto-load to slot (insert exactly here)
        lay.addWidget(self.chk_auto_load, r, 0, 1, 2)
        lay.addWidget(QtWidgets.QLabel("Slot:"), r, 2)
        lay.addWidget(self.spin_auto_slot, r, 3)
        r += 1
        lay.addWidget(self.chk_auto_display, r, 0, 1, 2)
        r += 1
        # <<< NEW end

        # then the Generate button row stays after the new options
        lay.addWidget(self.btn_generate, r, 0, 1, 5)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Target preview"), r, 0)
        lay.addWidget(QtWidgets.QLabel("Phase/BMP preview"), r, 2)
        r += 1
        lay.addWidget(self.lbl_preview_target, r, 0, 1, 2)
        lay.addWidget(self.lbl_preview_phase, r, 2, 1, 3)
        r += 1

        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

    def _build_tab_slm_params(self):
        lay = QtWidgets.QGridLayout(self.tab_slm_params)

        self.ed_slm_params = QtWidgets.QLineEdit()
        self.btn_slm_params_browse = QtWidgets.QPushButton("Browse.")
        self.btn_slm_params_load = QtWidgets.QPushButton("Load")
        self.btn_slm_params_browse.clicked.connect(
            lambda: self._pick_file(self.ed_slm_params, "JSON (*.json)"))
        self.btn_slm_params_load.clicked.connect(
            self.on_load_slm_params_clicked)

        self.spin_slm_nx = QtWidgets.QSpinBox()
        self.spin_slm_nx.setRange(64, 8192)
        self.spin_slm_nx.setValue(1272)
        self.spin_slm_ny = QtWidgets.QSpinBox()
        self.spin_slm_ny.setRange(64, 8192)
        self.spin_slm_ny.setValue(1024)

        self.dsb_px_side_m = QtWidgets.QDoubleSpinBox()
        self.dsb_px_side_m.setRange(1e-8, 1e-3)
        self.dsb_px_side_m.setDecimals(10)
        self.dsb_px_side_m.setValue(12.5e-6)
        self.dsb_px_side_m.setSingleStep(1e-6)
        self.dsb_px_side_m.setSuffix(" m")

        self.dsb_py_side_m = QtWidgets.QDoubleSpinBox()
        self.dsb_py_side_m.setRange(1e-8, 1e-3)
        self.dsb_py_side_m.setDecimals(10)
        self.dsb_py_side_m.setValue(12.5e-6)
        self.dsb_py_side_m.setSingleStep(1e-6)
        self.dsb_py_side_m.setSuffix(" m")

        self.dsb_fill_factor = QtWidgets.QDoubleSpinBox()
        self.dsb_fill_factor.setRange(0.0, 100.0)
        self.dsb_fill_factor.setDecimals(1)
        self.dsb_fill_factor.setValue(96.8)
        self.dsb_fill_factor.setSuffix(" %")

        self.spin_c2pi2unit = QtWidgets.QSpinBox()
        self.spin_c2pi2unit.setRange(1, 255)
        self.spin_c2pi2unit.setValue(204)

        r = 0
        lay.addWidget(QtWidgets.QLabel("SLM params JSON:"), r, 0)
        lay.addWidget(self.ed_slm_params, r, 1, 1, 2)
        lay.addWidget(self.btn_slm_params_browse, r, 3)
        lay.addWidget(self.btn_slm_params_load, r, 4)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Nx (pixels):"), r, 0)
        lay.addWidget(self.spin_slm_nx, r, 1)
        lay.addWidget(QtWidgets.QLabel("Ny (pixels):"), r, 2)
        lay.addWidget(self.spin_slm_ny, r, 3)
        r += 1

        lay.addWidget(QtWidgets.QLabel("px_side_m:"), r, 0)
        lay.addWidget(self.dsb_px_side_m, r, 1)
        lay.addWidget(QtWidgets.QLabel("py_side_m:"), r, 2)
        lay.addWidget(self.dsb_py_side_m, r, 3)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Fill factor (%):"), r, 0)
        lay.addWidget(self.dsb_fill_factor, r, 1)
        lay.addWidget(QtWidgets.QLabel("c2pi2unit:"), r, 2)
        lay.addWidget(self.spin_c2pi2unit, r, 3)
        r += 1

        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

    def _build_tab_vortex(self):
        lay = QtWidgets.QGridLayout(self.tab_vortex)

        self.ed_beam_params = QtWidgets.QLineEdit()
        self.btn_beam_params_browse = QtWidgets.QPushButton("Browse.")
        self.btn_beam_params_load = QtWidgets.QPushButton("Load")
        self.btn_beam_params_browse.clicked.connect(
            lambda: self._pick_file(self.ed_beam_params, "JSON (*.json)"))
        self.btn_beam_params_load.clicked.connect(
            self.on_load_beam_params_clicked)

        self.dsb_beam_lambda_m = QtWidgets.QDoubleSpinBox()
        self.dsb_beam_lambda_m.setRange(1e-9, 1e-3)
        self.dsb_beam_lambda_m.setDecimals(10)
        self.dsb_beam_lambda_m.setValue(7.75e-7)
        self.dsb_beam_lambda_m.setSingleStep(1e-9)
        self.dsb_beam_lambda_m.setSuffix(" m")

        self.ed_vortex_calib = QtWidgets.QLineEdit()
        self.btn_vortex_calib = QtWidgets.QPushButton("Browse.")
        self.btn_vortex_calib.clicked.connect(
            lambda: self._pick_file(self.ed_vortex_calib, "BMP (*.bmp)"))

        self.ed_vortex_output_dir = QtWidgets.QLineEdit()
        self.btn_vortex_output_dir = QtWidgets.QPushButton("Browse.")
        self.btn_vortex_output_dir.clicked.connect(
            lambda: self._pick_dir(self.ed_vortex_output_dir))

        self.ed_vortex_output_name = QtWidgets.QLineEdit()

        self.spin_vortex_ell = QtWidgets.QSpinBox()
        self.spin_vortex_ell.setRange(1, 1000)
        self.spin_vortex_ell.setValue(1)

        self.dsb_vortex_sft_x_mm = QtWidgets.QDoubleSpinBox()
        self.dsb_vortex_sft_x_mm.setRange(-1000.0, 1000.0)
        self.dsb_vortex_sft_x_mm.setDecimals(3)
        self.dsb_vortex_sft_x_mm.setSingleStep(0.01)
        self.dsb_vortex_sft_x_mm.setSuffix(" mm")

        self.dsb_vortex_sft_y_mm = QtWidgets.QDoubleSpinBox()
        self.dsb_vortex_sft_y_mm.setRange(-1000.0, 1000.0)
        self.dsb_vortex_sft_y_mm.setDecimals(3)
        self.dsb_vortex_sft_y_mm.setSingleStep(0.01)
        self.dsb_vortex_sft_y_mm.setSuffix(" mm")

        self.dsb_vortex_aperture_mm = QtWidgets.QDoubleSpinBox()
        self.dsb_vortex_aperture_mm.setRange(0.0, 1000.0)
        self.dsb_vortex_aperture_mm.setDecimals(3)
        self.dsb_vortex_aperture_mm.setSingleStep(0.01)
        self.dsb_vortex_aperture_mm.setSuffix(" mm")

        self.chk_vortex_use_forked = QtWidgets.QCheckBox(
            "Use forked grating")
        self.chk_vortex_use_forked.setChecked(True)
        self.chk_vortex_force_zero = QtWidgets.QCheckBox(
            "Force pixel-zero center")
        self.chk_vortex_force_zero.setChecked(True)
        self.chk_vortex_use_forked.stateChanged.connect(
            self._update_vortex_steer_mode)

        self.cmb_vortex_steer_mode = QtWidgets.QComboBox()
        self.cmb_vortex_steer_mode.addItems(["none", "angle", "shift"])
        self.cmb_vortex_steer_mode.currentTextChanged.connect(
            self._update_vortex_steer_mode)

        self.dsb_vortex_theta_x_deg = QtWidgets.QDoubleSpinBox()
        self.dsb_vortex_theta_x_deg.setRange(-89.0, 89.0)
        self.dsb_vortex_theta_x_deg.setDecimals(4)
        self.dsb_vortex_theta_x_deg.setSuffix(" deg")

        self.dsb_vortex_theta_y_deg = QtWidgets.QDoubleSpinBox()
        self.dsb_vortex_theta_y_deg.setRange(-89.0, 89.0)
        self.dsb_vortex_theta_y_deg.setDecimals(4)
        self.dsb_vortex_theta_y_deg.setSuffix(" deg")

        self.dsb_vortex_delta_x_mm = QtWidgets.QDoubleSpinBox()
        self.dsb_vortex_delta_x_mm.setRange(-1000.0, 1000.0)
        self.dsb_vortex_delta_x_mm.setDecimals(4)
        self.dsb_vortex_delta_x_mm.setSuffix(" mm")

        self.dsb_vortex_delta_y_mm = QtWidgets.QDoubleSpinBox()
        self.dsb_vortex_delta_y_mm.setRange(-1000.0, 1000.0)
        self.dsb_vortex_delta_y_mm.setDecimals(4)
        self.dsb_vortex_delta_y_mm.setSuffix(" mm")

        self.dsb_vortex_focal_length_m = QtWidgets.QDoubleSpinBox()
        self.dsb_vortex_focal_length_m.setRange(0.001, 10.0)
        self.dsb_vortex_focal_length_m.setDecimals(4)
        self.dsb_vortex_focal_length_m.setValue(0.2)
        self.dsb_vortex_focal_length_m.setSuffix(" m")

        self.btn_vortex_generate = QtWidgets.QPushButton(
            "Generate Vortex Mask")
        self.btn_vortex_generate.clicked.connect(
            self.on_generate_vortex_clicked)

        r = 0
        lay.addWidget(QtWidgets.QLabel("Beam params JSON:"), r, 0)
        lay.addWidget(self.ed_beam_params, r, 1, 1, 2)
        lay.addWidget(self.btn_beam_params_browse, r, 3)
        lay.addWidget(self.btn_beam_params_load, r, 4)
        r += 1

        lay.addWidget(QtWidgets.QLabel("lambda_m:"), r, 0)
        lay.addWidget(self.dsb_beam_lambda_m, r, 1)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Calibration mask (optional):"), r, 0)
        lay.addWidget(self.ed_vortex_calib, r, 1, 1, 3)
        lay.addWidget(self.btn_vortex_calib, r, 4)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Output dir:"), r, 0)
        lay.addWidget(self.ed_vortex_output_dir, r, 1, 1, 3)
        lay.addWidget(self.btn_vortex_output_dir, r, 4)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Output filename (optional):"), r, 0)
        lay.addWidget(self.ed_vortex_output_name, r, 1, 1, 3)
        r += 1

        lay.addWidget(QtWidgets.QLabel("ell:"), r, 0)
        lay.addWidget(self.spin_vortex_ell, r, 1)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Shift X (mm):"), r, 0)
        lay.addWidget(self.dsb_vortex_sft_x_mm, r, 1)
        lay.addWidget(QtWidgets.QLabel("Shift Y (mm):"), r, 2)
        lay.addWidget(self.dsb_vortex_sft_y_mm, r, 3)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Aperture radius (mm):"), r, 0)
        lay.addWidget(self.dsb_vortex_aperture_mm, r, 1)
        r += 1

        lay.addWidget(self.chk_vortex_use_forked, r, 0, 1, 2)
        lay.addWidget(self.chk_vortex_force_zero, r, 2, 1, 2)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Steering mode:"), r, 0)
        lay.addWidget(self.cmb_vortex_steer_mode, r, 1)
        r += 1

        lay.addWidget(QtWidgets.QLabel("theta_x_deg:"), r, 0)
        lay.addWidget(self.dsb_vortex_theta_x_deg, r, 1)
        lay.addWidget(QtWidgets.QLabel("theta_y_deg:"), r, 2)
        lay.addWidget(self.dsb_vortex_theta_y_deg, r, 3)
        r += 1

        lay.addWidget(QtWidgets.QLabel("delta_x_mm:"), r, 0)
        lay.addWidget(self.dsb_vortex_delta_x_mm, r, 1)
        lay.addWidget(QtWidgets.QLabel("delta_y_mm:"), r, 2)
        lay.addWidget(self.dsb_vortex_delta_y_mm, r, 3)
        r += 1

        lay.addWidget(QtWidgets.QLabel("focal_length_m:"), r, 0)
        lay.addWidget(self.dsb_vortex_focal_length_m, r, 1)
        r += 1

        lay.addWidget(self.btn_vortex_generate, r, 0, 1, 5)
        r += 1

        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)
        self._update_vortex_steer_mode()

    def _build_tab_slm(self):
        lay = QtWidgets.QGridLayout(self.tab_slm)

        # Connect/disconnect
        self.btn_connect = QtWidgets.QPushButton("Connect SLM")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)

        # Slot controls
        self.spin_slot = QtWidgets.QSpinBox()
        self.spin_slot.setRange(0, 818)
        self.spin_slot.setValue(0)
        self.btn_send_to_slot = QtWidgets.QPushButton(
            "Write current BMP to slot")
        self.btn_change_display = QtWidgets.QPushButton(
            "Change display to slot")
        self.btn_temp = QtWidgets.QPushButton("Read temperature")

        self.btn_send_to_slot.clicked.connect(self.on_send_to_slot)
        self.btn_change_display.clicked.connect(self.on_change_display)
        self.btn_temp.clicked.connect(self.on_temp_clicked)

        # Inspect frames
        self.btn_check_display = QtWidgets.QPushButton("Check DISPLAYED image")
        self.btn_check_fmem = QtWidgets.QPushButton(
            "Check image in CURRENT slot")
        self.btn_check_display.clicked.connect(self.on_check_display)
        self.btn_check_fmem.clicked.connect(self.on_check_fmem)

        # Watchdog
        self.chk_watchdog = QtWidgets.QCheckBox(
            "Temperature watchdog (head only)")
        self.spin_interval = QtWidgets.QSpinBox()
        self.spin_interval.setRange(1, 3600)
        self.spin_interval.setValue(10)
        self.spin_interval.setSuffix(" s")
        self.chk_watchdog.stateChanged.connect(self.on_watchdog_toggled)

        # Current BMP path (from gen tab)
        self.ed_current_bmp = QtWidgets.QLineEdit()
        self.btn_pick_current_bmp = QtWidgets.QPushButton("…")
        self.btn_pick_current_bmp.clicked.connect(
            lambda: self._pick_file(self.ed_current_bmp, "BMP (*.bmp)"))

        # Layout
        r = 0
        lay.addWidget(self.btn_connect, r, 0)
        lay.addWidget(self.btn_disconnect, r, 1)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Current BMP:"), r, 0)
        lay.addWidget(self.ed_current_bmp, r, 1, 1, 3)
        lay.addWidget(self.btn_pick_current_bmp, r, 4)
        r += 1

        lay.addWidget(QtWidgets.QLabel("Slot:"), r, 0)
        lay.addWidget(self.spin_slot, r, 1)
        lay.addWidget(self.btn_send_to_slot, r, 2, 1, 2)
        lay.addWidget(self.btn_change_display, r, 4)
        r += 1

        lay.addWidget(self.btn_check_display, r, 0)
        lay.addWidget(self.btn_check_fmem, r, 1)
        lay.addWidget(self.btn_temp, r, 2)
        lay.addWidget(self.chk_watchdog, r, 3)
        lay.addWidget(self.spin_interval, r, 4)
        r += 1

        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

    def eventFilter(self, obj, event):
        if obj is self.lbl_preview_target and event.type() == QtCore.QEvent.Resize:
            # re-scale to the new contents rect on every resize
            QtCore.QTimer.singleShot(0, self._refresh_target_preview)
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        try:
            defaults_path = Path(self.ed_defaults.text().strip())
            if defaults_path:
                self._save_defaults_json(defaults_path)
        except Exception as e:
            self.append_error(f"Failed to auto-save defaults: {e}")
        super().closeEvent(event)

    # ---- Actions ----
    def _pick_file(self, line_edit: QtWidgets.QLineEdit, filter_str: str):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select file", str(HERE), filter_str)
        if fn:
            line_edit.setText(fn)
            if line_edit is self.ed_target:
                self._refresh_target_preview()

    def _save_file(self, line_edit: QtWidgets.QLineEdit, filter_str: str):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save As", line_edit.text(), filter_str)
        if fn:
            if not fn.lower().endswith(".bmp"):
                fn += ".bmp"
            line_edit.setText(fn)

    def _pick_dir(self, line_edit: QtWidgets.QLineEdit):
        fn = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder", line_edit.text())
        if fn:
            line_edit.setText(fn)

    def on_load_slm_params_clicked(self):
        path_str = self.ed_slm_params.text().strip()
        if not path_str:
            self.append_error("SLM params JSON path is empty.")
            return
        self._load_slm_params_json(Path(path_str))

    def on_load_beam_params_clicked(self):
        path_str = self.ed_beam_params.text().strip()
        if not path_str:
            self.append_error("Beam params JSON path is empty.")
            return
        self._load_beam_params_json(Path(path_str))

    def _refresh_target_preview(self):
        try:
            fn = self.ed_target.text().strip()
            if not fn or not os.path.exists(fn):
                # show a neutral message in the label instead of a blank/white box
                self.lbl_preview_target.setText("No target loaded")
                self.lbl_preview_target.setAlignment(QtCore.Qt.AlignCenter)
                self.lbl_preview_target.setStyleSheet(
                    "color: gray; font-style: italic;")
                return

            # cache the loaded array so a pure resize doesn't re-read disk
            if not hasattr(self, "_target_preview_cache"):
                self._target_preview_cache = {"path": None, "arr": None}

            # only reload if path changed
            if self._target_preview_cache["path"] != fn:
                nx, ny = self.spin_slm_nx.value(), self.spin_slm_ny.value()
                self._target_preview_cache["arr"] = image_path_to_array(
                    fn, size=(nx, ny))
                self._target_preview_cache["path"] = fn

            arr = self._target_preview_cache["arr"]
            if arr is None:
                return

            # scale to the *contents* rect (excludes frame/borders)
            target_size = self.lbl_preview_target.contentsRect().size()
            if target_size.width() <= 0 or target_size.height() <= 0:
                # label not laid out yet — try again right after layout
                QtCore.QTimer.singleShot(0, self._refresh_target_preview)
                return

            pm = to_grayscale_qpixmap(arr).scaled(
                target_size,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.lbl_preview_target.setPixmap(pm)

        except Exception as e:
            self.append_error(exception_to_text(e))

    def _update_vortex_steer_mode(self):
        use_forked = self.chk_vortex_use_forked.isChecked()
        mode = self.cmb_vortex_steer_mode.currentText()
        angle = use_forked and mode == "angle"
        shift = use_forked and mode == "shift"

        self.dsb_vortex_theta_x_deg.setEnabled(angle)
        self.dsb_vortex_theta_y_deg.setEnabled(angle)
        self.dsb_vortex_delta_x_mm.setEnabled(shift)
        self.dsb_vortex_delta_y_mm.setEnabled(shift)
        self.dsb_vortex_focal_length_m.setEnabled(shift)

    def on_generate_vortex_clicked(self):
        try:
            slm = SlmParams(
                nx=int(self.spin_slm_nx.value()),
                ny=int(self.spin_slm_ny.value()),
                px_m=float(self.dsb_px_side_m.value()),
                py_m=float(self.dsb_py_side_m.value()),
                c2pi2unit=int(self.spin_c2pi2unit.value()),
            )
            beam = BeamParams(lambda_m=float(self.dsb_beam_lambda_m.value()))

            ell = int(self.spin_vortex_ell.value())
            sft_x = float(self.dsb_vortex_sft_x_mm.value()) * 1e-3
            sft_y = float(self.dsb_vortex_sft_y_mm.value()) * 1e-3
            use_forked = self.chk_vortex_use_forked.isChecked()
            force_zero = self.chk_vortex_force_zero.isChecked()

            aperture_radius_m = None
            aperture_mm = float(self.dsb_vortex_aperture_mm.value())
            if aperture_mm > 0:
                aperture_radius_m = aperture_mm * 1e-3

            steer_req = None
            if use_forked:
                mode = self.cmb_vortex_steer_mode.currentText()
                if mode == "angle":
                    steer_req = SteeringRequest(
                        theta_x_deg=float(self.dsb_vortex_theta_x_deg.value()),
                        theta_y_deg=float(self.dsb_vortex_theta_y_deg.value()),
                    )
                elif mode == "shift":
                    steer_req = SteeringRequest(
                        delta_x_mm=float(self.dsb_vortex_delta_x_mm.value()),
                        delta_y_mm=float(self.dsb_vortex_delta_y_mm.value()),
                        focal_length_m=float(
                            self.dsb_vortex_focal_length_m.value()),
                    )

            calib_mask = None
            calib_path = self.ed_vortex_calib.text().strip()
            if calib_path:
                resolved = self._resolve_path(Path(calib_path))
                calib_mask = load_calibration_mask(resolved)

            result = generate_mask(
                slm=slm,
                beam=beam,
                ell=ell,
                sft_x=sft_x,
                sft_y=sft_y,
                steer_req=steer_req,
                use_forked=use_forked,
                aperture_radius_m=aperture_radius_m,
                force_zero=force_zero,
                calib_mask=calib_mask,
            )

            out_dir_str = self.ed_vortex_output_dir.text().strip()
            if out_dir_str:
                out_dir = self._resolve_path(Path(out_dir_str))
            else:
                out_dir = self._resolve_path(Path("masks_out"))
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = self.ed_vortex_output_name.text().strip()
            if out_name:
                if not out_name.lower().endswith(".bmp"):
                    out_name += ".bmp"
                out_path = out_dir / out_name
            else:
                out_path = out_dir / result.filename

            save_mask(out_path, result.mask_u8)
            self.append_log(f"Vortex mask saved: {out_path}")
            if result.steer:
                self.append_log(
                    "Steer mode: {mode} | dx={dx:.3f} mm, dy={dy:.3f} mm | "
                    "clamped={clamped}".format(
                        mode=result.steer.mode,
                        dx=result.steer.delta_x_mm,
                        dy=result.steer.delta_y_mm,
                        clamped=result.steer.clamped,
                    )
                )
        except Exception as e:
            self.append_error(exception_to_text(e))

    def _build_params(self):
        """Collect UI params for gen_phase_map.run"""
        target = self.ed_target.text().strip()
        if not target:
            raise ValueError("Please select a Target image before generating.")

        corr = self.ed_correction.text().strip()  # optional
        src = self.ed_source.text().strip()
        out_bmp = self.ed_output.text().strip()
        ensure_parent_dir(Path(out_bmp))

        nx, ny = int(self.spin_slm_nx.value()), int(self.spin_slm_ny.value())
        wavelength_m = float(self.dsb_wavelength_nm.value()) * 1e-9
        pixel_size_m = float(self.dsb_px_side_m.value())

        params = dict(
            num_iter=int(self.spin_iter.value()),
            slm_nx=nx,
            slm_ny=ny,
            size_mode=self.cmb_size_mode.currentText(),
            correction_pattern_fp=corr,            # may be '' (optional)
            signal_2pi=int(self.spin_c2pi2unit.value()),
            target_fp=target,
            output_bmp_fp=out_bmp,
            source_fp=src if src else '',
            method=self.cmb_method.currentText(),
            z=float(self.dsb_z.value()),
            M=float(self.dsb_M.value()),
            tilt_x_deg=float(self.dsb_tilt_x.value()),
            tilt_y_deg=float(self.dsb_tilt_y.value()),
            wavelength=wavelength_m,
            pixel_size=pixel_size_m,
        )
        return params

    def on_generate_clicked(self):
        try:
            params = self._build_params()
        except Exception as e:
            self.append_error(str(e))
            return

        self.cancel_event.clear()
        self.append_log("Starting CGH generation...")

        # Determinate progress dialog
        self.progress_dialog = QtWidgets.QProgressDialog(
            "Generating CGH...", "Cancel", 0, params['num_iter'], self)
        self.progress_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.progress_dialog.setValue(0)
        self.progress_dialog.canceled.connect(self.on_cancel_generation)
        self.progress_dialog.show()

        # Optional GS live plots
        if self.chk_plot.isChecked():
            self.gs_window = GSProgressWindow(self)
            self.gs_window.show()
        else:
            self.gs_window = None

        # Worker
        self.cgh_thread = QtCore.QThread(self)
        self.cgh_worker = CGHWorker(
            params, self.cancel_event, self.chk_plot.isChecked())
        self.cgh_worker.moveToThread(self.cgh_thread)
        self.cgh_thread.started.connect(self.cgh_worker.run)
        self.cgh_worker.progress.connect(self.on_progress_tick)
        self.cgh_worker.progress_data.connect(self.on_progress_data)
        self.cgh_worker.finished.connect(self.on_cgh_finished)
        self.cgh_worker.failed.connect(self.on_cgh_failed)
        self.cgh_worker.finished.connect(self._cleanup_cgh_thread)
        self.cgh_worker.failed.connect(self._cleanup_cgh_thread)
        self.cgh_thread.start()

    def on_progress_tick(self, it: int):
        if self.progress_dialog:
            self.progress_dialog.setValue(it)

    def on_progress_data(self, data: dict):
        if self.gs_window:
            self.gs_window.update_from_data(data)

    def _cleanup_cgh_thread(self):
        if self.progress_dialog:
            self.progress_dialog.reset()
            self.progress_dialog = None
        try:
            self.cgh_thread.quit()
            self.cgh_thread.wait(1000)
        except Exception:
            pass
        # close GS window when done
        if self.gs_window:
            try:
                self.gs_window.close()
            except Exception:
                pass
            self.gs_window = None

    def on_cancel_generation(self):
        self.append_log("Cancel requested — generation will stop shortly.")
        self.cancel_event.set()

    def on_cgh_failed(self, msg: str):
        self.append_error(msg)

    def on_cgh_finished(self, out_bmp_path: str, phase_arr: np.ndarray):
        if self.cancel_event.is_set():
            self.append_log("Generation canceled; result ignored.")
            return
        self.append_log(f"CGH ready → {out_bmp_path}")
        self.ed_current_bmp.setText(out_bmp_path)
        try:
            self.lbl_preview_phase.setPixmap(to_grayscale_qpixmap(phase_arr).scaled(
                self.lbl_preview_phase.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        except Exception as e:
            self.append_error(exception_to_text(e))
        try:
            if self.chk_auto_load.isChecked():
                slot = int(self.spin_auto_slot.value())
                nx, ny = int(self.spin_slm_nx.value()), int(
                    self.spin_slm_ny.value())
                self._invoke_in_slm_thread(
                    self.slm_worker.send_bmp_to_slot, out_bmp_path, slot, nx, ny)
                if self.chk_auto_display.isChecked():
                    self._invoke_in_slm_thread(
                        self.slm_worker.change_display_slot, slot)
        except Exception as e:
            self.append_error(exception_to_text(e))

    def on_send_to_slot(self):
        bmp = self.ed_current_bmp.text().strip()
        if not bmp or not os.path.exists(bmp):
            self.append_error("No BMP selected or path does not exist.")
            return
        slot = int(self.spin_slot.value())
        nx, ny = int(self.spin_slm_nx.value()), int(self.spin_slm_ny.value())
        self._invoke_in_slm_thread(
            self.slm_worker.send_bmp_to_slot, bmp, slot, nx, ny)

    def on_change_display(self):
        slot = int(self.spin_slot.value())
        self._invoke_in_slm_thread(self.slm_worker.change_display_slot, slot)

    def on_temp_clicked(self):
        self._invoke_in_slm_thread(self.slm_worker.read_temperature)

    def on_check_display(self):
        nx, ny = int(self.spin_slm_nx.value()), int(self.spin_slm_ny.value())
        self._invoke_in_slm_thread(self.slm_worker.read_display_image, nx, ny)
        QtCore.QTimer.singleShot(500, self._show_tmp_display)

    def on_check_fmem(self):
        slot = int(self.spin_slot.value())
        nx, ny = int(self.spin_slm_nx.value()), int(self.spin_slm_ny.value())
        self._invoke_in_slm_thread(
            self.slm_worker.read_frame_memory, slot, nx, ny)
        QtCore.QTimer.singleShot(500, self._show_tmp_fmem)

    def _show_tmp_display(self):
        tmp = HERE / 'tmp_display.npy'
        if tmp.exists():
            arr = np.load(tmp)
            (ImageViewer("Displayed image (current SLM output)", arr, self).exec_()
             if QT_LIB == 'PyQt5' else
             ImageViewer("Displayed image (current SLM output)", arr, self).exec())
            try:
                tmp.unlink()
            except Exception:
                pass

    def _show_tmp_fmem(self):
        tmp = HERE / 'tmp_fmem.npy'
        if tmp.exists():
            arr = np.load(tmp)
            (ImageViewer(f"Frame memory (slot {self.spin_slot.value()})", arr, self).exec_()
             if QT_LIB == 'PyQt5' else
             ImageViewer(f"Frame memory (slot {self.spin_slot.value()})", arr, self).exec())
            try:
                tmp.unlink()
            except Exception:
                pass

    def _invoke_in_slm_thread(self, func, *args):
        QtCore.QMetaObject.invokeMethod(
            self.slm_worker,
            func.__name__,
            QtCore.Qt.QueuedConnection,
            *[QtCore.Q_ARG(type(a), a) for a in args]
        )

    def on_slm_connected(self, msg: str):
        self.append_log(msg)
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)

    def on_slm_disconnected(self):
        self.append_log("SLM disconnected.")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self._stop_watchdog()

    # ---- Watchdog ----
    def on_watchdog_toggled(self, state):
        if self.chk_watchdog.isChecked():
            self._start_watchdog()
        else:
            self._stop_watchdog()

    def _start_watchdog(self):
        if self.temp_timer:
            self.temp_timer.stop()
        self.temp_timer = QtCore.QTimer(self)
        self.temp_timer.timeout.connect(
            lambda: self._invoke_in_slm_thread(self.slm_worker.read_temperature))
        self.temp_timer.start(self.spin_interval.value() * 1000)
        self.append_log(
            f"Temperature watchdog enabled: threshold {self.temp_threshold_c:.1f} °C, every {self.spin_interval.value()} s."
        )

    def _stop_watchdog(self):
        if self.temp_timer:
            self.temp_timer.stop()
            self.temp_timer = None
            self.append_log("Temperature watchdog disabled.")

    def on_head_temp(self, head_c: float):
        if self.chk_watchdog.isChecked() and head_c > self.temp_threshold_c:
            QtWidgets.QMessageBox.warning(
                self, "Temperature Watchdog",
                f"Head temperature {head_c:.1f} °C exceeds {self.temp_threshold_c:.1f} °C !")
        # Status text already emitted by worker

    # ---- Logging ----
    def append_log(self, text: str):
        self.log.appendPlainText(text)

    def append_error(self, text: str):
        self.log.appendPlainText("ERROR: " + text)

    # ---- Defaults JSON ----
    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (ROOT / path).resolve()

    def _load_slm_params_json(self, path: Path) -> bool:
        resolved = self._resolve_path(path)
        try:
            with open(resolved, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception as e:
            self.append_error(f"Failed to load SLM params JSON: {e}")
            return False

        if 'Nx' in cfg:
            self.spin_slm_nx.setValue(int(cfg['Nx']))
        if 'Ny' in cfg:
            self.spin_slm_ny.setValue(int(cfg['Ny']))
        if 'px_side_m' in cfg:
            self.dsb_px_side_m.setValue(float(cfg['px_side_m']))
        if 'py_side_m' in cfg:
            self.dsb_py_side_m.setValue(float(cfg['py_side_m']))
        if 'Fill_factor_percent' in cfg:
            self.dsb_fill_factor.setValue(float(cfg['Fill_factor_percent']))
        if 'c2pi2unit' in cfg:
            self.spin_c2pi2unit.setValue(int(cfg['c2pi2unit']))

        self.append_log(f"SLM parameters loaded from: {resolved}")
        return True

    def _load_beam_params_json(self, path: Path) -> bool:
        resolved = self._resolve_path(path)
        try:
            beam = load_beam_params(resolved)
        except Exception as e:
            self.append_error(f"Failed to load beam params JSON: {e}")
            return False

        self.dsb_beam_lambda_m.setValue(float(beam.lambda_m))
        self.append_log(f"Beam parameters loaded from: {resolved}")
        return True

    def _load_defaults_json(self, path: Path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception as e:
            self.append_log(f"(Optional) defaults JSON not loaded: {e}")
            return

        # Apply to UI with fallback
        slm_params_fp = cfg.get('slm_params_fp', str(DEFAULT_SLM_PARAMS_JSON))
        self.ed_slm_params.setText(slm_params_fp)
        slm_loaded = self._load_slm_params_json(Path(slm_params_fp))
        if not slm_loaded:
            self.spin_slm_nx.setValue(int(cfg.get('slm_nx', 1272)))
            self.spin_slm_ny.setValue(int(cfg.get('slm_ny', 1024)))
            if 'px_side_m' in cfg:
                self.dsb_px_side_m.setValue(float(cfg['px_side_m']))
            elif 'pixel_size_um' in cfg:
                self.dsb_px_side_m.setValue(float(cfg['pixel_size_um']) * 1e-6)
            if 'py_side_m' in cfg:
                self.dsb_py_side_m.setValue(float(cfg['py_side_m']))
            elif 'pixel_size_um' in cfg:
                self.dsb_py_side_m.setValue(float(cfg['pixel_size_um']) * 1e-6)
            self.dsb_fill_factor.setValue(
                float(cfg.get('fill_factor_percent', 96.8)))
            self.spin_c2pi2unit.setValue(
                int(cfg.get('c2pi2unit', cfg.get('signal_2pi', 204))))

        beam_params_fp = cfg.get(
            'beam_params_fp', str(DEFAULT_BEAM_PARAMS_JSON))
        self.ed_beam_params.setText(beam_params_fp)
        beam_loaded = self._load_beam_params_json(Path(beam_params_fp))
        if not beam_loaded:
            self.dsb_beam_lambda_m.setValue(
                float(cfg.get('beam_lambda_m', 7.75e-7)))

        default_out_dir = str(ROOT / "masks_out")
        self.ed_vortex_calib.setText(cfg.get('vortex_calib_mask_fp', ''))
        self.ed_vortex_output_dir.setText(
            cfg.get('vortex_output_dir', default_out_dir))
        self.ed_vortex_output_name.setText(cfg.get('vortex_output_name', ''))
        self.spin_vortex_ell.setValue(int(cfg.get('vortex_ell', 1)))
        self.dsb_vortex_sft_x_mm.setValue(
            float(cfg.get('vortex_sft_x_mm', 0.0)))
        self.dsb_vortex_sft_y_mm.setValue(
            float(cfg.get('vortex_sft_y_mm', 0.0)))
        self.dsb_vortex_aperture_mm.setValue(
            float(cfg.get('vortex_aperture_radius_mm', 0.0)))
        self.chk_vortex_use_forked.setChecked(
            bool(cfg.get('vortex_use_forked', True)))
        self.chk_vortex_force_zero.setChecked(
            bool(cfg.get('vortex_force_zero', True)))
        self.cmb_vortex_steer_mode.setCurrentText(
            cfg.get('vortex_steer_mode', 'none'))
        self.dsb_vortex_theta_x_deg.setValue(
            float(cfg.get('vortex_theta_x_deg', 0.0)))
        self.dsb_vortex_theta_y_deg.setValue(
            float(cfg.get('vortex_theta_y_deg', 0.0)))
        self.dsb_vortex_delta_x_mm.setValue(
            float(cfg.get('vortex_delta_x_mm', 0.0)))
        self.dsb_vortex_delta_y_mm.setValue(
            float(cfg.get('vortex_delta_y_mm', 0.0)))
        self.dsb_vortex_focal_length_m.setValue(
            float(cfg.get('vortex_focal_length_m', 0.2)))
        self.ed_target.setText(cfg.get('target_fp', self.ed_target.text()))
        self.ed_correction.setText(cfg.get('correction_pattern_fp', ''))
        self.ed_source.setText(cfg.get('source_fp', ''))
        self.ed_output.setText(cfg.get('output_bmp_fp', self.ed_output.text()))
        self.cmb_size_mode.setCurrentText(cfg.get('size_mode', 'resized'))
        self.cmb_method.setCurrentText(cfg.get('method', 'far'))
        self.dsb_z.setValue(float(cfg.get('z', 0.4)))
        self.dsb_M.setValue(float(cfg.get('M', 1.0)))
        self.spin_iter.setValue(int(cfg.get('num_iter', 100)))
        self.dsb_tilt_x.setValue(float(cfg.get('tilt_x_deg', 0.0)))
        self.dsb_tilt_y.setValue(float(cfg.get('tilt_y_deg', 0.0)))
        self.spin_slot.setValue(int(cfg.get('slot', 0)))
        self.spin_interval.setValue(int(cfg.get('watchdog_interval_s', 10)))
        self.chk_watchdog.setChecked(bool(cfg.get('watchdog_enabled', False)))
        self.chk_plot.setChecked(bool(cfg.get('show_gs_plot', False)))

        # New: optics
        self.dsb_wavelength_nm.setValue(float(cfg.get('wavelength_nm', 775.0)))

        # New: temp threshold
        self.temp_threshold_c = float(cfg.get('temperature_threshold_c', 35.0))

        self._update_vortex_steer_mode()
        self.append_log(f"Defaults loaded from: {path}")

    def _save_defaults_json(self, path: Path):
        cfg = dict(
            target_fp=self.ed_target.text().strip(),
            correction_pattern_fp=self.ed_correction.text().strip(),
            source_fp=self.ed_source.text().strip(),
            output_bmp_fp=self.ed_output.text().strip(),
            slm_params_fp=self.ed_slm_params.text().strip(),
            beam_params_fp=self.ed_beam_params.text().strip(),
            beam_lambda_m=float(self.dsb_beam_lambda_m.value()),
            vortex_calib_mask_fp=self.ed_vortex_calib.text().strip(),
            vortex_output_dir=self.ed_vortex_output_dir.text().strip(),
            vortex_output_name=self.ed_vortex_output_name.text().strip(),
            vortex_ell=int(self.spin_vortex_ell.value()),
            vortex_sft_x_mm=float(self.dsb_vortex_sft_x_mm.value()),
            vortex_sft_y_mm=float(self.dsb_vortex_sft_y_mm.value()),
            vortex_use_forked=bool(self.chk_vortex_use_forked.isChecked()),
            vortex_force_zero=bool(self.chk_vortex_force_zero.isChecked()),
            vortex_aperture_radius_mm=float(
                self.dsb_vortex_aperture_mm.value()),
            vortex_steer_mode=self.cmb_vortex_steer_mode.currentText(),
            vortex_theta_x_deg=float(self.dsb_vortex_theta_x_deg.value()),
            vortex_theta_y_deg=float(self.dsb_vortex_theta_y_deg.value()),
            vortex_delta_x_mm=float(self.dsb_vortex_delta_x_mm.value()),
            vortex_delta_y_mm=float(self.dsb_vortex_delta_y_mm.value()),
            vortex_focal_length_m=float(
                self.dsb_vortex_focal_length_m.value()),
            size_mode=self.cmb_size_mode.currentText(),
            c2pi2unit=int(self.spin_c2pi2unit.value()),
            signal_2pi=int(self.spin_c2pi2unit.value()),
            method=self.cmb_method.currentText(),
            z=float(self.dsb_z.value()),
            M=float(self.dsb_M.value()),
            num_iter=int(self.spin_iter.value()),
            tilt_x_deg=float(self.dsb_tilt_x.value()),
            tilt_y_deg=float(self.dsb_tilt_y.value()),
            slot=int(self.spin_slot.value()),
            watchdog_interval_s=int(self.spin_interval.value()),
            watchdog_enabled=bool(self.chk_watchdog.isChecked()),
            show_gs_plot=bool(self.chk_plot.isChecked()),
            # New:
            wavelength_nm=float(self.dsb_wavelength_nm.value()),
            slm_nx=int(self.spin_slm_nx.value()),
            slm_ny=int(self.spin_slm_ny.value()),
            px_side_m=float(self.dsb_px_side_m.value()),
            py_side_m=float(self.dsb_py_side_m.value()),
            pixel_size_um=float(self.dsb_px_side_m.value()) * 1e6,
            fill_factor_percent=float(self.dsb_fill_factor.value()),
            temperature_threshold_c=float(self.temp_threshold_c),
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            self.append_log(f"Defaults saved to: {path}")
        except Exception as e:
            self.append_error(f"Failed to save defaults: {e}")

# ---------------------- Entrypoint ----------------------


def main():
    app = QtWidgets.QApplication(sys.argv)
    QtCore.QCoreApplication.setOrganizationName("STED")
    QtCore.QCoreApplication.setApplicationName("SLMControl")

    store = SlmParamsStore()
    params_window = SlmParamsWindow(store)
    vortex_window = VortexWindow(store, params_window)
    control_window = SlmControlWindow(store, params_window, vortex_window)
    vortex_window.set_slm_control(control_window)
    control_window.show()

    app.aboutToQuit.connect(control_window.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
