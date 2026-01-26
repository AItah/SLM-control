from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from slm_params import SlmParams, load_slm_params
from slm_store import SlmParamsStore


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SLM_PARAMS = ROOT / "vendor" / "LCOS_SLM_X15213.json"


class SlmParamsWindow(QtWidgets.QWidget):
    visibility_changed = QtCore.Signal(bool)

    def __init__(self, store: SlmParamsStore, parent=None) -> None:
        super().__init__(parent)
        self._store = store
        self._block_push = False
        self._force_close = False

        self.setWindowTitle("SLM Parameters")
        self.resize(520, 260)

        layout = QtWidgets.QVBoxLayout(self)

        # JSON loader
        top = QtWidgets.QHBoxLayout()
        self.ed_params_path = QtWidgets.QLineEdit(str(DEFAULT_SLM_PARAMS))
        self.btn_browse = QtWidgets.QPushButton("Browse...")
        self.btn_load = QtWidgets.QPushButton("Load")
        self.btn_browse.clicked.connect(self._browse_json)
        self.btn_load.clicked.connect(self._load_json_clicked)
        top.addWidget(QtWidgets.QLabel("SLM params JSON:"))
        top.addWidget(self.ed_params_path, 1)
        top.addWidget(self.btn_browse)
        top.addWidget(self.btn_load)
        layout.addLayout(top)

        grid = QtWidgets.QGridLayout()
        r = 0

        self.spin_nx = QtWidgets.QSpinBox()
        self.spin_nx.setRange(1, 10000)
        self.spin_ny = QtWidgets.QSpinBox()
        self.spin_ny.setRange(1, 10000)
        self.dsb_px_side_m = QtWidgets.QDoubleSpinBox()
        self.dsb_px_side_m.setRange(0.0, 1.0)
        self.dsb_px_side_m.setDecimals(9)
        self.dsb_px_side_m.setSingleStep(0.000001)
        self.dsb_px_side_m.setSuffix(" m")
        self.dsb_py_side_m = QtWidgets.QDoubleSpinBox()
        self.dsb_py_side_m.setRange(0.0, 1.0)
        self.dsb_py_side_m.setDecimals(9)
        self.dsb_py_side_m.setSingleStep(0.000001)
        self.dsb_py_side_m.setSuffix(" m")
        self.dsb_fill_factor = QtWidgets.QDoubleSpinBox()
        self.dsb_fill_factor.setRange(0.0, 100.0)
        self.dsb_fill_factor.setDecimals(2)
        self.dsb_fill_factor.setSuffix(" %")
        self.spin_c2pi2unit = QtWidgets.QSpinBox()
        self.spin_c2pi2unit.setRange(1, 1024)

        grid.addWidget(QtWidgets.QLabel("Nx (pixels):"), r, 0)
        grid.addWidget(self.spin_nx, r, 1)
        grid.addWidget(QtWidgets.QLabel("Ny (pixels):"), r, 2)
        grid.addWidget(self.spin_ny, r, 3)
        r += 1
        grid.addWidget(QtWidgets.QLabel("px_side_m:"), r, 0)
        grid.addWidget(self.dsb_px_side_m, r, 1)
        grid.addWidget(QtWidgets.QLabel("py_side_m:"), r, 2)
        grid.addWidget(self.dsb_py_side_m, r, 3)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Fill factor (%):"), r, 0)
        grid.addWidget(self.dsb_fill_factor, r, 1)
        grid.addWidget(QtWidgets.QLabel("c2pi2unit:"), r, 2)
        grid.addWidget(self.spin_c2pi2unit, r, 3)

        layout.addLayout(grid)

        self.lbl_status = QtWidgets.QLabel("")
        layout.addWidget(self.lbl_status)

        self._wire_value_changes()
        if not self._restore_settings():
            self._try_autoload_defaults()

    def _wire_value_changes(self) -> None:
        self.spin_nx.valueChanged.connect(self._push_params)
        self.spin_ny.valueChanged.connect(self._push_params)
        self.dsb_px_side_m.valueChanged.connect(self._push_params)
        self.dsb_py_side_m.valueChanged.connect(self._push_params)
        self.dsb_fill_factor.valueChanged.connect(self._push_params)
        self.spin_c2pi2unit.valueChanged.connect(self._push_params)

    def _restore_settings(self) -> bool:
        settings = QtCore.QSettings()
        settings.beginGroup("slm_params_window")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        stored_path = settings.value("params_path", "")
        if stored_path:
            self.ed_params_path.setText(stored_path)
            path = Path(stored_path)
            if path.exists():
                self._load_from_path(path, show_error=False)

        has_values = settings.contains("nx")
        if has_values:
            self._block_push = True
            try:
                self.spin_nx.setValue(int(settings.value("nx", 1)))
                self.spin_ny.setValue(int(settings.value("ny", 1)))
                self.dsb_px_side_m.setValue(float(settings.value("px_side_m", 0.0)))
                self.dsb_py_side_m.setValue(float(settings.value("py_side_m", 0.0)))
                self.dsb_fill_factor.setValue(float(settings.value("fill_factor", 0.0)))
                self.spin_c2pi2unit.setValue(int(settings.value("c2pi2unit", 1)))
            finally:
                self._block_push = False
            self._push_params()

        settings.endGroup()
        return bool(stored_path) or has_values

    def _save_settings(self) -> None:
        settings = QtCore.QSettings()
        settings.beginGroup("slm_params_window")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("params_path", self.ed_params_path.text().strip())
        settings.setValue("nx", int(self.spin_nx.value()))
        settings.setValue("ny", int(self.spin_ny.value()))
        settings.setValue("px_side_m", float(self.dsb_px_side_m.value()))
        settings.setValue("py_side_m", float(self.dsb_py_side_m.value()))
        settings.setValue("fill_factor", float(self.dsb_fill_factor.value()))
        settings.setValue("c2pi2unit", int(self.spin_c2pi2unit.value()))
        settings.setValue("visible", self.isVisible())
        settings.endGroup()

    def _try_autoload_defaults(self) -> None:
        if DEFAULT_SLM_PARAMS.exists():
            self._load_from_path(DEFAULT_SLM_PARAMS, show_error=False)

    def _browse_json(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select SLM params JSON", str(ROOT), "JSON (*.json)"
        )
        if path:
            self.ed_params_path.setText(path)

    def _load_json_clicked(self) -> None:
        path = self.ed_params_path.text().strip()
        if not path:
            return
        self._load_from_path(Path(path), show_error=True)

    def ensure_loaded(self) -> bool:
        path_text = self.ed_params_path.text().strip()
        if not path_text:
            return False
        path = Path(path_text)
        if not path.exists():
            return False
        self._load_from_path(path, show_error=False)
        return True

    def _load_from_path(self, path: Path, show_error: bool) -> None:
        try:
            params = load_slm_params(path)
        except Exception as exc:
            if show_error:
                QtWidgets.QMessageBox.warning(
                    self, "SLM params", f"Failed to load JSON: {exc}"
                )
            return

        self._block_push = True
        try:
            self.ed_params_path.setText(str(path))
            self.spin_nx.setValue(params.nx)
            self.spin_ny.setValue(params.ny)
            self.dsb_px_side_m.setValue(params.px_side_m)
            self.dsb_py_side_m.setValue(params.py_side_m)
            if params.fill_factor_percent is not None:
                self.dsb_fill_factor.setValue(params.fill_factor_percent)
            self.spin_c2pi2unit.setValue(params.c2pi2unit)
        finally:
            self._block_push = False

        self._push_params()
        self.lbl_status.setText(f"Loaded: {path}")

    def _push_params(self) -> None:
        if self._block_push:
            return

        path_text = self.ed_params_path.text().strip()
        source_path = Path(path_text) if path_text else DEFAULT_SLM_PARAMS

        params = SlmParams(
            nx=int(self.spin_nx.value()),
            ny=int(self.spin_ny.value()),
            px_side_m=float(self.dsb_px_side_m.value()),
            py_side_m=float(self.dsb_py_side_m.value()),
            fill_factor_percent=float(self.dsb_fill_factor.value()),
            c2pi2unit=int(self.spin_c2pi2unit.value()),
            source_path=source_path,
        )
        self._store.set_params(params)

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
            event.accept()
            return
        event.ignore()
        self.hide()

    def force_close(self) -> None:
        self._force_close = True
        self.close()
