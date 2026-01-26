from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from image_utils import exception_to_text
from slm_params import SlmParams
from slm_params_window import SlmParamsWindow
from slm_store import SlmParamsStore
from vortex_mask import BeamParams, SteeringRequest, generate_mask, load_calibration_mask

if TYPE_CHECKING:
    from slm_control_window import SlmControlWindow

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "out_mask"


class VortexWindow(QtWidgets.QWidget):
    visibility_changed = QtCore.Signal(bool)

    def __init__(
        self,
        store: SlmParamsStore,
        params_window: SlmParamsWindow,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._params_window = params_window
        self._slm_params: SlmParams | None = None
        self._slm_control: Optional["SlmControlWindow"] = None
        self._block_settings = False

        self.setWindowTitle("Vortex Generator")
        self.resize(620, 520)

        self._build_ui()
        self._store.changed.connect(self._on_params_changed)

        self._restore_settings()
        self._update_steer_mode()

    def set_slm_control(self, control: "SlmControlWindow") -> None:
        self._slm_control = control

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QHBoxLayout()
        self.lbl_params = QtWidgets.QLabel("SLM params: not loaded")
        info.addWidget(self.lbl_params)
        info.addStretch(1)
        layout.addLayout(info)

        # Vortex section
        vortex_group = QtWidgets.QGroupBox("Vortex")
        vortex_layout = QtWidgets.QGridLayout(vortex_group)
        r = 0

        self.dsb_wavelength_nm = QtWidgets.QDoubleSpinBox()
        self.dsb_wavelength_nm.setRange(200.0, 2000.0)
        self.dsb_wavelength_nm.setDecimals(3)
        self.dsb_wavelength_nm.setSuffix(" nm")
        self.dsb_wavelength_nm.setValue(775.0)

        self.spin_ell = QtWidgets.QSpinBox()
        self.spin_ell.setRange(1, 1000)
        self.spin_ell.setValue(1)

        self.dsb_offset_x_mm = QtWidgets.QDoubleSpinBox()
        self.dsb_offset_x_mm.setRange(-1000.0, 1000.0)
        self.dsb_offset_x_mm.setDecimals(3)
        self.dsb_offset_x_mm.setSuffix(" mm")

        self.dsb_offset_y_mm = QtWidgets.QDoubleSpinBox()
        self.dsb_offset_y_mm.setRange(-1000.0, 1000.0)
        self.dsb_offset_y_mm.setDecimals(3)
        self.dsb_offset_y_mm.setSuffix(" mm")

        self.dsb_aperture_mm = QtWidgets.QDoubleSpinBox()
        self.dsb_aperture_mm.setRange(0.0, 1000.0)
        self.dsb_aperture_mm.setDecimals(3)
        self.dsb_aperture_mm.setSuffix(" mm")

        self.chk_force_zero = QtWidgets.QCheckBox("Force pixel-zero center")
        self.chk_force_zero.setChecked(True)

        vortex_layout.addWidget(QtWidgets.QLabel("Wavelength:"), r, 0)
        vortex_layout.addWidget(self.dsb_wavelength_nm, r, 1)
        vortex_layout.addWidget(QtWidgets.QLabel("ell:"), r, 2)
        vortex_layout.addWidget(self.spin_ell, r, 3)
        r += 1

        vortex_layout.addWidget(QtWidgets.QLabel("Offset X:"), r, 0)
        vortex_layout.addWidget(self.dsb_offset_x_mm, r, 1)
        vortex_layout.addWidget(QtWidgets.QLabel("Offset Y:"), r, 2)
        vortex_layout.addWidget(self.dsb_offset_y_mm, r, 3)
        r += 1

        vortex_layout.addWidget(QtWidgets.QLabel("Aperture radius:"), r, 0)
        vortex_layout.addWidget(self.dsb_aperture_mm, r, 1)
        vortex_layout.addWidget(self.chk_force_zero, r, 2, 1, 2)

        layout.addWidget(vortex_group)

        # Calibration
        calib_group = QtWidgets.QGroupBox("Calibration")
        calib_layout = QtWidgets.QHBoxLayout(calib_group)
        self.ed_calib = QtWidgets.QLineEdit()
        self.btn_calib_browse = QtWidgets.QPushButton("Browse...")
        self.btn_calib_browse.clicked.connect(self._pick_calibration)
        calib_layout.addWidget(QtWidgets.QLabel("Calibration mask:"))
        calib_layout.addWidget(self.ed_calib, 1)
        calib_layout.addWidget(self.btn_calib_browse)
        layout.addWidget(calib_group)

        # Steering section
        steer_group = QtWidgets.QGroupBox("Steering")
        steer_layout = QtWidgets.QGridLayout(steer_group)
        r = 0

        self.chk_use_fork = QtWidgets.QCheckBox("Use fork grating")
        self.chk_use_fork.setChecked(True)
        self.chk_use_fork.stateChanged.connect(self._update_steer_mode)

        self.cmb_steer_mode = QtWidgets.QComboBox()
        self.cmb_steer_mode.addItems(["none", "angle", "shift"])
        self.cmb_steer_mode.currentTextChanged.connect(self._update_steer_mode)

        self.dsb_theta_x = QtWidgets.QDoubleSpinBox()
        self.dsb_theta_x.setRange(-89.0, 89.0)
        self.dsb_theta_x.setDecimals(4)
        self.dsb_theta_x.setSuffix(" deg")

        self.dsb_theta_y = QtWidgets.QDoubleSpinBox()
        self.dsb_theta_y.setRange(-89.0, 89.0)
        self.dsb_theta_y.setDecimals(4)
        self.dsb_theta_y.setSuffix(" deg")

        self.dsb_delta_x = QtWidgets.QDoubleSpinBox()
        self.dsb_delta_x.setRange(-1000.0, 1000.0)
        self.dsb_delta_x.setDecimals(4)
        self.dsb_delta_x.setSuffix(" mm")

        self.dsb_delta_y = QtWidgets.QDoubleSpinBox()
        self.dsb_delta_y.setRange(-1000.0, 1000.0)
        self.dsb_delta_y.setDecimals(4)
        self.dsb_delta_y.setSuffix(" mm")

        self.dsb_focal_m = QtWidgets.QDoubleSpinBox()
        self.dsb_focal_m.setRange(0.001, 10.0)
        self.dsb_focal_m.setDecimals(4)
        self.dsb_focal_m.setSuffix(" m")
        self.dsb_focal_m.setValue(0.2)

        steer_layout.addWidget(self.chk_use_fork, r, 0, 1, 2)
        steer_layout.addWidget(QtWidgets.QLabel("Steering mode:"), r, 2)
        steer_layout.addWidget(self.cmb_steer_mode, r, 3)
        r += 1

        steer_layout.addWidget(QtWidgets.QLabel("theta_x:"), r, 0)
        steer_layout.addWidget(self.dsb_theta_x, r, 1)
        steer_layout.addWidget(QtWidgets.QLabel("theta_y:"), r, 2)
        steer_layout.addWidget(self.dsb_theta_y, r, 3)
        r += 1

        steer_layout.addWidget(QtWidgets.QLabel("delta_x:"), r, 0)
        steer_layout.addWidget(self.dsb_delta_x, r, 1)
        steer_layout.addWidget(QtWidgets.QLabel("delta_y:"), r, 2)
        steer_layout.addWidget(self.dsb_delta_y, r, 3)
        r += 1

        steer_layout.addWidget(QtWidgets.QLabel("Focal length:"), r, 0)
        steer_layout.addWidget(self.dsb_focal_m, r, 1)

        layout.addWidget(steer_group)

        # Actions
        actions = QtWidgets.QHBoxLayout()
        self.spin_slot = QtWidgets.QSpinBox()
        self.spin_slot.setRange(-1, 15)
        self.spin_slot.setValue(-1)
        self.spin_slot.setToolTip("Slot 0-15 writes to SLM; -1 saves to file only.")
        self.btn_generate = QtWidgets.QPushButton("Generate Vortex Mask")
        self.btn_generate.clicked.connect(self._on_generate)
        actions.addWidget(QtWidgets.QLabel("Slot:"))
        actions.addWidget(self.spin_slot)
        actions.addStretch(1)
        actions.addWidget(self.btn_generate)
        layout.addLayout(actions)

        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(1000)
        self.log.setPlaceholderText("Vortex status messages appear here...")
        layout.addWidget(self.log, 1)

    def _on_params_changed(self, params: SlmParams) -> None:
        self._slm_params = params
        self.lbl_params.setText(
            f"SLM params: {params.nx} x {params.ny}, c2pi2unit {params.c2pi2unit}"
        )

    def _ensure_params(self) -> Optional[SlmParams]:
        if self._slm_params is None:
            self._params_window.ensure_loaded()
        if self._slm_params is None:
            self._append_error("Load SLM parameters first.")
            return None
        return self._slm_params

    def _pick_calibration(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select calibration mask", str(ROOT), "BMP (*.bmp)"
        )
        if path:
            self.ed_calib.setText(path)

    def _update_steer_mode(self) -> None:
        use_fork = self.chk_use_fork.isChecked()
        mode = self.cmb_steer_mode.currentText()
        angle = use_fork and mode == "angle"
        shift = use_fork and mode == "shift"

        self.cmb_steer_mode.setEnabled(use_fork)
        self.dsb_theta_x.setEnabled(angle)
        self.dsb_theta_y.setEnabled(angle)
        self.dsb_delta_x.setEnabled(shift)
        self.dsb_delta_y.setEnabled(shift)
        self.dsb_focal_m.setEnabled(shift)

    def _format_filename(
        self,
        use_forked: bool,
        ell: int,
        slm: SlmParams,
        steer_req: Optional[SteeringRequest],
        steer_result,
        offset_x_mm: float,
        offset_y_mm: float,
        focal_length_m: float,
    ) -> str:
        tag = "forked" if use_forked else "spiral"
        dx = steer_result.delta_x_mm if steer_result is not None else 0.0
        dy = steer_result.delta_y_mm if steer_result is not None else 0.0
        focal_mm = focal_length_m * 1e3

        return (
            f"slm_vortex_{tag}_ell_{ell}_{slm.nx}x{slm.ny}_"
            f"f_focus_{focal_mm:.0f}mm_dx_{dx:.3f}mm_dy_{dy:.3f}mm_"
            f"maskX_{offset_x_mm:.3f}mm_maskY_{offset_y_mm:.3f}mm.bmp"
        )

    def _on_generate(self) -> None:
        try:
            slm = self._ensure_params()
            if slm is None:
                return

            wavelength_nm = float(self.dsb_wavelength_nm.value())
            beam = BeamParams(lambda_m=wavelength_nm * 1e-9)

            ell = int(self.spin_ell.value())
            offset_x_mm = float(self.dsb_offset_x_mm.value())
            offset_y_mm = float(self.dsb_offset_y_mm.value())
            sft_x_m = offset_x_mm * 1e-3
            sft_y_m = offset_y_mm * 1e-3

            aperture_radius_m = None
            aperture_mm = float(self.dsb_aperture_mm.value())
            if aperture_mm > 0:
                aperture_radius_m = aperture_mm * 1e-3

            use_fork = self.chk_use_fork.isChecked()
            force_zero = self.chk_force_zero.isChecked()
            steer_req = None
            if use_fork:
                mode = self.cmb_steer_mode.currentText()
                if mode == "angle":
                    steer_req = SteeringRequest(
                        theta_x_deg=float(self.dsb_theta_x.value()),
                        theta_y_deg=float(self.dsb_theta_y.value()),
                        focal_length_m=float(self.dsb_focal_m.value()),
                    )
                elif mode == "shift":
                    steer_req = SteeringRequest(
                        delta_x_mm=float(self.dsb_delta_x.value()),
                        delta_y_mm=float(self.dsb_delta_y.value()),
                        focal_length_m=float(self.dsb_focal_m.value()),
                    )

            calib_mask = None
            calib_path = self.ed_calib.text().strip()
            if calib_path:
                calib_mask = load_calibration_mask(calib_path)

            result = generate_mask(
                slm=slm,
                beam=beam,
                ell=ell,
                sft_x_m=sft_x_m,
                sft_y_m=sft_y_m,
                steer_req=steer_req,
                use_forked=use_fork,
                aperture_radius_m=aperture_radius_m,
                force_zero=force_zero,
                calib_mask=calib_mask,
            )

            filename = self._format_filename(
                use_forked=use_fork,
                ell=ell,
                slm=slm,
                steer_req=steer_req,
                steer_result=result.steer,
                offset_x_mm=offset_x_mm,
                offset_y_mm=offset_y_mm,
                focal_length_m=float(self.dsb_focal_m.value()),
            )

            slot = int(self.spin_slot.value())
            if 0 <= slot <= 15:
                if self._slm_control is None:
                    self._append_error("SLM control window not available.")
                    return
                sent = self._slm_control.send_mask_to_slot(result.mask_u8, slot)
                if sent:
                    self._append_log(f"Mask sent to SLM slot {slot}.")
                return

            if slot != -1:
                self._append_error("Slot must be -1 or between 0 and 15.")
                return

            DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
            suggested = DEFAULT_OUT_DIR / filename
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save vortex mask", str(suggested), "BMP (*.bmp)"
            )
            if not path:
                return
            save_path = Path(path)
            if save_path.suffix.lower() != ".bmp":
                save_path = save_path.with_suffix(".bmp")
            Image.fromarray(result.mask_u8, mode="L").save(save_path)
            self._append_log(f"Vortex mask saved: {save_path}")
        except Exception as exc:
            self._append_error(exception_to_text(exc))

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _append_error(self, text: str) -> None:
        self.log.appendPlainText("ERROR: " + text)

    def _restore_settings(self) -> None:
        settings = QtCore.QSettings()
        settings.beginGroup("vortex_window")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        self.ed_calib.setText(settings.value("calib_path", ""))
        self.dsb_wavelength_nm.setValue(float(settings.value("wavelength_nm", 775.0)))
        self.spin_ell.setValue(int(settings.value("ell", 1)))
        self.dsb_offset_x_mm.setValue(float(settings.value("offset_x_mm", 0.0)))
        self.dsb_offset_y_mm.setValue(float(settings.value("offset_y_mm", 0.0)))
        self.dsb_aperture_mm.setValue(float(settings.value("aperture_mm", 0.0)))
        self.chk_force_zero.setChecked(bool(settings.value("force_zero", True, bool)))
        self.chk_use_fork.setChecked(bool(settings.value("use_fork", True, bool)))
        self.cmb_steer_mode.setCurrentText(settings.value("steer_mode", "shift"))
        self.dsb_theta_x.setValue(float(settings.value("theta_x", 0.0)))
        self.dsb_theta_y.setValue(float(settings.value("theta_y", 0.0)))
        self.dsb_delta_x.setValue(float(settings.value("delta_x", -0.3)))
        self.dsb_delta_y.setValue(float(settings.value("delta_y", 0.0)))
        self.dsb_focal_m.setValue(float(settings.value("focal_m", 0.2)))
        self.spin_slot.setValue(int(settings.value("slot", -1)))
        settings.endGroup()

    def _save_settings(self) -> None:
        settings = QtCore.QSettings()
        settings.beginGroup("vortex_window")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("calib_path", self.ed_calib.text().strip())
        settings.setValue("wavelength_nm", float(self.dsb_wavelength_nm.value()))
        settings.setValue("ell", int(self.spin_ell.value()))
        settings.setValue("offset_x_mm", float(self.dsb_offset_x_mm.value()))
        settings.setValue("offset_y_mm", float(self.dsb_offset_y_mm.value()))
        settings.setValue("aperture_mm", float(self.dsb_aperture_mm.value()))
        settings.setValue("force_zero", self.chk_force_zero.isChecked())
        settings.setValue("use_fork", self.chk_use_fork.isChecked())
        settings.setValue("steer_mode", self.cmb_steer_mode.currentText())
        settings.setValue("theta_x", float(self.dsb_theta_x.value()))
        settings.setValue("theta_y", float(self.dsb_theta_y.value()))
        settings.setValue("delta_x", float(self.dsb_delta_x.value()))
        settings.setValue("delta_y", float(self.dsb_delta_y.value()))
        settings.setValue("focal_m", float(self.dsb_focal_m.value()))
        settings.setValue("slot", int(self.spin_slot.value()))
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
        if QtCore.QCoreApplication.closingDown():
            event.accept()
            return
        event.ignore()
        self.hide()
