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
    from donut_optimization_window import DonutOptimizationWindow

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
        self._camera_window = None
        self._block_settings = False
        self._force_close = False
        self._donut_window: Optional["DonutOptimizationWindow"] = None
        self._pending_donut_visible = False

        self.setWindowTitle("Vortex Generator")
        self.resize(620, 520)

        self._build_ui()
        self._store.changed.connect(self._on_params_changed)

        self._restore_settings()
        self._update_steer_mode()

    def set_slm_control(self, control: "SlmControlWindow") -> None:
        self._slm_control = control
        self._maybe_restore_donut_window()

    def set_camera_window(self, camera_window) -> None:
        self._camera_window = camera_window
        self._maybe_restore_donut_window()

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
        self.chk_force_zero.setToolTip("Force the vortex center to land on an integer pixel.")

        self.dsb_phase_offset_deg = QtWidgets.QDoubleSpinBox()
        self.dsb_phase_offset_deg.setRange(-360.0, 360.0)
        self.dsb_phase_offset_deg.setDecimals(2)
        self.dsb_phase_offset_deg.setSuffix(" deg")
        self.dsb_phase_offset_deg.setValue(0.0)
        self.dsb_phase_offset_deg.setToolTip("Phase offset of the vortex (degrees).")
        self.dsb_axis_rotation_deg = QtWidgets.QDoubleSpinBox()
        self.dsb_axis_rotation_deg.setRange(-360.0, 360.0)
        self.dsb_axis_rotation_deg.setDecimals(2)
        self.dsb_axis_rotation_deg.setSuffix(" deg")
        self.dsb_axis_rotation_deg.setValue(0.0)
        self.dsb_axis_rotation_deg.setToolTip("Rotate vortex axes (degrees).")
        self.dsb_alpha = QtWidgets.QDoubleSpinBox()
        self.dsb_alpha.setRange(0.1, 10.0)
        self.dsb_alpha.setDecimals(2)
        self.dsb_alpha.setValue(1.0)
        self.dsb_alpha.setToolTip("X-axis scaling (alpha).")
        self.dsb_beta = QtWidgets.QDoubleSpinBox()
        self.dsb_beta.setRange(0.1, 10.0)
        self.dsb_beta.setDecimals(2)
        self.dsb_beta.setValue(1.0)
        self.dsb_beta.setToolTip("Y-axis scaling (beta).")

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
        r += 1
        vortex_layout.addWidget(QtWidgets.QLabel("Phase offset:"), r, 0)
        vortex_layout.addWidget(self.dsb_phase_offset_deg, r, 1)
        vortex_layout.addWidget(QtWidgets.QLabel("Rotation:"), r, 2)
        vortex_layout.addWidget(self.dsb_axis_rotation_deg, r, 3)
        r += 1
        vortex_layout.addWidget(QtWidgets.QLabel("Alpha:"), r, 0)
        vortex_layout.addWidget(self.dsb_alpha, r, 1)
        vortex_layout.addWidget(QtWidgets.QLabel("Beta:"), r, 2)
        vortex_layout.addWidget(self.dsb_beta, r, 3)

        layout.addWidget(vortex_group)

        # Calibration
        calib_group = QtWidgets.QGroupBox("Calibration")
        calib_layout = QtWidgets.QHBoxLayout(calib_group)
        self.ed_calib = QtWidgets.QLineEdit()
        self.btn_calib_browse = QtWidgets.QPushButton("Browse...")
        self.btn_calib_browse.setToolTip("Select a calibration mask file.")
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
        self.chk_use_fork.setToolTip("Enable fork grating for beam steering.")
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

        # Zernike section
        zern_group = QtWidgets.QGroupBox("Zernike Corrections")
        zern_layout = QtWidgets.QGridLayout(zern_group)
        r = 0
        self.chk_use_zernike = QtWidgets.QCheckBox("Enable Zernike corrections")
        self.chk_use_zernike.setChecked(False)
        self.chk_use_zernike.setToolTip("Enable Zernike aberration corrections.")
        zern_layout.addWidget(self.chk_use_zernike, r, 0, 1, 2)
        r += 1

        self.dsb_astig_v = QtWidgets.QDoubleSpinBox()
        self.dsb_astig_v.setRange(-10.0, 10.0)
        self.dsb_astig_v.setDecimals(4)
        self.dsb_astig_v.setSingleStep(0.01)

        self.dsb_astig_o = QtWidgets.QDoubleSpinBox()
        self.dsb_astig_o.setRange(-10.0, 10.0)
        self.dsb_astig_o.setDecimals(4)
        self.dsb_astig_o.setSingleStep(0.01)

        self.dsb_coma_y = QtWidgets.QDoubleSpinBox()
        self.dsb_coma_y.setRange(-10.0, 10.0)
        self.dsb_coma_y.setDecimals(4)
        self.dsb_coma_y.setSingleStep(0.01)

        self.dsb_coma_x = QtWidgets.QDoubleSpinBox()
        self.dsb_coma_x.setRange(-10.0, 10.0)
        self.dsb_coma_x.setDecimals(4)
        self.dsb_coma_x.setSingleStep(0.01)

        self.dsb_spher = QtWidgets.QDoubleSpinBox()
        self.dsb_spher.setRange(-10.0, 10.0)
        self.dsb_spher.setDecimals(4)
        self.dsb_spher.setSingleStep(0.01)

        zern_layout.addWidget(QtWidgets.QLabel("Astig V (c_astig_v):"), r, 0)
        zern_layout.addWidget(self.dsb_astig_v, r, 1)
        zern_layout.addWidget(QtWidgets.QLabel("Astig O (c_astig_o):"), r, 2)
        zern_layout.addWidget(self.dsb_astig_o, r, 3)
        r += 1
        zern_layout.addWidget(QtWidgets.QLabel("Coma Y (c_coma_y):"), r, 0)
        zern_layout.addWidget(self.dsb_coma_y, r, 1)
        zern_layout.addWidget(QtWidgets.QLabel("Coma X (c_coma_x):"), r, 2)
        zern_layout.addWidget(self.dsb_coma_x, r, 3)
        r += 1
        zern_layout.addWidget(QtWidgets.QLabel("Spherical (c_spher):"), r, 0)
        zern_layout.addWidget(self.dsb_spher, r, 1)

        layout.addWidget(zern_group)

        # Actions
        actions = QtWidgets.QHBoxLayout()
        self.spin_slot = QtWidgets.QSpinBox()
        self.spin_slot.setRange(-1, 15)
        self.spin_slot.setValue(-1)
        self.spin_slot.setToolTip("Slot 0-15 writes to SLM; -1 saves to file only.")
        self.btn_generate = QtWidgets.QPushButton("Generate Vortex Mask")
        self.btn_generate.setToolTip("Generate the vortex mask and save/send it.")
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

        self.btn_optimize = QtWidgets.QPushButton("Donut Optimization...")
        self.btn_optimize.setToolTip("Open the Donut Optimization Wizard.")
        self.btn_optimize.clicked.connect(self._open_donut_opt)
        layout.addWidget(self.btn_optimize, 0)

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
            result, filename = self._build_mask_from_ui()
            filename = self._append_zernike_to_filename(filename)

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

    def build_mask(self, offset_x_mm: float, offset_y_mm: float) -> np.ndarray:
        result, _ = self._build_mask_from_ui(offset_x_mm, offset_y_mm)
        return result.mask_u8

    def get_offsets_mm(self) -> tuple[float, float]:
        return float(self.dsb_offset_x_mm.value()), float(self.dsb_offset_y_mm.value())

    def set_offsets_mm(self, x_mm: float, y_mm: float) -> None:
        self.dsb_offset_x_mm.setValue(float(x_mm))
        self.dsb_offset_y_mm.setValue(float(y_mm))

    def get_zernike_values(self) -> tuple[float, float, float, float, float]:
        return (
            float(self.dsb_astig_v.value()),
            float(self.dsb_astig_o.value()),
            float(self.dsb_coma_x.value()),
            float(self.dsb_coma_y.value()),
            float(self.dsb_spher.value()),
        )

    def get_shape_params(self) -> tuple[float, float, float, float]:
        return (
            float(self.dsb_axis_rotation_deg.value()),
            float(self.dsb_phase_offset_deg.value()),
            float(self.dsb_alpha.value()),
            float(self.dsb_beta.value()),
        )

    def set_zernike_values(
        self,
        astig_v: Optional[float] = None,
        astig_o: Optional[float] = None,
        coma_x: Optional[float] = None,
        coma_y: Optional[float] = None,
        spher: Optional[float] = None,
    ) -> None:
        if astig_v is not None:
            self.dsb_astig_v.setValue(float(astig_v))
        if astig_o is not None:
            self.dsb_astig_o.setValue(float(astig_o))
        if coma_x is not None:
            self.dsb_coma_x.setValue(float(coma_x))
        if coma_y is not None:
            self.dsb_coma_y.setValue(float(coma_y))
        if spher is not None:
            self.dsb_spher.setValue(float(spher))

    def set_shape_params(
        self,
        axis_rotation_deg: Optional[float] = None,
        phase_offset_deg: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        if axis_rotation_deg is not None:
            self.dsb_axis_rotation_deg.setValue(float(axis_rotation_deg))
        if phase_offset_deg is not None:
            self.dsb_phase_offset_deg.setValue(float(phase_offset_deg))
        if alpha is not None:
            self.dsb_alpha.setValue(float(alpha))
        if beta is not None:
            self.dsb_beta.setValue(float(beta))

    def build_mask_with_params(
        self,
        offset_x_mm: Optional[float] = None,
        offset_y_mm: Optional[float] = None,
        phase_offset_deg: Optional[float] = None,
        axis_rotation_deg: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        c_astig_v: Optional[float] = None,
        c_astig_o: Optional[float] = None,
        c_coma_y: Optional[float] = None,
        c_coma_x: Optional[float] = None,
        c_spher: Optional[float] = None,
    ) -> np.ndarray:
        result, _ = self._build_mask_from_ui(
            offset_x_mm=offset_x_mm,
            offset_y_mm=offset_y_mm,
            phase_offset_deg=phase_offset_deg,
            axis_rotation_deg=axis_rotation_deg,
            alpha=alpha,
            beta=beta,
            c_astig_v=c_astig_v,
            c_astig_o=c_astig_o,
            c_coma_y=c_coma_y,
            c_coma_x=c_coma_x,
            c_spher=c_spher,
        )
        return result.mask_u8

    def _build_mask_from_ui(
        self,
        offset_x_mm: Optional[float] = None,
        offset_y_mm: Optional[float] = None,
        phase_offset_deg: Optional[float] = None,
        axis_rotation_deg: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        c_astig_v: Optional[float] = None,
        c_astig_o: Optional[float] = None,
        c_coma_y: Optional[float] = None,
        c_coma_x: Optional[float] = None,
        c_spher: Optional[float] = None,
    ):
        slm = self._ensure_params()
        if slm is None:
            raise RuntimeError("SLM parameters not loaded.")

        wavelength_nm = float(self.dsb_wavelength_nm.value())
        beam = BeamParams(lambda_m=wavelength_nm * 1e-9)

        ell = int(self.spin_ell.value())
        off_x = (
            float(self.dsb_offset_x_mm.value())
            if offset_x_mm is None
            else float(offset_x_mm)
        )
        off_y = (
            float(self.dsb_offset_y_mm.value())
            if offset_y_mm is None
            else float(offset_y_mm)
        )
        sft_x_m = off_x * 1e-3
        sft_y_m = off_y * 1e-3
        phase_offset_deg = (
            float(self.dsb_phase_offset_deg.value())
            if phase_offset_deg is None
            else float(phase_offset_deg)
        )
        axis_rotation_deg = (
            float(self.dsb_axis_rotation_deg.value())
            if axis_rotation_deg is None
            else float(axis_rotation_deg)
        )
        alpha_val = (
            float(self.dsb_alpha.value()) if alpha is None else float(alpha)
        )
        beta_val = (
            float(self.dsb_beta.value()) if beta is None else float(beta)
        )

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
            axis_rotation_deg=axis_rotation_deg,
            phase_offset_deg=phase_offset_deg,
            alpha=alpha_val,
            beta=beta_val,
            zernike_offset_x_m=sft_x_m,
            zernike_offset_y_m=sft_y_m,
            c_astig_v=float(self.dsb_astig_v.value())
            if c_astig_v is None
            else float(c_astig_v),
            c_astig_o=float(self.dsb_astig_o.value())
            if c_astig_o is None
            else float(c_astig_o),
            c_coma_y=float(self.dsb_coma_y.value())
            if c_coma_y is None
            else float(c_coma_y),
            c_coma_x=float(self.dsb_coma_x.value())
            if c_coma_x is None
            else float(c_coma_x),
            c_spher=float(self.dsb_spher.value())
            if c_spher is None
            else float(c_spher),
            use_zernike=self.chk_use_zernike.isChecked(),
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
            offset_x_mm=off_x,
            offset_y_mm=off_y,
            focal_length_m=float(self.dsb_focal_m.value()),
        )
        return result, filename

    def _append_zernike_to_filename(self, filename: str) -> str:
        if not self.chk_use_zernike.isChecked():
            return filename
        return (
            filename[:-4]
            + f"_zern_av_{self.dsb_astig_v.value():.3f}"
            + f"_ao_{self.dsb_astig_o.value():.3f}"
            + f"_cx_{self.dsb_coma_x.value():.3f}"
            + f"_cy_{self.dsb_coma_y.value():.3f}"
            + f"_s_{self.dsb_spher.value():.3f}.bmp"
        )

    def _open_donut_opt(self) -> None:
        if self._slm_control is None or self._camera_window is None:
            self._append_error("Camera and SLM control must be available.")
            return
        if self._donut_window is None:
            from donut_optimization_window import DonutOptimizationWindow
            self._donut_window = DonutOptimizationWindow(
                self, self._slm_control, self._camera_window, self
            )
        self._donut_window.show()
        self._donut_window.raise_()
        self._donut_window.activateWindow()

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
        if settings.contains("phase_offset_deg"):
            self.dsb_phase_offset_deg.setValue(
                float(settings.value("phase_offset_deg", 0.0))
            )
        else:
            self.dsb_phase_offset_deg.setValue(float(settings.value("rotation_deg", 0.0)))
        self.dsb_axis_rotation_deg.setValue(
            float(settings.value("axis_rotation_deg", 0.0))
        )
        self.dsb_alpha.setValue(float(settings.value("alpha", 1.0)))
        self.dsb_beta.setValue(float(settings.value("beta", 1.0)))
        self.chk_force_zero.setChecked(bool(settings.value("force_zero", True, bool)))
        self.chk_use_fork.setChecked(bool(settings.value("use_fork", True, bool)))
        self.cmb_steer_mode.setCurrentText(settings.value("steer_mode", "shift"))
        self.dsb_theta_x.setValue(float(settings.value("theta_x", 0.0)))
        self.dsb_theta_y.setValue(float(settings.value("theta_y", 0.0)))
        self.dsb_delta_x.setValue(float(settings.value("delta_x", -0.3)))
        self.dsb_delta_y.setValue(float(settings.value("delta_y", 0.0)))
        self.dsb_focal_m.setValue(float(settings.value("focal_m", 0.2)))
        self.spin_slot.setValue(int(settings.value("slot", -1)))
        self.chk_use_zernike.setChecked(bool(settings.value("use_zernike", False, bool)))
        self.dsb_astig_v.setValue(float(settings.value("astig_v", 0.0)))
        self.dsb_astig_o.setValue(float(settings.value("astig_o", 0.0)))
        self.dsb_coma_y.setValue(float(settings.value("coma_y", 0.0)))
        self.dsb_coma_x.setValue(float(settings.value("coma_x", 0.0)))
        self.dsb_spher.setValue(float(settings.value("spher", 0.0)))
        settings.endGroup()
        settings.beginGroup("donut_opt_window")
        self._pending_donut_visible = bool(settings.value("visible", False, bool))
        settings.endGroup()
        self._maybe_restore_donut_window()

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
        settings.setValue("phase_offset_deg", float(self.dsb_phase_offset_deg.value()))
        settings.setValue(
            "axis_rotation_deg", float(self.dsb_axis_rotation_deg.value())
        )
        settings.setValue("alpha", float(self.dsb_alpha.value()))
        settings.setValue("beta", float(self.dsb_beta.value()))
        settings.setValue("force_zero", self.chk_force_zero.isChecked())
        settings.setValue("use_fork", self.chk_use_fork.isChecked())
        settings.setValue("steer_mode", self.cmb_steer_mode.currentText())
        settings.setValue("theta_x", float(self.dsb_theta_x.value()))
        settings.setValue("theta_y", float(self.dsb_theta_y.value()))
        settings.setValue("delta_x", float(self.dsb_delta_x.value()))
        settings.setValue("delta_y", float(self.dsb_delta_y.value()))
        settings.setValue("focal_m", float(self.dsb_focal_m.value()))
        settings.setValue("slot", int(self.spin_slot.value()))
        settings.setValue("use_zernike", self.chk_use_zernike.isChecked())
        settings.setValue("astig_v", float(self.dsb_astig_v.value()))
        settings.setValue("astig_o", float(self.dsb_astig_o.value()))
        settings.setValue("coma_y", float(self.dsb_coma_y.value()))
        settings.setValue("coma_x", float(self.dsb_coma_x.value()))
        settings.setValue("spher", float(self.dsb_spher.value()))
        settings.setValue("visible", self.isVisible())
        settings.endGroup()

    def _maybe_restore_donut_window(self) -> None:
        if not self._pending_donut_visible:
            return
        if self._slm_control is None or self._camera_window is None:
            return
        self._pending_donut_visible = False
        self._open_donut_opt()

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
