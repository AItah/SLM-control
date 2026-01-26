from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class SlmParams:
    nx: int
    ny: int
    px_m: float
    py_m: float
    c2pi2unit: int


@dataclass
class BeamParams:
    lambda_m: float


@dataclass
class Coordinates:
    nx: int
    ny: int
    px_m: float
    py_m: float
    xi: np.ndarray
    yi: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    center_mode: str


@dataclass
class SteeringRequest:
    theta_x_deg: float = 0.0
    theta_y_deg: float = 0.0
    delta_x_mm: float = 0.0
    delta_y_mm: float = 0.0
    focal_length_m: float = 0.2


@dataclass
class SteeringResult:
    mode: str
    theta_x_rad: float
    theta_y_rad: float
    fcp_x: float
    fcp_y: float
    delta_x_mm: float
    delta_y_mm: float
    clamped: bool
    scale: float


@dataclass
class MaskResult:
    phi_wrapped: np.ndarray
    mask_u8: np.ndarray
    steer: Optional[SteeringResult]
    filename: str


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_slm_params(path: str | Path) -> SlmParams:
    data = load_json(path)
    return SlmParams(
        nx=int(data["Nx"]),
        ny=int(data["Ny"]),
        px_m=float(data["px_side_m"]),
        py_m=float(data["py_side_m"]),
        c2pi2unit=int(data["c2pi2unit"]),
    )


def load_beam_params(path: str | Path) -> BeamParams:
    data = load_json(path)
    return BeamParams(lambda_m=float(data["lambda_m"]))


def load_calibration_mask(path: str | Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode not in ("L", "P"):
        raise ValueError(f"Calibration mask must be 8-bit grayscale. Mode: {img.mode}")
    data = np.array(img)
    if data.ndim != 2:
        raise ValueError(f"Calibration mask must be 2D. Got shape: {data.shape}")
    return data.astype(np.uint8)


def make_coordinates(
    nx: int,
    ny: int,
    px_m: float,
    py_m: float,
    force_zero: bool = False,
) -> Coordinates:
    if force_zero:
        x_pix = np.arange(nx) - (nx / 2 if nx % 2 == 0 else (nx - 1) / 2)
        y_pix = np.arange(ny) - (ny / 2 if ny % 2 == 0 else (ny - 1) / 2)
        center_mode = "pixel_zero"
    else:
        x_pix = np.arange(nx) - (nx - 1) / 2
        y_pix = np.arange(ny) - (ny - 1) / 2
        center_mode = "symmetric"

    xi, yi = np.meshgrid(x_pix, y_pix, indexing="xy")
    x_m = x_pix * px_m
    y_m = y_pix * py_m
    X = xi * px_m
    Y = yi * py_m

    return Coordinates(
        nx=nx,
        ny=ny,
        px_m=px_m,
        py_m=py_m,
        xi=xi,
        yi=yi,
        X=X,
        Y=Y,
        x_m=x_m,
        y_m=y_m,
        x_mm=x_m * 1e3,
        y_mm=y_m * 1e3,
        center_mode=center_mode,
    )


def make_vortex_phase(coords: Coordinates, ell: int, sft_x: float = 0.0, sft_y: float = 0.0) -> np.ndarray:
    if not isinstance(ell, int) or ell < 1:
        raise ValueError("ell must be a positive integer.")
    theta = np.arctan2(coords.Y - sft_y, coords.X - sft_x)
    return ell * theta


def make_shift_phase(coords: Coordinates, fcp_x: float, fcp_y: float) -> np.ndarray:
    return 2 * np.pi * (fcp_x * coords.xi + fcp_y * coords.yi)


def make_circular_aperture(coords: Coordinates, radius_m: float) -> np.ndarray:
    r = np.hypot(coords.X, coords.Y)
    return (r <= radius_m).astype(np.float64)


def theta_from_shift(delta_x_m: float, delta_y_m: float, f_m: float) -> tuple[float, float]:
    if f_m <= 0:
        raise ValueError("f_m must be > 0.")
    return np.arctan(delta_x_m / f_m), np.arctan(delta_y_m / f_m)


def fcp_from_theta(theta_x_rad: float, theta_y_rad: float, slm: SlmParams, lambda_m: float) -> tuple[float, float]:
    if lambda_m <= 0:
        raise ValueError("lambda_m must be > 0.")
    return theta_x_rad * slm.px_m / lambda_m, theta_y_rad * slm.py_m / lambda_m


def clamp_fcp_nyquist(fcp_x: float, fcp_y: float, limit: float = 0.5) -> tuple[float, float, bool, float]:
    mag = np.hypot(fcp_x, fcp_y)
    if mag > limit:
        scale = limit / mag
        return fcp_x * scale, fcp_y * scale, True, scale
    return fcp_x, fcp_y, False, 1.0


def resolve_steering(req: SteeringRequest, slm: SlmParams, beam: BeamParams) -> SteeringResult:
    if req.theta_x_deg or req.theta_y_deg:
        mode = "angle"
        theta_x = np.deg2rad(req.theta_x_deg)
        theta_y = np.deg2rad(req.theta_y_deg)
    elif req.delta_x_mm or req.delta_y_mm:
        mode = "shift"
        dx_m = req.delta_x_mm * 1e-3
        dy_m = req.delta_y_mm * 1e-3
        theta_x, theta_y = theta_from_shift(dx_m, dy_m, req.focal_length_m)
    else:
        mode = "angle"
        theta_x, theta_y = 0.0, 0.0

    fcp_x, fcp_y = fcp_from_theta(theta_x, theta_y, slm, beam.lambda_m)
    fcp_x, fcp_y, clamped, scale = clamp_fcp_nyquist(fcp_x, fcp_y)

    delta_x_mm = 1e3 * req.focal_length_m * beam.lambda_m * fcp_x / slm.px_m
    delta_y_mm = 1e3 * req.focal_length_m * beam.lambda_m * fcp_y / slm.py_m

    return SteeringResult(
        mode=mode,
        theta_x_rad=theta_x,
        theta_y_rad=theta_y,
        fcp_x=fcp_x,
        fcp_y=fcp_y,
        delta_x_mm=delta_x_mm,
        delta_y_mm=delta_y_mm,
        clamped=clamped,
        scale=scale,
    )


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.mod(phase, 2 * np.pi)


def phase_to_gray(phi_wrapped: np.ndarray, c2pi2unit: int) -> np.ndarray:
    phase_gray = c2pi2unit * (phi_wrapped / (2 * np.pi))
    phase_gray = np.floor(phase_gray + 0.5)
    return np.minimum(c2pi2unit, phase_gray).astype(np.uint8)


def apply_calibration(mask_u8: np.ndarray, calib_mask: np.ndarray, c2pi2unit: int) -> np.ndarray:
    combined = (mask_u8.astype(np.uint16) + calib_mask.astype(np.uint16)) % c2pi2unit
    return combined.astype(np.uint8)


def format_filename(
    use_forked: bool,
    ell: int,
    slm: SlmParams,
    sft_x: float,
    sft_y: float,
    steer_req: Optional[SteeringRequest],
    steer: Optional[SteeringResult],
) -> str:
    tag = "forked" if use_forked else "spiral"

    shift_tag_mm = 0.0
    if steer_req:
        if steer_req.theta_x_deg or steer_req.theta_y_deg:
            shift_tag_mm = steer.delta_x_mm if steer else 0.0
        elif steer_req.delta_x_mm or steer_req.delta_y_mm:
            shift_tag_mm = steer_req.delta_x_mm
        elif steer:
            shift_tag_mm = steer.delta_x_mm
    elif steer:
        shift_tag_mm = steer.delta_x_mm

    return (
        f"slm_vortex_{tag}_ell_{ell}_{slm.nx}x{slm.ny}_"
        f"{shift_tag_mm:.3f}mm_sft_x_{sft_x*1e3:.3f}mm_sft_y_{sft_y*1e3:.3f}mm.bmp"
    )


def generate_mask(
    slm: SlmParams,
    beam: BeamParams,
    ell: int,
    sft_x: float = 0.0,
    sft_y: float = 0.0,
    steer_req: Optional[SteeringRequest] = None,
    use_forked: bool = True,
    aperture_radius_m: Optional[float] = None,
    force_zero: bool = True,
    calib_mask: Optional[np.ndarray] = None,
) -> MaskResult:
    coords = make_coordinates(slm.nx, slm.ny, slm.px_m, slm.py_m, force_zero=force_zero)

    vortex = make_vortex_phase(coords, ell, sft_x=sft_x, sft_y=sft_y)
    if aperture_radius_m is not None:
        vortex = vortex * make_circular_aperture(coords, aperture_radius_m)

    steer = None
    if use_forked:
        steer_req = steer_req or SteeringRequest()
        steer = resolve_steering(steer_req, slm, beam)
        shift = make_shift_phase(coords, steer.fcp_x, steer.fcp_y)
    else:
        shift = np.zeros_like(vortex)

    desired = vortex + shift
    phi_wrapped = wrap_phase(desired)
    mask_u8 = phase_to_gray(phi_wrapped, slm.c2pi2unit)

    if calib_mask is not None:
        mask_u8 = apply_calibration(mask_u8, calib_mask, slm.c2pi2unit)

    filename = format_filename(use_forked, ell, slm, sft_x, sft_y, steer_req, steer)
    return MaskResult(phi_wrapped=phi_wrapped, mask_u8=mask_u8, steer=steer, filename=filename)


def save_mask(path: str | Path, mask_u8: np.ndarray) -> None:
    path = Path(path)
    Image.fromarray(mask_u8, mode="L").save(path)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    slm_path = base_dir / "vendor/LCOS_SLM_X15213.json"
    beam_path = base_dir / "GaussianBeam.json"
    calib_path = base_dir / "vendor" / "Correction_patterns" / "CAL_LSH0805598_770nm.bmp"


    if not slm_path.exists():
        raise FileNotFoundError(f"Missing SLM file: {slm_path}")
    if not beam_path.exists():
        raise FileNotFoundError(f"Missing beam file: {beam_path}")

    slm = load_slm_params(slm_path)
    beam = load_beam_params(beam_path)

    steer_req = SteeringRequest(
        delta_x_mm=-0.3,
        delta_y_mm=0.0,
        focal_length_m=0.2,
    )

    calib_mask = load_calibration_mask(calib_path) if calib_path.exists() else None
    result = generate_mask(
        slm=slm,
        beam=beam,
        ell=1,
        sft_x=0.3e-3,
        sft_y=-0.55e-3,
        steer_req=steer_req,
        use_forked=True,
        calib_mask=calib_mask,
    )

    output_dir = Path("masks_out")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / result.filename
    save_mask(output_path, result.mask_u8)

    if result.steer:
        print(
            "Steer mode: {mode} | dx={dx:.3f} mm, dy={dy:.3f} mm | clamped={clamped}".format(
                mode=result.steer.mode,
                dx=result.steer.delta_x_mm,
                dy=result.steer.delta_y_mm,
                clamped=result.steer.clamped,
            )
        )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

