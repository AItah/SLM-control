from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from slm_params import SlmParams


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
    r_norm: np.ndarray
    x_norm: np.ndarray
    y_norm: np.ndarray


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


def load_calibration_mask(path: str | Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode not in ("L", "P"):
        raise ValueError(f"Calibration mask must be 8-bit grayscale. Mode: {img.mode}")
    data = np.array(img)
    if data.ndim != 2:
        raise ValueError(f"Calibration mask must be 2D. Got shape: {data.shape}")
    return data.astype(np.uint8)


def make_coordinates(
    slm: SlmParams,
    force_zero: bool = False,
    aperture_radius_m: Optional[float] = None,
) -> Coordinates:
    nx, ny = slm.nx, slm.ny
    if force_zero:
        x_pix = np.arange(nx) - (nx / 2 if nx % 2 == 0 else (nx - 1) / 2)
        y_pix = np.arange(ny) - (ny / 2 if ny % 2 == 0 else (ny - 1) / 2)
        center_mode = "pixel_zero"
    else:
        x_pix = np.arange(nx) - (nx - 1) / 2
        y_pix = np.arange(ny) - (ny - 1) / 2
        center_mode = "symmetric"

    xi, yi = np.meshgrid(x_pix, y_pix, indexing="xy")
    x_m = x_pix * slm.px_side_m
    y_m = y_pix * slm.py_side_m
    X = xi * slm.px_side_m
    Y = yi * slm.py_side_m

    if aperture_radius_m and aperture_radius_m > 0:
        norm_radius = aperture_radius_m
    else:
        norm_radius = min(nx * slm.px_side_m, ny * slm.py_side_m) / 2.0
    if norm_radius <= 0:
        norm_radius = 1.0
    x_norm = (X / norm_radius).astype(np.float64)
    y_norm = (Y / norm_radius).astype(np.float64)
    r_norm = np.hypot(x_norm, y_norm)

    return Coordinates(
        nx=nx,
        ny=ny,
        px_m=slm.px_side_m,
        py_m=slm.py_side_m,
        xi=xi,
        yi=yi,
        X=X,
        Y=Y,
        x_m=x_m,
        y_m=y_m,
        x_mm=x_m * 1e3,
        y_mm=y_m * 1e3,
        center_mode=center_mode,
        r_norm=r_norm,
        x_norm=x_norm,
        y_norm=y_norm,
    )

def make_vortex_phase(
    coords: Coordinates, 
    ell: int, 
    sft_x: float = 0.0, 
    sft_y: float = 0.0,
    rotation: float = 0.0,
    alpha: float = 1.0,  # X-axis scaling
    beta: float = 1.0    # Y-axis scaling
) -> np.ndarray:
    if not isinstance(ell, int) or ell < 1:
        raise ValueError("ell must be a positive integer.")
    
    # Scale coordinates to compensate for elliptical beam shape
    dy = beta * (coords.Y - sft_y)
    dx = alpha * (coords.X - sft_x)
    
    # Calculate azimuthal angle with rotation
    theta = np.arctan2(dy, dx) - rotation
    
    return ell * theta

def make_zernike_phase(
    coords: Coordinates,
    offset_x_m: float,
    offset_y_m: float,
    norm_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = coords.X - offset_x_m
    y = coords.Y - offset_y_m
    norm_radius = norm_radius_m if norm_radius_m > 0 else 1.0
    xn = x / norm_radius
    yn = y / norm_radius
    r2 = xn * xn + yn * yn

    z_astig_v = 2.0 * xn * yn
    z_astig_o = xn * xn - yn * yn
    z_coma_y = (3.0 * r2 - 2.0) * yn
    z_coma_x = (3.0 * r2 - 2.0) * xn
    z_spher = 6.0 * r2 * r2 - 6.0 * r2 + 1.0

    return z_astig_v, z_astig_o, z_coma_y, z_coma_x, z_spher


def make_shift_phase(coords: Coordinates, fcp_x: float, fcp_y: float) -> np.ndarray:
    return 2 * np.pi * (fcp_x * coords.xi + fcp_y * coords.yi)


def make_circular_aperture(
    coords: Coordinates, radius_m: float, sft_x_m: float = 0.0, sft_y_m: float = 0.0
) -> np.ndarray:
    r = np.hypot(coords.X - sft_x_m, coords.Y - sft_y_m)
    return (r <= radius_m).astype(np.float64)


def theta_from_shift(delta_x_m: float, delta_y_m: float, f_m: float) -> tuple[float, float]:
    if f_m <= 0:
        raise ValueError("f_m must be > 0.")
    return np.arctan(delta_x_m / f_m), np.arctan(delta_y_m / f_m)


def fcp_from_theta(
    theta_x_rad: float, theta_y_rad: float, slm: SlmParams, lambda_m: float
) -> tuple[float, float]:
    if lambda_m <= 0:
        raise ValueError("lambda_m must be > 0.")
    return theta_x_rad * slm.px_side_m / lambda_m, theta_y_rad * slm.py_side_m / lambda_m


def clamp_fcp_nyquist(
    fcp_x: float, fcp_y: float, limit: float = 0.5
) -> tuple[float, float, bool, float]:
    mag = np.hypot(fcp_x, fcp_y)
    if mag > limit:
        scale = limit / mag
        return fcp_x * scale, fcp_y * scale, True, scale
    return fcp_x, fcp_y, False, 1.0


def resolve_steering(
    req: SteeringRequest, slm: SlmParams, beam: BeamParams
) -> SteeringResult:
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

    delta_x_mm = 1e3 * req.focal_length_m * beam.lambda_m * fcp_x / slm.px_side_m
    delta_y_mm = 1e3 * req.focal_length_m * beam.lambda_m * fcp_y / slm.py_side_m

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


def generate_mask(
    slm: SlmParams,
    beam: BeamParams,
    ell: int,
    sft_x_m: float = 0.0,
    sft_y_m: float = 0.0,
    rotation_rad: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    zernike_offset_x_m: float = 0.0,
    zernike_offset_y_m: float = 0.0,
    c_astig_v: float = 0.0,
    c_astig_o: float = 0.0,
    c_coma_y: float = 0.0,
    c_coma_x: float = 0.0,
    c_spher: float = 0.0,
    use_zernike: bool = False,
    steer_req: Optional[SteeringRequest] = None,
    use_forked: bool = True,
    aperture_radius_m: Optional[float] = None,
    force_zero: bool = True,
    calib_mask: Optional[np.ndarray] = None,
) -> MaskResult:
    coords = make_coordinates(
        slm, force_zero=force_zero, aperture_radius_m=aperture_radius_m
    )

    vortex = make_vortex_phase(
        coords,
        ell,
        sft_x=sft_x_m,
        sft_y=sft_y_m,
        rotation=rotation_rad,
        alpha=alpha,
        beta=beta,
    )
    if aperture_radius_m is not None:
        vortex = vortex * make_circular_aperture(
            coords, aperture_radius_m, sft_x_m=sft_x_m, sft_y_m=sft_y_m
        )

    steer = None
    if use_forked:
        steer_req = steer_req or SteeringRequest()
        steer = resolve_steering(steer_req, slm, beam)
        shift = make_shift_phase(coords, steer.fcp_x, steer.fcp_y)
    else:
        shift = np.zeros_like(vortex)

    zernike = 0.0
    if use_zernike:
        z_astig_v, z_astig_o, z_coma_y, z_coma_x, z_spher = make_zernike_phase(
            coords,
            offset_x_m=zernike_offset_x_m,
            offset_y_m=zernike_offset_y_m,
            norm_radius_m=(
                aperture_radius_m
                if (aperture_radius_m and aperture_radius_m > 0)
                else min(slm.nx * slm.px_side_m, slm.ny * slm.py_side_m) / 2.0
            ),
        )
        zernike = (
            c_astig_v * z_astig_v
            + c_astig_o * z_astig_o
            + c_coma_y * z_coma_y
            + c_coma_x * z_coma_x
            + c_spher * z_spher
        )

    desired = vortex + shift + zernike
    phi_wrapped = wrap_phase(desired)
    mask_u8 = phase_to_gray(phi_wrapped, slm.c2pi2unit)

    if calib_mask is not None:
        mask_u8 = apply_calibration(mask_u8, calib_mask, slm.c2pi2unit)

    return MaskResult(phi_wrapped=phi_wrapped, mask_u8=mask_u8, steer=steer)
