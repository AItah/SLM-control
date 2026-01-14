import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from gen_phase_map import make_linear_tilt_degrees, img_to_phase
from PIL import Image


def generate_spiral_phase(nx=512, ny=512, ell=1.0, x0=None, y0=None):
    if x0 is None:
        x0 = (nx - 1) / 2.0
    if y0 is None:
        y0 = (ny - 1) / 2.0

    x = np.arange(nx) - x0
    y = np.arange(ny) - y0
    X, Y = np.meshgrid(x, y)

    theta = np.arctan2(Y, X)
    phase = (ell * theta) % (2 * np.pi)
    return phase


def generate_fork_phase(nx=512, ny=512, ell=1, period=32, x0=None, y0=None):
    if x0 is None:
        x0 = (nx - 1) / 2.0
    if y0 is None:
        y0 = (ny - 1) / 2.0

    x = np.arange(nx) - x0
    y = np.arange(ny) - y0
    X, Y = np.meshgrid(x, y)

    theta = np.arctan2(Y, X)
    grating = 2 * np.pi * X / period      # linear grating term
    phase = (grating + ell * theta) % (2 * np.pi)
    return phase


def gaussian_beam_amplitude(nx=512, ny=512, w0=80, x0=None, y0=None):
    if x0 is None:
        x0 = (nx - 1) / 2.0
    if y0 is None:
        y0 = (ny - 1) / 2.0

    x = np.arange(nx) - x0
    y = np.arange(ny) - y0
    X, Y = np.meshgrid(x, y)

    R2 = X*2 + Y*2
    A = np.exp(-R2 / (w0**2))
    return A


def far_field_intensity(field):
    """
    Simple Fraunhofer propagation: squared magnitude of 2D FFT.
    """
    F = np.fft.fftshift(np.fft.fft2(field))
    I = np.abs(F)**2
    I /= I.max()  # normalize
    return I


def benchmark_methods():
    nx = ny = 512
    x0 = y0 = (nx - 1) / 2.0
    gauss_amp = gaussian_beam_amplitude(nx, ny, w0=80, x0=x0, y0=y0)

    # -------- Spiral type: 3 different ells --------
    spiral_ells = [10, 20, 30]

    for ell in spiral_ells:
        phase = generate_spiral_phase(nx, ny, ell=ell, x0=x0, y0=y0)
        field = gauss_amp * np.exp(1j * phase)
        intensity = far_field_intensity(field)

        plt.figure(figsize=(5, 5))
        plt.imshow(phase, origin="lower")
        plt.title(f"Spiral Phase (ell = {ell})")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.colorbar(label="Phase [rad]")
        plt.tight_layout()

        plt.figure(figsize=(5, 5))
        vmin_val = intensity.min()
        vmax_val = intensity.max()
        plt.imshow(intensity, cmap=cm.rainbow,
                   norm=colors.LogNorm(vmin=vmin_val, vmax=vmax_val))
        plt.title(f"Far-field Intensity (Spiral, ell = {ell})")
        plt.xlabel("kx")
        plt.ylabel("ky")
        plt.colorbar(label="Normalized Intensity")
        plt.tight_layout()

    # -------- Fork type: 4 combinations (2 ells × 2 periods) --------
    fork_ells = [10, 20]
    fork_periods = [32, 64]

    for ell in fork_ells:
        for period in fork_periods:
            phase = generate_fork_phase(
                nx, ny, ell=ell, period=period, x0=x0, y0=y0)
            field = gauss_amp * np.exp(1j * phase)
            intensity = far_field_intensity(field)

            plt.figure(figsize=(5, 5))
            plt.imshow(phase, origin="lower")
            plt.title(f"Fork Phase (ell = {ell}, period = {period} px)")
            plt.xlabel("x (pixels)")
            plt.ylabel("y (pixels)")
            plt.colorbar(label="Phase [rad]")
            plt.tight_layout()

            plt.figure(figsize=(5, 5))
            vmin_val = intensity.min()
            vmax_val = intensity.max()
            plt.imshow(intensity, cmap=cm.rainbow,
                       norm=colors.LogNorm(vmin=vmin_val, vmax=vmax_val))
            plt.title(
                f"Far-field Intensity (Fork, ell = {ell}, period = {period} px)")
            plt.xlabel("kx")
            plt.ylabel("ky")
            plt.colorbar(label="Normalized Intensity")
            plt.tight_layout()
    plt.show()


def run(
        slm_nx: int,
        slm_ny: int,
        correction_pattern_fp: str,
        signal_2pi: int,
        output_bmp_fp,
        tilt_x_deg: float = 0.0,
        tilt_y_deg: float = 0.0,
        ell: float = 1.0,
        wavelength: float = 775e-9,
        pixel_size: float = 12.5e-6
):

    calc_phase_mask = generate_spiral_phase(slm_nx, slm_ny, ell=ell)

    # linear tilt
    tilt_phase = make_linear_tilt_degrees(slm_nx, slm_ny,
                                          theta_x_deg=tilt_x_deg,
                                          theta_y_deg=tilt_y_deg,
                                          wavelength=wavelength,
                                          pixel_size=pixel_size)

    # correction pattern (optional)
    if correction_pattern_fp:
        corr_phase = img_to_phase(np.asarray(
            Image.open(correction_pattern_fp)), 255.0)
    else:
        corr_phase = np.zeros((slm_ny, slm_nx), dtype=np.float64)

    out_phase_mask = (corr_phase + calc_phase_mask + tilt_phase) % (2 * np.pi)

    # writing to bmp
    out_phase_image = signal_2pi*(out_phase_mask) / (2*np.pi)
    Image.fromarray(out_phase_image).convert('L').save(output_bmp_fp)


if __name__ == '__main__':
    slm_nx = 1272
    slm_ny = 1024

    correction_pattern_fp = r"C:\SLM\Correction_patterns\CAL_LSH0805598_770nm.bmp"
    # correction_pattern_fp = None

    # signal_2pi = 255.0
    signal_2pi = 204.0

    tilt_x_deg = -0.5
    # tilt_x_deg = 0.0

    tilt_y_deg = 0.0
    # tilt_y_deg = -0.5

    ell = 1

    output_bmp_fp = rf"C:\SLM\python_code\new_app\output\vortex\vortex_ell{ell}_x{tilt_x_deg}_y{tilt_y_deg}.bmp"

    run(slm_nx, slm_ny, correction_pattern_fp, signal_2pi,
        output_bmp_fp, tilt_x_deg, tilt_y_deg, ell)
