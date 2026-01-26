import numpy as np
from PIL import Image
from gerchberg_saxton import run_gerchberg_saxton


def energy_normalize_target(target_amp, source_amp):
    target_amp = target_amp.astype(float)
    source_energy = np.sum(np.abs(source_amp)**2)
    target_energy = np.sum(target_amp**2)
    if target_energy > 0:
        scaling = np.sqrt(source_energy / target_energy)
        return target_amp * scaling
    else:
        return target_amp


def resize_target_amp(target_amp: np.ndarray, slm_nx: int, slm_ny: int) -> np.ndarray:
    target_amp = np.asarray(target_amp, dtype=np.float32)
    input_energy = float(np.sum(target_amp ** 2))
    pil_in = Image.fromarray(target_amp)
    pil_out = pil_in.resize((slm_nx, slm_ny), resample=Image.BILINEAR)
    target_resized = np.array(pil_out, dtype=np.float32, copy=True)
    output_energy = float(np.sum(target_resized ** 2))
    if output_energy > 0 and input_energy > 0:
        scale = np.sqrt(input_energy / output_energy)
        return target_resized * scale
    else:
        return target_resized


def resize_amp_to_match(source_amp: np.ndarray, target_shape) -> np.ndarray:
    """
    Resize source_amp to match target_shape (ny, nx) while preserving total energy.
    Accepts float arrays in [0,1] or arbitrary scale; returns float32.
    """
    source_amp = np.asarray(source_amp, dtype=np.float32)

    # If RGB, convert to grayscale amplitude (luminance)
    if source_amp.ndim == 3 and source_amp.shape[2] in (3, 4):
        # Drop alpha if present and use standard luminance weights
        rgb = source_amp[..., :3]
        source_amp = (0.2126 * rgb[..., 0] + 0.7152 *
                      rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)

    # Already correct shape
    if source_amp.shape == tuple(target_shape):
        return source_amp

    in_energy = float(np.sum(source_amp ** 2))
    # PIL expects (width, height)
    ny, nx = target_shape
    pil_in = Image.fromarray(source_amp)
    pil_out = pil_in.resize((nx, ny), resample=Image.BILINEAR)
    out_amp = np.array(pil_out, dtype=np.float32, copy=True)

    # Energy-preserving scale
    out_energy = float(np.sum(out_amp ** 2))
    if in_energy > 0 and out_energy > 0:
        out_amp *= np.sqrt(in_energy / out_energy)

    return out_amp


def make_linear_tilt_degrees(nx, ny, theta_x_deg, theta_y_deg, wavelength, pixel_size):
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y)  # (nx, ny)
    theta_x = np.deg2rad(theta_x_deg)
    theta_y = np.deg2rad(theta_y_deg)
    fx = (pixel_size / wavelength) * np.sin(theta_x)
    fy = (pixel_size / wavelength) * np.sin(theta_y)
    tilt_phase = 2.0 * np.pi * (fx * X + fy * Y)
    return tilt_phase.astype(np.float64)


# def phase_to_img(phase_mask):
#     return 255.0 * phase_mask / (2*np.pi)


def img_to_phase(phase_img, signal_2pi):
    return phase_img * (2*np.pi) / signal_2pi


def run(num_iter: int,
        slm_nx: int,
        slm_ny: int,
        size_mode: str,
        correction_pattern_fp: str,
        signal_2pi: int,
        target_fp: str,
        output_bmp_fp,
        source_fp: str = '',
        method: str = 'far',
        z: float = None,
        M: float = None,
        tilt_x_deg: float = 0.0,
        tilt_y_deg: float = 0.0,
        wavelength: float = 775e-9,
        pixel_size: float = 12.5e-6,
        progress_cb=None,
        cancel_event=None):

    # read target image (amplitude)
    target_amp = np.asarray(Image.open(target_fp))
    if size_mode == 'resized':
        target_amp = resize_target_amp(target_amp, slm_nx, slm_ny)
    nx, ny = target_amp.shape

    # initial source field amplitude
    if source_fp:
        source_img = Image.open(source_fp)
        source_amp = np.array(source_img, dtype=float) / 255.0
        source_amp = resize_amp_to_match(source_amp, (nx, ny))
    else:
        source_amp = np.ones((nx, ny), dtype=float)

    # normalize target energy
    # target_amp = energy_normalize_target(target_amp, source_amp)

    # Cooperative cancel check
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("CANCELED")

    # run GS with callbacks (no plotting here; UI handles plotting)
    calc_phase_mask, out_recon_amp = run_gerchberg_saxton(
        source_amp=source_amp,
        target_amp=target_amp,
        num_iter=num_iter,
        wavelength=wavelength,
        pixel_size=pixel_size,
        method=method,
        z=z,
        M=M,
        plot_progress=False,
        return_history=False,
        progress_cb=progress_cb,
        cancel_event=cancel_event
    )

    if size_mode == 'tiled':
        reps_x = int(np.ceil(slm_nx / calc_phase_mask.shape[1]))
        reps_y = int(np.ceil(slm_ny / calc_phase_mask.shape[0]))
        calc_phase_mask = np.tile(calc_phase_mask, (reps_y, reps_x))[
            :slm_ny, :slm_nx]

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


if __name__ == "__main__":
    slm_nx = 1272
    slm_ny = 1024

    correction_pattern_fp = r"C:\SLM\Correction_patterns\CAL_LSH0805598_770nm.bmp"
    correction_pattern_fp = None

    signal_2pi = 255.0

    # target_fp = rr"C:\SLM\python_code\images\point-13_100x100.bmp"
    # target_fp = r"C:\SLM\python_code\images\Test_1272x1024.bmp"
    # target_fp = r"C:\SLM\python_code\images\Test_1272x1024_small.bmp"
    # target_fp = r"C:\SLM\python_code\images\Test_159x128.bmp"
    # target_fp = r"C:\SLM\python_code\images\circle_159x128.bmp"
    # target_fp = r"C:\SLM\python_code\images\circle_1272x1024.bmp"
    target_fp = r"C:\SLM\python_code\images\char_hpk_128x128.bmp"
    # target_fp = r"C:\SLM\python_code\images\hole_1272x1024.bmp"
    # target_fp = r"C:\SLM\python_code\images\point-13_1272x1024.bmp"

    size_mode = 'resized'  # 'resized' ro 'tiled'
    # size_mode = ''  # 'resized' ro 'tiled'
    # size_mode = 'tiled'  # 'resized' ro 'tiled'

    output_bmp_fp = r"C:\SLM\python_code\new_app\output\test_python.bmp"

    tilt_y_deg = 0.0
    # tilt_y_deg = -3.0

    method = 'far'
    z = 0.4
    num_iter = 5
    M = None
    run(num_iter, slm_nx, slm_ny, size_mode, correction_pattern_fp, signal_2pi, target_fp, output_bmp_fp, tilt_y_deg=tilt_y_deg,
        method=method, z=z, M=M)

    def _ifft2c(x):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), norm='ortho'))

    ham_phase = img_to_phase(np.asarray(
        # Image.open(r"C:\SLM\python_code\new_app\output\ham_tiled.bmp")), signal_2pi)
        Image.open(r"C:\SLM\python_code\new_app\output\slm_vortex_forked_ell100_1272x1024.bmp")), signal_2pi)
    # Image.open(r"C:\SLM\python_code\new_app\output\test_ham.bmp")), signal_2pi)
    python_phase = img_to_phase(np.asarray(
        Image.open(r"C:\SLM\python_code\new_app\output\slm_vortex_forked_ell30_1272x1024.bmp")), signal_2pi)
    # Image.open(r"C:\SLM\python_code\new_app\output\test_python.bmp")), signal_2pi)
    input = np.ones_like(ham_phase)

    out_ham = np.abs(_ifft2c(input * np.exp(1j * ham_phase)))**2
    out_python = np.abs(_ifft2c(input * np.exp(1j * python_phase)))**2

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.imshow(ham_phase)
    plt.title("HAM")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(python_phase)
    plt.title("Python")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    # log1p(x) = log(1 + x), avoids log(0)
    plt.imshow(np.log1p(out_ham), cmap='gray')
    plt.title("HAM (log scale)")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(np.log1p(out_python), cmap='gray')
    plt.title("Python (log scale)")
    plt.axis('off')

    plt.show()
