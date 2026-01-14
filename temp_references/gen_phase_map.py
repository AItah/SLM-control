import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gerchberg_saxton import run_gerchberg_saxton


def energy_normalize_target(target_amp, source_amp):
    """
    Normalize target so total energy matches source energy
    """
    target_amp = target_amp.astype(float)
    source_energy = np.sum(np.abs(source_amp)**2)
    target_energy = np.sum(target_amp**2)

    if target_energy > 0:
        scaling = np.sqrt(source_energy / target_energy)
        return target_amp * scaling
    else:
        return target_amp


def resize_target_amp(target_amp: np.ndarray, slm_nx: int, slm_ny: int) -> np.ndarray:
    """
    Resize target amplitude to (slm_nx, slm_ny) and preserve total energy (sum of amp^2).
    Returns a writable float32 array.
    """
    # ensure float32
    target_amp = np.asarray(target_amp, dtype=np.float32)

    # input energy (intensity sum)
    input_energy = float(np.sum(target_amp ** 2))

    # PIL resize (bilinear). Use copy=True to ensure writable array later.
    pil_in = Image.fromarray(target_amp)
    pil_out = pil_in.resize((slm_nx, slm_ny), resample=Image.BILINEAR)

    # Make sure we get a writable array (copy=True)
    target_resized = np.array(pil_out, dtype=np.float32, copy=True)

    # Preserve energy (avoid in-place to dodge read-only issues)
    output_energy = float(np.sum(target_resized ** 2))
    if output_energy > 0 and input_energy > 0:
        scale = np.sqrt(input_energy / output_energy)
        return target_resized * scale
    else:
        # If either energy is zero, just return the resized map
        return target_resized


def make_linear_tilt_degrees(nx, ny, theta_x_deg, theta_y_deg, wavelength, pixel_size):
    """
    Create a linear phase ramp from steering angles (degrees) at the SLM.
    theta_x_deg, theta_y_deg: angles in degrees (about x and y axes).
    wavelength: meters, pixel_size: meters (SLM pitch).

    Returns:
        tilt_phase (nx, ny) in radians (wrap later if desired).
    """
    # grid: X along columns [0..ny-1], Y along rows [0..nx-1]
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y)  # shapes (nx, ny)

    # convert to radians
    theta_x = np.deg2rad(theta_x_deg)
    theta_y = np.deg2rad(theta_y_deg)

    # slope in cycles/pixel from angle: f = (pixel_size / wavelength) * sin(theta)
    # cycles/pixel along x (columns)
    fx = (pixel_size / wavelength) * np.sin(theta_x)
    # cycles/pixel along y (rows)
    fy = (pixel_size / wavelength) * np.sin(theta_y)

    # phase ramp
    tilt_phase = 2.0 * np.pi * (fx * X + fy * Y)
    return tilt_phase.astype(np.float64)


def plot_single(source_amp, target_amp, phase_mask, reconst_amp, title=''):
    plt.figure(figsize=(10, 4))
    plt.title(title)

    plt.subplot(2, 2, 1)
    plt.title("Source Amplitude")
    plt.imshow(source_amp, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Target Amplitude")
    plt.imshow(target_amp, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Phase Map")
    plt.imshow(phase_mask, cmap='hsv')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Reconst. Amplitude")
    plt.imshow(reconst_amp, cmap='gray')
    plt.axis('off')


def phase_to_img(phase_mask):
    return 255.0 * phase_mask / (2*np.pi)


def img_to_phase(phase_img):
    return phase_img * (2*np.pi) / 255.0


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
        plot: bool = True,
        tilt_x_deg: float = 0.0,
        tilt_y_deg: float = 0.0):

    # read target image
    target_amp = np.asarray(Image.open(target_fp))
    # resize if needed
    if size_mode == 'resized':
        target_amp = resize_target_amp(target_amp, slm_nx, slm_ny)
    nx, ny = target_amp.shape

    # initial source field amplitude & phase
    source_input = Image.open(source_fp) if source_fp else 1.0

    if np.isscalar(source_input):
        source_amp = np.ones((nx, ny), dtype=float) * float(source_input)
    else:
        source_amp = np.array(source_input, dtype=float)
        if source_amp.shape != target_amp.shape:
            raise ValueError("source_amp shape must match target_amp shape")

    # normalize the target so the total energy remains the same
    target_amp = energy_normalize_target(target_amp, source_amp)

    # run the algorithm
    wavelength = 775e-9  # relevant for near field only
    pixel_size = 12.5e-6  # relevant for near field only

    calc_phase_mask, out_recon_amp = run_gerchberg_saxton(
        source_amp=source_amp,
        target_amp=target_amp,
        num_iter=num_iter,
        wavelength=wavelength,
        pixel_size=pixel_size,
        method=method,
        z=z,
        M=M,
        plot_progress=plot,
        return_history=False)

    if size_mode == 'tiled':
        # Tile from top-left, crop on right/bottom as needed
        reps_x = int(np.ceil(slm_nx / calc_phase_mask.shape[1]))
        reps_y = int(np.ceil(slm_ny / calc_phase_mask.shape[0]))
        calc_phase_mask = np.tile(calc_phase_mask, (reps_y, reps_x))[
            :slm_ny, :slm_nx]

    # linear tilt from angles in degrees
    tilt_phase = make_linear_tilt_degrees(slm_nx, slm_ny,
                                          theta_x_deg=tilt_x_deg,
                                          theta_y_deg=tilt_y_deg,
                                          wavelength=wavelength,
                                          pixel_size=pixel_size)

    # applying correction pattern (+ GS phase + tilt), wrapped to [0, 2π)
    corr_phase = img_to_phase(np.asarray(Image.open(correction_pattern_fp)))
    out_phase_mask = (corr_phase + calc_phase_mask + tilt_phase) % (2 * np.pi)

    # writing to bmp
    out_phase_image = signal_2pi*(out_phase_mask + np.pi) / (2*np.pi)
    Image.fromarray(out_phase_image).convert('L').save(output_bmp_fp)

    # # plotting
    # if plot:
    #     plot_single(source_amp, target_amp, calc_phase_mask,
    #                 out_recon_amp, 'Output')
    #     plt.show()


if __name__ == "__main__":
    slm_nx = 1272
    slm_ny = 1024

    correction_pattern_fp = r"C:\SLM\Correction_patterns\CAL_LSH0805598_770nm.bmp"

    signal_2pi = 205

    # target_fp = rr"C:\SLM\python_code\images\point-13_100x100.bmp"
    target_fp = r"C:\SLM\python_code\images\Test_1272x1024.bmp"
    target_fp = r"C:\SLM\python_code\images\Test_1272x1024_small.bmp"
    target_fp = r"C:\SLM\python_code\images\Test_159x128.bmp"
    target_fp = r"C:\SLM\python_code\images\circle_159x128.bmp"
    target_fp = r"C:\SLM\python_code\images\circle_1272x1024.bmp"
    # target_fp = r"C:\SLM\python_code\images\hole_1272x1024.bmp"
    # target_fp = r"C:\SLM\python_code\images\point-13_1272x1024.bmp"

    size_mode = 'resized'  # 'resized' ro 'tiled'
    # size_mode = 'tiled'  # 'resized' ro 'tiled'

    output_bmp_fp = r"C:\SLM\python_code\phase_maps\output_mask_old.bmp"

    tilt_y_deg = 0.0
    # tilt_y_deg = -3.0

    method = 'near'
    z = 0.4
    num_iter = 10
    M = None
    run(num_iter, slm_nx, slm_ny, size_mode, correction_pattern_fp, signal_2pi, target_fp, output_bmp_fp, tilt_y_deg=tilt_y_deg,
        method=method, z=z, M=M, plot=True)
