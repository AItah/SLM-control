import numpy as np
import matplotlib.pyplot as plt


def _fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm='ortho'))


def _ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), norm='ortho'))


def _angular_spectrum_propagate(u0, wavelength, dx, z):
    k = 2 * np.pi / wavelength
    nx, ny = u0.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    rho_sq = FX**2 + FY**2
    kz = np.sqrt((k**2 - (2*np.pi)**2 * rho_sq) + 0j)
    H = np.exp(1j * z * kz)
    mask = rho_sq < (1/wavelength)**2
    H = H * mask
    U0 = _fft2c(u0)
    Uz = U0 * H
    uz = _ifft2c(Uz)
    return uz


def _fraunhofer_propagate(u0):
    return _fft2c(u0)


def _xy_coords(shape, dx):
    nx, ny = shape
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def _thin_lens_phase(shape, wavelength, dx, f):
    k = 2 * np.pi / wavelength
    X, Y = _xy_coords(shape, dx)
    return np.exp(-1j * k * (X**2 + Y**2) / (2.0 * f))


def _embed_center(small, canvas_shape):
    NXc, NYc = canvas_shape
    nx, ny = small.shape
    canvas = np.zeros((NXc, NYc), dtype=small.dtype)
    x0 = (NXc - nx) // 2
    y0 = (NYc - ny) // 2
    xs = max(0, -x0)
    ys = max(0, -y0)
    xe = min(nx, NXc - x0)
    ye = min(ny, NYc - y0)
    canvas[max(0, x0):max(0, x0)+xe-xs,
           max(0, y0):max(0, y0)+ye-ys] = small[xs:xe, ys:ye]
    return canvas


def _virtual_4f_forward_M(u_in, wavelength, dx, M):
    f1 = 1.0
    f2 = float(M) * f1
    u = u_in * _thin_lens_phase(u_in.shape, wavelength, dx, f1)
    u = _angular_spectrum_propagate(u, wavelength, dx, f1)
    u = u * _thin_lens_phase(u.shape, wavelength, dx, f2)
    u = _angular_spectrum_propagate(u, wavelength, dx, f2)
    return u


def _virtual_4f_backward_M(u_out, wavelength, dx, M):
    f1 = 1.0
    f2 = float(M) * f1
    u = _angular_spectrum_propagate(u_out, wavelength, dx, -f2)
    u = u * np.conj(_thin_lens_phase(u.shape, wavelength, dx, f2))
    u = _angular_spectrum_propagate(u, wavelength, dx, -f1)
    u = u * np.conj(_thin_lens_phase(u.shape, wavelength, dx, f1))
    return u


def run_gerchberg_saxton(source_amp,
                         target_amp,
                         num_iter=100,
                         wavelength=632.8e-9,
                         pixel_size=8e-6,
                         z=None,
                         method='far',            # 'far' or 'near'
                         initial_phase=None,
                         return_complex=False,
                         return_history=False,
                         M=None,                  # magnification only used for method='near'
                         # (disabled in UI path; we stream data via callback)
                         plot_progress=False,
                         plot_interval=1,
                         # NEW: callback(it, total, u_source, u_target, target_canvas, mse)
                         progress_cb=None,
                         cancel_event=None):      # NEW: cooperative cancel
    """
    GS with optional progress callback + cooperative cancel.
    """
    if method not in ('far', 'near'):
        raise ValueError("method must be 'far' or 'near'")

    source_amp = np.asarray(source_amp, dtype=float)
    if source_amp.ndim != 2:
        raise ValueError("source_amp must be 2D")
    NX, NY = source_amp.shape

    tgt = np.asarray(target_amp, dtype=float)
    if tgt.ndim != 2:
        raise ValueError("target_amp must be 2D")

    target_canvas = tgt.copy() if tgt.shape == (
        NX, NY) else _embed_center(tgt, (NX, NY))

    if initial_phase is None:
        phi = np.random.uniform(-np.pi, np.pi, size=(NX, NY))
    else:
        phi = np.asarray(initial_phase, dtype=float)
        if phi.shape != (NX, NY):
            raise ValueError("initial_phase shape must match source_amp shape")

    u_source = source_amp * np.exp(1j * phi)
    history = [] if return_history else None

    # Optional Matplotlib live plotting still available for standalone usage
    if plot_progress:
        plt.ion()
        fig, ax = plt.subplots(1, 1)
        h = ax.imshow(np.abs(_fraunhofer_propagate(u_source)), cmap='gray')
        plt.show()

    for it in range(1, int(num_iter) + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("CANCELED")

        # Forward
        if method == 'far':
            u_target = _fraunhofer_propagate(u_source)
        else:
            if M is None:
                if z is None:
                    raise ValueError(
                        "For method='near' with M=None, supply z (propagation distance).")
                u_target = _angular_spectrum_propagate(
                    u_source, wavelength, pixel_size, z)
            else:
                u_target = _virtual_4f_forward_M(
                    u_source, wavelength, pixel_size, M)

        # Callback / plotting
        if progress_cb is not None or plot_progress:
            mse = float(np.mean(np.abs(np.abs(u_target) - target_canvas)**2))
            if progress_cb is not None:
                progress_cb(it, int(num_iter), u_source,
                            u_target, target_canvas, mse)
            if plot_progress and (it % plot_interval == 0 or it == num_iter):
                h.set_data(np.abs(u_target))
                h.set_clim(vmin=np.min(np.abs(u_target)),
                           vmax=np.max(np.abs(u_target)))
                fig.canvas.draw_idle()
                plt.pause(0.001)

        # Impose target amplitude
        u_target = target_canvas * np.exp(1j * np.angle(u_target))
        if return_history:
            history.append(np.abs(u_target) ** 2)

        # Backward
        if method == 'far':
            u_source = _ifft2c(u_target)
        else:
            if M is None:
                u_source = _angular_spectrum_propagate(
                    u_target, wavelength, pixel_size, -z)
            else:
                u_source = _virtual_4f_backward_M(
                    u_target, wavelength, pixel_size, M)

        # Phase-only SLM constraint
        u_source = source_amp * np.exp(1j * np.angle(u_source))

    if method == 'far':
        u_target_final = _fraunhofer_propagate(u_source)
    else:
        if M is None:
            u_target_final = _angular_spectrum_propagate(
                u_source, wavelength, pixel_size, z)
        else:
            u_target_final = _virtual_4f_forward_M(
                u_source, wavelength, pixel_size, M)

    recon_amp = np.abs(u_target_final)
    final_phase_mask = np.angle(u_source)
    final_phase_mask[final_phase_mask < 0] += 2*np.pi   # (0, 2π]

    if plot_progress:
        plt.ioff()
        plt.show()

    outputs = (final_phase_mask, recon_amp)
    if return_complex:
        outputs = outputs + (u_source,)
    if return_history:
        outputs = outputs + (history,)
    return outputs if len(outputs) > 1 else outputs[0]


if __name__ == '__main__':
    from PIL import Image

    # Image size
    width, height = 1272, 1024  # You can change this

    # Create a grayscale image ("L" mode = 8-bit pixels, black and white)
    img = Image.new("L", (width, height), color=205)

    # Save as BMP
    img.save("C:\SLM\python_code\phase_maps\gray_205.bmp")
