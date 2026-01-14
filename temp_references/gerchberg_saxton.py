import numpy as np
import matplotlib.pyplot as plt

# ==================== Core FFT helpers (unchanged) ====================


def _fft2c(x):
    """Centered 2D FFT with orthonormal normalization."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm='ortho'))


def _ifft2c(x):
    """Centered 2D inverse FFT with orthonormal normalization."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), norm='ortho'))

# ==================== Angular Spectrum (near-field) - CORRECTED ====================


def _angular_spectrum_propagate(u0, wavelength, dx, z):
    """
    Corrected angular spectrum propagation.
    """
    k = 2 * np.pi / wavelength
    nx, ny = u0.shape

    # Frequency grids in proper scaling (1/m)
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    # Note: indexing='ij' for consistency
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # Spatial frequencies squared
    rho_sq = FX**2 + FY**2

    # Propagation kernel - handle evanescent waves properly
    kz = np.sqrt((k**2 - (2*np.pi)**2 * rho_sq) + 0j)

    # For evanescent waves (imaginary kz), we typically set to 0 or use small value
    # This prevents numerical instability while maintaining physical accuracy
    H = np.exp(1j * z * kz)

    # Apply low-pass filter to remove evanescent waves beyond numerical precision
    # This is often the cause of grid artifacts
    mask = rho_sq < (1/wavelength)**2  # Physical propagation region
    H = H * mask

    # Propagate
    U0 = _fft2c(u0)
    Uz = U0 * H
    uz = _ifft2c(Uz)

    return uz

# ==================== Fraunhofer (far-field) ====================


def _fraunhofer_propagate(u0):
    """Fraunhofer (far-field) using centered FFT."""
    return _fft2c(u0)

# ==================== Utilities ====================


def _xy_coords(shape, dx):
    """Centered coordinate grids X,Y in meters for given shape and pixel size dx."""
    nx, ny = shape
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    # Use 'ij' indexing for consistency
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def _thin_lens_phase(shape, wavelength, dx, f):
    """Quadratic phase of a thin lens with focal length f (radians)."""
    k = 2 * np.pi / wavelength
    X, Y = _xy_coords(shape, dx)
    return np.exp(-1j * k * (X**2 + Y**2) / (2.0 * f))


def _embed_center(small, canvas_shape):
    """Center-embed/crop small into a zeros canvas of canvas_shape (no resampling)."""
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

# ==================== Virtual 4f (near-field) with magnification M ====================


def _virtual_4f_forward_M(u_in, wavelength, dx, M):
    """
    Forward propagation through a virtual 4f system with magnification M.
    Uses f1=1, f2=M (ratio matters). Works on the same array/grid; the image content
    is magnified by ~M within that grid (no resampling).
    """
    f1 = 1.0
    f2 = float(M) * f1
    u = u_in * _thin_lens_phase(u_in.shape, wavelength, dx, f1)
    u = _angular_spectrum_propagate(u, wavelength, dx, f1)
    u = u * _thin_lens_phase(u.shape, wavelength, dx, f2)
    u = _angular_spectrum_propagate(u, wavelength, dx, f2)
    return u


def _virtual_4f_backward_M(u_out, wavelength, dx, M):
    """Backward (adjoint) of the virtual 4f relay."""
    f1 = 1.0
    f2 = float(M) * f1
    u = _angular_spectrum_propagate(u_out, wavelength, dx, -f2)
    u = u * np.conj(_thin_lens_phase(u.shape, wavelength, dx, f2))
    u = _angular_spectrum_propagate(u, wavelength, dx, -f1)
    u = u * np.conj(_thin_lens_phase(u.shape, wavelength, dx, f1))
    return u

# ==================== Plotting Function ====================


def setup_gs_plot():
    """Set up the matplotlib figure for GS algorithm visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    plt.ion()  # Turn on interactive mode

    # Create initial empty plots and store references for updates
    im1 = axes[0, 0].imshow(np.zeros((10, 10)), cmap='hot')
    axes[0, 0].set_title('Reconstructed Amplitude')
    axes[0, 0].set_xlabel('X pixels')
    axes[0, 0].set_ylabel('Y pixels')
    cbar1 = fig.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(np.zeros((10, 10)), cmap='hot')
    axes[0, 1].set_title('Target Amplitude')
    axes[0, 1].set_xlabel('X pixels')
    axes[0, 1].set_ylabel('Y pixels')
    cbar2 = fig.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].imshow(np.zeros((10, 10)), cmap='hsv')
    axes[1, 0].set_title('Current Phase Mask')
    axes[1, 0].set_xlabel('X pixels')
    axes[1, 0].set_ylabel('Y pixels')
    cbar3 = fig.colorbar(im3, ax=axes[1, 0])

    error_line, = axes[1, 1].plot([], [], 'b-', linewidth=2)
    axes[1, 1].set_title('Error Progression')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Mean Squared Error')
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    # "+X+Y" → X=100px from left, Y=200px from top
    # manager.window.wm_geometry("+20+20")
    plt.show()

    # Store references for updating
    plot_refs = {
        'fig': fig,
        'axes': axes,
        'im1': im1,
        'im2': im2,
        'im3': im3,
        'cbar1': cbar1,
        'cbar2': cbar2,
        'cbar3': cbar3,
        'error_line': error_line
    }

    return plot_refs


def update_gs_plot(plot_refs, iteration, u_source, u_target, target_canvas, error_history):
    """Update the GS algorithm plots without creating new colorbars."""
    current_amp = np.abs(u_target)
    current_phase = np.angle(u_source)

    # Update image data
    plot_refs['im1'].set_data(current_amp)
    plot_refs['im2'].set_data(target_canvas)
    plot_refs['im3'].set_data(current_phase)

    # Update colorbar limits
    plot_refs['im1'].set_clim(vmin=current_amp.min(), vmax=current_amp.max())
    plot_refs['im2'].set_clim(vmin=target_canvas.min(),
                              vmax=target_canvas.max())
    plot_refs['im3'].set_clim(vmin=-np.pi, vmax=np.pi)

    # Update titles with iteration info
    plot_refs['axes'][0, 0].set_title(
        f'Iteration {iteration}: Reconstructed Amplitude')

    # Update error plot with full history
    iterations = list(range(len(error_history)))
    plot_refs['error_line'].set_data(iterations, error_history)

    # Adjust error plot limits
    if len(error_history) > 0:
        plot_refs['axes'][1, 1].set_xlim(0, max(iterations))
        plot_refs['axes'][1, 1].set_ylim(
            min(error_history) - 0.1*(max(error_history) - min(error_history)),
            max(error_history) + 0.1*(max(error_history) - min(error_history))
        )

    plt.draw()
    plt.pause(0.001)  # Short pause to update the plot

# ==================== Gerchberg–Saxton (near-field supports M) ====================


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
                         plot_progress=True,     # New: enable animated plotting
                         plot_interval=1         # New: update plot every N iterations
                         ):
    """
    If method='far':
        - Ignores M; uses standard Fraunhofer (FFT).
    If method='near':
        - If M is None -> standard ASM with distance z (must be given).
        - If M is given -> virtual 4f with magnification M (z is ignored).
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

    # center-embed target onto SLM grid (no resampling)
    target_canvas = tgt.copy() if tgt.shape == (
        NX, NY) else _embed_center(tgt, (NX, NY))

    # # normalize target amplitude for GS stability
    # tmax = target_canvas.max()
    # if tmax > 0:
    #     target_canvas = target_canvas / tmax

    # init phase
    if initial_phase is None:
        phi = np.random.uniform(-np.pi, np.pi, size=(NX, NY))
    else:
        phi = np.asarray(initial_phase, dtype=float)
        if phi.shape != (NX, NY):
            raise ValueError("initial_phase shape must match source_amp shape")

    u_source = source_amp * np.exp(1j * phi)
    history = [] if return_history else None

    # Setup plotting if enabled
    if plot_progress:
        plot_refs = setup_gs_plot()
        error_history = []

    for it in range(num_iter):
        print(f'{it} / {num_iter}')

        # ---------- Forward ----------
        if method == 'far':
            u_target = _fraunhofer_propagate(u_source)
        else:  # near
            if M is None:
                if z is None:
                    raise ValueError(
                        "For method='near' with M=None, supply z (propagation distance).")
                u_target = _angular_spectrum_propagate(
                    u_source, wavelength, pixel_size, z)
            else:
                u_target = _virtual_4f_forward_M(
                    u_source, wavelength, pixel_size, M)

        # ---------- Update plot if enabled ----------
        if plot_progress and (it % plot_interval == 0 or it == num_iter - 1):
            current_error = np.mean(
                np.abs(np.abs(u_target) - target_canvas)**2)
            error_history.append(current_error)
            update_gs_plot(plot_refs, it, u_source, u_target,
                           target_canvas, error_history)

        # ---------- Impose target amplitude ----------
        u_target = target_canvas * np.exp(1j * np.angle(u_target))
        if return_history:
            history.append(np.abs(u_target) ** 2)

        # ---------- Backward ----------
        if method == 'far':
            u_source = _ifft2c(u_target)
        else:  # near
            if M is None:
                u_source = _angular_spectrum_propagate(
                    u_target, wavelength, pixel_size, -z)
            else:
                u_source = _virtual_4f_backward_M(
                    u_target, wavelength, pixel_size, M)

        # ---------- Source constraint (phase-only SLM) ----------
        u_source = source_amp * np.exp(1j * np.angle(u_source))

    # ---------- Final forward (for report) ----------
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
    final_phase_mask = np.angle(u_source) + np.pi  # radians in (0, 2*pi]

    # Close the interactive plot if it was opened
    if plot_progress:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the final plot open

    outputs = (final_phase_mask, recon_amp)
    if return_complex:
        outputs = outputs + (u_source,)
    if return_history:
        outputs = outputs + (history,)
    return outputs if len(outputs) > 1 else outputs[0]
