# Vortex Mask Generation (Implementation Flow)

This document describes how the **vortex mask is generated in software**, how each term contributes to the phase, and how the SLM mask is constructed.

---

## 1) Coordinate System

The SLM grid is converted to physical coordinates:

$$
x = (i - x_0)\,\Delta x, \quad y = (j - y_0)\,\Delta y
$$

Where:
- $\Delta x, \Delta y$ are pixel pitches (meters).
- $(x_0, y_0)$ is the chosen center (including offset).

The polar coordinates are:

$$
r = \sqrt{x^2 + y^2}, \quad \theta = \tan^{-1}(y/x)
$$

---

## 2) Base Vortex Phase

The core helical phase:

$$
\phi_{\text{vortex}}(r,\theta) = \ell \theta
$$

This is converted to phase values and wrapped into $[0, 2\pi)$.

---

## 3) Offsets

Offsets shift the vortex center:

$$
x \leftarrow x - x_{\text{offset}}, \quad y \leftarrow y - y_{\text{offset}}
$$

This is how **Shift** optimization moves the dark spot relative to the donut.

---

## 4) Steering (Fork Grating / Tilt)

Steering adds a linear phase:

$$
\phi_{\text{steer}}(x,y) = k_x x + k_y y
$$

This can be represented as:
- **Angle mode**: derived from target angles.
- **Shift mode**: derived from target shifts.

The steering term is added to the vortex phase:

$$
\phi = \phi_{\text{vortex}} + \phi_{\text{steer}}
$$

---

## 5) Aperture

An optional circular aperture clips the mask:

$$
M(r) =
 \begin{cases}
 1, & r \le R \\
 0, & r > R
 \end{cases}
$$

The final mask becomes:

$$
E(r,\theta) \leftarrow e^{i\phi(r,\theta)} \cdot M(r)
$$

---

## 6) Zernike Corrections

If enabled, Zernike phase corrections are added:

$$
\phi \leftarrow \phi + \sum c_n Z_n(r,\theta)
$$

Where $c_n$ are the astigmatism, coma, and spherical coefficients.

---

## 7) Final SLM Encoding

The phase is mapped to the SLM integer range:

$$
\text{SLM\_value} = \text{round}\!\left(\frac{\phi \bmod 2\pi}{2\pi}\cdot c_{2\pi}\right)
$$

Where $c_{2\pi}$ is the **c2pi2unit** scaling parameter.

---

## 8) Suggested Images

- Phase accumulation diagram  
  `Docs/images/mask_phase_stack.png`
- Example of base vortex phase  
  `Docs/images/base_vortex_phase.png`
- Example of steering phase  
  `Docs/images/steering_phase.png`
- Aperture masking example  
  `Docs/images/aperture_mask.png`

