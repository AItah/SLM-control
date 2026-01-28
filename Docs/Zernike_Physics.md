# Zernike Corrections: Physics & Usage

This document explains **Zernike polynomials**, how they represent aberrations, and why they improve astigmatism, coma, and spherical errors.

---

## 1) Why Zernike?

Optical aberrations distort the wavefront $W(r,\theta)$.  
Zernike polynomials form an **orthogonal basis** on a unit disk, so any smooth wavefront can be written as:

$$
W(r,\theta) = \sum_{n,m} c_n^m Z_n^m(r,\theta)
$$

Where:
- $Z_n^m$ are Zernike polynomials.
- $c_n^m$ are coefficients (strength of aberrations).

Because they are orthogonal, each coefficient can be tuned independently.

---

## 2) Common Aberrations

Below are the most relevant terms (noting that exact normalization may vary):

### Astigmatism
Two orthogonal astigmatism modes:

$$
Z_{\text{astig,0}} \propto r^2 \cos(2\theta), \quad
Z_{\text{astig,45}} \propto r^2 \sin(2\theta)
$$

Astigmatism makes the focus elliptical, stretching in orthogonal axes.

### Coma
Coma causes asymmetric “comet‑shaped” blur:

$$
Z_{\text{coma,x}} \propto r^3 \cos(\theta), \quad
Z_{\text{coma,y}} \propto r^3 \sin(\theta)
$$

### Spherical
Spherical aberration makes the wavefront too steep near edges:

$$
Z_{\text{spher}} \propto 6r^4 - 6r^2 + 1
$$

---

## 3) Applying Zernike Corrections

The SLM applies a phase correction proportional to the wavefront:

$$
\phi_{\text{corr}}(r,\theta) = \frac{2\pi}{\lambda} W(r,\theta)
$$

This correction is added to the base vortex phase:

$$
\phi_{\text{total}} = \phi_{\text{vortex}} + \phi_{\text{steer}} + \phi_{\text{corr}}
$$

By tuning $c_n^m$, the algorithm compensates for aberrations and improves:
- Donut symmetry
- Dark core depth
- Centering stability

---

## 4) Why It Helps the Donut

Aberrations distort the phase and cause:
- **Off‑center dark core**
- **Asymmetric ring**
- **Reduced contrast**

Applying Zernike corrections restores symmetry and maximizes the contrast ratio used in the spherical optimization.

---

## 5) Suggested Images

- Zernike basis visualization  
  `Docs/images/zernike_basis.png`
- Astigmatism vs corrected  
  `Docs/images/astig_correction.png`
- Coma vs corrected  
  `Docs/images/coma_correction.png`
- Spherical vs corrected  
  `Docs/images/spherical_correction.png`

