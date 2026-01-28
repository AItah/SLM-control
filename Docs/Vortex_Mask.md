# Vortex Mask Overview

This document explains what a **vortex mask** is, why it creates a donut intensity pattern, and how the mask parameters affect the beam.

---

## 1) What is a Vortex Mask?

A vortex phase mask imposes a **helical phase** on the wavefront:

$$
E(r,\theta) = A(r)\,e^{i \ell \theta}
$$

Where:
- $A(r)$ is the amplitude profile of the input beam.
- $\theta$ is the azimuthal angle.
- $\ell$ is the **topological charge** (integer).

The phase singularity at $r=0$ forces destructive interference at the center, forming a **dark core**.

---

## 2) Why Does It Produce a Donut?

The helical phase means the wavefront has a $2\pi \ell$ phase winding around the center.  
At the center, the phase is undefined → the field cancels → **zero intensity**.  
Energy is pushed into a ring, forming the donut.

The donut radius depends on:
- **Topological charge** $\ell$
- **Beam waist** and optical system (NA, focal length)
- **Aperture size**

---

## 3) Key Parameters

### Topological charge ($\ell$)
Higher $\ell$ means more phase winding and a **larger dark core**:

- $\ell = 1$: small donut.
- $\ell = 2$: larger donut.
- Larger $\ell$ → wider ring.

### Offsets (X/Y)
Shifts the vortex center relative to the optical axis.  
Used for alignment when the donut is not centered on the camera.

### Aperture
Applies a circular aperture:

$$
E(r,\theta) \leftarrow E(r,\theta) \cdot \Pi\!\left(\frac{r}{R}\right)
$$

Where $\Pi$ is 1 inside radius $R$ and 0 outside.

### Steering (Fork Grating)
Adds a linear or angular phase term to steer the beam.

---

## 4) Physics Notes

The vortex mask is **pure phase**: it does not block light, it rearranges it.  
This is efficient and preserves power (aside from system losses).

The donut is usually observed at the **focal plane** of a lens where the vortex phase is converted into an annular intensity distribution.

---

## 5) Suggested Images

- Helical phase map  
  `Docs/images/vortex_phase.png`
- Donut intensity profile  
  `Docs/images/donut_intensity.png`
- Effect of different $\ell$  
  `Docs/images/ell_comparison.png`

Example markdown:

```
![Vortex phase](images/vortex_phase.png)
```

