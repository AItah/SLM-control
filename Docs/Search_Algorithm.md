# Donut Optimization Search Algorithm

This document explains how the **fast search** works, the data flow, and the cost metrics used during optimization.

---

## 1) High‑Level Flow

1. **User selection**
   - Pick dark spot center (or refine automatically).
   - Draw donut circle (or auto‑create from radius).
2. **Single‑frame analysis (optional)**
   - Compute masked crop inside circle.
   - Plot angle lines + cross‑section.
3. **Optimization run**
   - For each selected scan mode (Shift → Astig → Coma → Spherical):
     - (Optional) reset parameters as configured.
     - Run fast search until convergence.
     - Update the best parameters on the Vortex Generator.
4. **Logging + CSV**
   - Each step logs the parameters and cost.
   - CSV saved with parameters, cost, and step size.

---

## 2) Fast Search Algorithm (Directional Pattern Search)

The optimizer uses **directional pattern search**:

### Core idea
- A **direction vector** is chosen from the angle lines.
- Move in that direction by the current step size.
- If cost decreases, keep moving in the same direction.
- If not, try the next direction.
- If none improve, **reduce step size** and repeat.

### Direction set
Directions are taken from the **same angles shown in the debug view**:

$$
\theta_i = \theta_0 + \frac{2\pi i}{N}, \quad i = 0,1,\dots,N-1
$$

Where:
- $N$ is the “Angles” setting.
- $\theta_0$ is the base direction (dark‑spot line).

### Step schedule
Initial step = user‑defined `X step` and `Y step`.  
Each time no direction improves the cost, steps shrink by:

$$
\text{step} \leftarrow \frac{\text{step}}{\text{shrink\_factor}}
$$

The algorithm stops after a final pass at **min step**.

---

## 3) Cost Functions

### A) Shift / Astig / Coma
The cost measures **donut symmetry** and **center leakage** from the cropped circle:

1. Sample radial profiles for multiple angles.
2. Normalize by max intensity.
3. Sum radial variance across angles.
4. Add center leakage penalty.

In code this behaves like:

$$
\text{cost} = \sum_r \sigma_r + 10 \cdot \bar{I}_{\text{center}}
$$

Where $\sigma_r$ is the variance across angles at radius $r$.

### B) Spherical (contrast‑ratio metric)
For spherical optimization we maximize the contrast between donut peak and dark center:

$$
\text{ratio} = \frac{I_{\max}-I_{\text{noise}}}{I_{\text{dark}}-I_{\text{noise}}}
$$

The optimizer **minimizes** cost by using:

$$
\text{cost} = \frac{1}{\text{ratio}}
$$

Where:
- $I_{\max}$ = max intensity inside the cropped donut region.
- $I_{\text{dark}}$ = minimum intensity near dark spot center.
- $I_{\text{noise}}$ = median of corner patches from full image.

---

## 4) Multi‑Mode Scans

When multiple modes are selected, they run sequentially:

1. **Shift**
2. **Astigmatism**
3. **Coma**
4. **Spherical**

Reset rules:
- If **Shift** is included: offsets + Zernikes reset to zero.
- If **Shift** not included: Zernikes reset to zero, offsets preserved.
- If only one mode selected: no reset performed.

---

## 5) Debug Outputs

The debug window shows:
- Masked crop inside circle.
- Angle lines (first line is the dark‑spot line).
- Cross‑section along first line.
- Cost metrics and diagnostics.

---

## 6) Images (Add Later)

Suggested images:

- **Algorithm flow diagram**
  - `Docs/images/search_flow.png`

- **Directional search diagram**
  - `Docs/images/search_directions.png`

- **Cost function illustration**
  - `Docs/images/cost_profile.png`

Use Markdown to include:

```
![Directional search](images/search_directions.png)
```

