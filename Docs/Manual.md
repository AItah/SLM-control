# SLM-control User Manual

## Overview
SLM-control provides a GUI workflow to configure an SLM, generate vortex masks, view the camera stream, and optimize donut alignment via automated scans. The UI is split into windows:

- **SLM Control** (main window)
- **SLM Parameters**
- **Vortex Generator**
- **Camera Viewer**
- **Donut Optimization Wizard** + **Debug Window**

Settings (window positions and control values) are persisted to an INI file under `settings/` in the project root.

---

## 1) SLM Control (Main Window)
The main hub for hardware connection and other windows.

### Connection
- **Connect SLM**: Open the device connection.
- **Disconnect**: Close the device connection.

### Windows
- **SLM Parameters**: Show/hide the SLM parameter editor.
- **Vortex Generator**: Show/hide the vortex mask generator.
- **Camera Viewer**: Show/hide the camera viewer.

### BMP / Slot Controls
- **Current BMP**: Path to a BMP mask file.
- **Browse...**: Select BMP file.
- **Slot**: SLM slot index used for send/read operations.
- **Send BMP to slot**: Write the BMP into the selected slot.
- **Change display slot**: Display the selected slot on the SLM.

### Diagnostics
- **Read temperature**: Query the SLM head temperature.
- **Check DISPLAYED image**: Read back the current displayed image.
- **Check FMEM image**: Read back the image stored in the selected slot.

### Temperature Watchdog
- **Temperature watchdog**: Periodically checks temperature.
- **Interval**: Polling interval (seconds).

---

## 2) SLM Parameters Window
Edit and load core SLM parameters.

### File Controls
- **SLM params JSON**: Path to JSON file with SLM parameters.
- **Browse...**: Select a JSON file.
- **Load**: Load parameters from the file.

### Parameters
- **Nx, Ny**: SLM pixel dimensions.
- **px_side_m / py_side_m**: Pixel pitch (meters).
- **Fill factor (%)**: Fill factor of the SLM.
- **c2pi2unit**: Phase scaling; 2π corresponds to this integer.

---

## 3) Vortex Generator Window
Generate vortex masks and optionally apply steering or Zernike corrections.

### Vortex
- **Wavelength (nm)**: Laser wavelength.
- **ell**: Topological charge.
- **Offset X/Y (mm)**: Mask center offsets.
- **Aperture radius (mm)**: Optional circular aperture.
- **Force pixel-zero center**: Snap vortex center to an integer pixel.

### Calibration
- **Calibration mask**: Optional external calibration mask path.
- **Browse...**: Select calibration mask file.

### Steering
- **Use fork grating**: Enable steering via fork grating.
- **Steering mode**: `none`, `angle`, or `shift`.
- **theta_x / theta_y (deg)**: Steering angles (when mode is `angle`).
- **delta_x / delta_y (mm)**: Steering shifts (when mode is `shift`).
- **Focal length (m)**: Used for steering calculations.

### Zernike Corrections
- **Enable Zernike corrections**: Toggle corrections.
- **Astig V / Astig O**: Astigmatism coefficients.
- **Coma Y / Coma X**: Coma coefficients.
- **Spherical**: Spherical coefficient.

### Actions
- **Slot**: Choose target slot (-1 = save to file only).
- **Generate Vortex Mask**: Create and send/save the mask.
- **Donut Optimization...**: Open the donut optimization workflow.

---

## 4) Camera Viewer
Receives and displays the camera stream.

### Connection
- **Endpoint**: ZMQ endpoint address.
- **Topic**: Image topic.
- **Status topic**: Status topic.
- **Show status**: Overlay status text on viewer.
- **Bind**: Bind to endpoint instead of connecting.
- **RCV HWM**: Receive high-water mark for the subscriber.
- **Poll**: Poll interval (ms).

### Streaming
- **Start**: Begin receiving frames.
- **Stop**: Stop receiving frames.

### Viewer Filter
- **Zero below threshold**: Apply a viewer-only threshold (pixels below the value are zeroed).
- **Threshold**: The value used by the filter.

### Selection (used by Donut Optimization)
When the Donut Optimization Wizard asks you to select a point or draw a circle, you do it on the camera viewer:
- **Pick dark spot**: Click a point.
- **Draw donut circle**: Click/drag to set radius; drag to move or resize.

---

## 5) Donut Optimization Wizard
Guided workflow to align the donut pattern and optimize parameters.

### Manual Target
- **Pick dark spot**: Click the dark spot in the camera view.
- **Draw donut circle**: Draw the donut circle in the camera view.
- **Refine dark spot**: Refine the selected dark spot using image processing.
- **Donut analysis**: Analyze a single frame and populate the debug window.
- **Camera pixel size (um/px)**: Used for cross-section scaling and circle conversion.
- **Zero below threshold**: Threshold applied before analysis and scanning.
- **Auto circle radius (mm)**: If no circle is drawn, a circle is created centered on the dark spot using this radius.
- **Clear**: Clear manual selections (dark spot + circle).

### Scan Settings
- **X/Y range ±**: Kept for compatibility; fast search uses step size and directions.
- **X/Y step**: Base step sizes for the scan.
- **Settle time**: Wait time after updating the SLM.
- **SLM slot**: Slot used for mask updates during scanning.
- **Angles**: Number of direction angles to test (also used for debug angle lines).
- **Fast min step**: Minimum allowed step size.
- **Shrink factor**: Step reduction factor each time the search tightens.

### Scan Modes (multi-select)
Select one or more modes to run sequentially in this order:
1. **Shift**
2. **Astigmatism**
3. **Coma**
4. **Spherical**

Reset behavior:
- If multiple modes are selected and **Shift** is included: offsets + all Zernike values are reset to zero before scanning.
- If multiple modes are selected and **Shift** is NOT included: Zernike values reset to zero; offsets are preserved.
- If only one mode is selected: no reset is performed.

### Execution
- **Start**: Runs the selected scans in order.
- **Stop**: Cancels current scan.
- **Progress**: Progress indicator.
- **Log**: Detailed output of scan steps and costs.

### Debug View
- **Enable debug view**: Enable debug plots during scanning.
- **Open Debug Window**: Opens the debug visualization.

---

## 6) Donut Optimization Debug Window
Visual and numeric diagnostics for a single frame or scan step.

### Panels
- **Circle crop (masked)**: The cropped donut region (outside circle masked).
- **Angle lines**: The sampled directions (yellow lines, red first line).
- **Cross-section**: Intensity profile along the first line.
- **Debug data**: Numeric values such as center, radius, cost, thresholds, dark spot metrics, and spherical ratio values.

---

## Common Workflows

### Generate a Vortex Mask
1. Open **SLM Parameters** and load the JSON file.
2. Open **Vortex Generator** and set wavelength, ell, offsets, aperture, and steering.
3. Click **Generate Vortex Mask**.

### Donut Optimization (Shift Only)
1. Start **Camera Viewer**.
2. In **Donut Optimization Wizard**, click **Pick dark spot** and click the dark spot.
3. Draw the donut circle or set **Auto circle radius** and click **Start**.
4. Select only **Shift** if you want offset optimization only.

### Multi-Mode Optimization
1. Select multiple scan modes (Shift → Astig → Coma → Spherical).
2. Click **Start**; each mode will run in order.

---

## Notes
- Settings (window positions and control values) are persisted in `settings/`.
- The fast search uses directional steps aligned to the angle lines.
