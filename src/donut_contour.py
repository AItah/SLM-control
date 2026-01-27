import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from skimage import segmentation, color, filters

def get_file():
    root = tk.Tk(); root.withdraw()
    return filedialog.askopenfilename()

def mouse_callback(event, x, y, flags, param):
    global center, img
    if event == cv2.EVENT_LBUTTONDOWN:
        if center is None:
            center = (x, y)
        else:
            radius = int(np.linalg.norm(np.array(center) - np.array((x, y))))
            run_morphological_snake(center, radius)
            center = None

def run_morphological_snake(c, r):
    global img
    # 1. Grayscale conversion (Preserving original resolution)
    gray = color.rgb2gray(img)
    
    # 2. Edge Map Calculation
    # Increase alpha to 200 to make the gradient 'snappier' like MATLAB
    gimage = segmentation.inverse_gaussian_gradient(gray, alpha=200, sigma=1.0)

    # 3. Create initial mask
    mask = np.zeros(gray.shape, dtype=np.int8)
    cv2.circle(mask, c, r, 1, -1)

    # 4. Morphological Geodesic Active Contour
    # FIXED: Changed num_iters to num_iter based on your traceback
    res = segmentation.morphological_geodesic_active_contour(
        gimage, 
        num_iter=500, 
        init_level_set=mask,
        smoothing=2, 
        threshold='auto', 
        balloon=1
    )

    # 5. Boundary Visualization
    display = img.copy()
    contours, _ = cv2.findContours(res.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if contours:
        # We sort by area to ensure we draw the main donut shape
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        diameter = 2 * np.sqrt(area / np.pi)
        
        # Draw the yellow 'MATLAB-style' boundary
        cv2.drawContours(display, [cnt], -1, (0, 255, 255), 2)
        
        label = f"Diameter: {diameter:.2f} px"
        cv2.putText(display, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Scientific Analysis", display)

# --- Main Setup ---
center = None
file_path = get_file()
if file_path:
    img = cv2.imread(file_path)
    if img is None:
        print("Error: File not found.")
    else:
        # Keep original image size
        cv2.namedWindow("Scientific Analysis", cv2.WINDOW_NORMAL) 
        cv2.setMouseCallback("Scientific Analysis", mouse_callback)
        
        print("INSTRUCTIONS:")
        print("Click 1: Center of dark spot")
        print("Click 2: Edge of the yellow glow")
        cv2.imshow("Scientific Analysis", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()