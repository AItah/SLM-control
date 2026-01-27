import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def get_donut_cost(img_gray, center, max_r=100, num_angles=72, num_pts=100):
    """
    Calculates the asymmetry and leakage of a vortex donut.
    Lower score = more perfect donut.
    """
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    profiles = np.zeros((num_pts, num_angles))

    # 1. Radial Sampling (Vectorized where possible)
    for i, theta in enumerate(angles):
        # Coordinates from center outward
        x_indices = cx + np.linspace(0, max_r * np.cos(theta), num_pts)
        y_indices = cy + np.linspace(0, max_r * np.sin(theta), num_pts)
        
        # Sampling pixels
        for j in range(num_pts):
            px, py = int(x_indices[j]), int(y_indices[j])
            if 0 <= py < img_gray.shape[0] and 0 <= px < img_gray.shape[1]:
                profiles[j, i] = img_gray[py, px]

    # 2. Normalize Intensity to handle power fluctuations
    max_val = np.max(profiles)
    if max_val > 0:
        profiles = profiles / max_val

    # 3. Calculate Asymmetry Score (Std Dev across angles)
    # A perfect donut has identical profiles at every angle
    radial_variance = np.std(profiles, axis=1)
    asymmetry_score = np.sum(radial_variance)

    # 4. Calculate Center Leakage (Hole Quality)
    # Average intensity in the inner 10% of the radius
    hole_idx = int(num_pts * 0.1)
    center_leakage = np.mean(profiles[:hole_idx, :])

    # 5. Combined Cost (Weighting asymmetry and darkness)
    return asymmetry_score + (center_leakage * 10.0)

def select_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Select Donut Image")
    root.destroy()
    return path

if __name__ == "__main__":
    # 1. Load File
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting.")
        exit()

    full_img = cv2.imread(file_path)
    if full_img is None:
        print("Could not decode image.")
        exit()

    # 2. Select Crop Region (ROI)
    # Drag a box around the donut and press ENTER
    print("INSTRUCTION: Drag a box around the donut and press ENTER.")
    roi = cv2.selectROI("Step 1: Crop Donut", full_img, fromCenter=False)
    cv2.destroyWindow("Step 1: Crop Donut")
    
    x, y, w, h = roi
    img_crop = full_img[y:y+h, x:x+w]
    gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

    # 3. Select Donut Center
    # Click the exact center of the dark hole
    center_pt = []
    def click_center(event, x_clk, y_clk, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            center_pt.append((x_clk, y_clk))
            print(f"Center selected at ROI coords: {x_clk, y_clk}")

    cv2.imshow("Step 2: Click the Center of the Hole", img_crop)
    cv2.setMouseCallback("Step 2: Click the Center of the Hole", click_center)
    
    print("INSTRUCTION: Click the center of the dark spot, then press any key.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if center_pt:
        # 4. Calculate Cost
        # max_r should be roughly the distance to the outer edge of the glow
        final_center = center_pt[0]
        
        # Estimate max_r as half the crop width unless specified
        est_r = w // 2 
        
        cost = get_donut_cost(gray_crop, final_center, max_r=est_r)
        
        print("\n" + "="*30)
        print(f"ANALYSIS RESULTS")
        print(f"Center (Local): {final_center}")
        print(f"Calculated Cost: {cost:.4f}")
        print("="*30)
        print("Note: Lower cost = higher symmetry and darker center.")
    else:
        print("No center selected.")