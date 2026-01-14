import numpy as np
from PIL import Image


def gaussian_circle_image(
    width=512,
    height=512,
    radius=150,
    center=None,          # (cx, cy); defaults to image center
    sigma=None,           # Gaussian spread; defaults to radius/2
    background=0,         # 0..255 grayscale value outside the circle
    invert=False,         # True -> dark center, bright edge/background
    filename="circle_gaussian.png"
):
    # Center defaults to image center
    if center is None:
        center = (width // 2, height // 2)
    cx, cy = center

    # Sensible default for sigma (controls how fast it fades)
    if sigma is None:
        sigma = radius / 2.0

    # Coordinate grid
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Circle mask
    mask = r <= radius

    # Radial Gaussian (peak at r=0)
    g = np.exp(-(r**2) / (2.0 * sigma**2))

    # Keep only inside the circle
    g_circle = np.zeros_like(g)
    g_circle[mask] = g[mask]

    # Normalize intensities of the circle region to full 0..255
    # (so your circle uses the whole 8-bit range regardless of sigma)
    if np.any(mask):
        vals = g_circle[mask]
        vmin, vmax = vals.min(), vals.max()
        # avoid divide-by-zero if sigma is extremely small
        if vmax > vmin:
            g_circle[mask] = (vals - vmin) / (vmax - vmin)
        else:
            g_circle[mask] = 1.0

    # Map to 0..255 and set background
    img = np.full((height, width), fill_value=background, dtype=np.float32)
    circle_values = g_circle * 255.0
    if invert:
        circle_values = 255.0 - circle_values
    img[mask] = circle_values[mask]

    # Clip and convert to 8-bit
    img8 = np.clip(img, 0, 255).astype(np.uint8)

    # Save
    Image.fromarray(img8, mode="L").save(filename)
    print(f"Wrote {filename}")


if __name__ == "__main__":
    # Example usage
    gaussian_circle_image(
        width=1272,
        height=1024,
        radius=1200,
        center=None,        # center of the image
        # center=(int((3/4)*1272/2.0), 1024/2),        # center of the image
        sigma=200,         # defaults to radius/2
        background=0,       # black background
        invert=False,       # bright center, dark edge
        filename="circle_gaussian_offset_.bmp"
    )
