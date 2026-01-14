import argparse
from slm_cls import SLM, SLMError
from PIL import Image
import numpy as np


def bmp_to_array(path):
    """Convert BMP to 8-bit grayscale 1D array (1272x1024)."""
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((1272, 1024))       # ensure correct resolution
    arr = np.array(img, dtype=np.uint8).flatten()
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Load a BMP file into the SLM using Write_FMemArray.")
    parser.add_argument("--bmp", required=True, help="Path to BMP file")
    parser.add_argument("--slot", required=True, type=int,
                        help="Slot number (0–818)")
    args = parser.parse_args()

    try:
        with SLM() as slm:
            print(f"Devices found: {slm._num_devices}")

            # Convert BMP to array
            array = bmp_to_array(args.bmp)

            # Write array to SLM frame memory
            slm.write_frame_array(array, 1272, 1024, args.slot)
            print(f"Loaded BMP '{args.bmp}' into slot {args.slot} (via array)")

            # Change display slot
            slm.change_display_slot(args.slot)
            print(f"Display changed to slot {args.slot}")

    except SLMError as e:
        print(f"SLM Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
