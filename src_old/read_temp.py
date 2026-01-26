from slm_cls import SLM
from pathlib import Path 

def main():
    with SLM() as slm:
        print("Devices found:", slm._num_devices)
        print("Head Serial:", slm.get_head_serial())

        head, cb = slm.get_temperature()
        print(f"Head Temp = {head:.2f} °C, Controller Temp = {cb:.2f} °C")


if __name__ == "__main__":
    main()
