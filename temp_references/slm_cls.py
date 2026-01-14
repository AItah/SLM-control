import ctypes
from ctypes import c_int32, c_uint8, c_double, c_uint32, POINTER, c_char_p
from pathlib import Path 

class SLMError(Exception):
    """Custom exception for SLM DLL errors."""
    pass

class SLM:
    def __init__(self, dll_path="", max_devices=8):
        if dll_path == "":
            dll_path = Path(__file__).parent / 'USB_control_SDK' / 'hpkSLMdaLV_cdecl_64bit' / 'hpkSLMdaLV.dll'

        # Load DLL
        self._dll = ctypes.cdll.LoadLibrary(dll_path)
        self._max_devices = max_devices
        self._bIDList = (c_uint8 * self._max_devices)()
        self._num_devices = 0
        self._open = False
        self._configure_functions()

    def _configure_functions(self):
        d = self._dll

        # Connection
        d.Open_Dev.argtypes = [POINTER(c_uint8), c_int32]
        d.Open_Dev.restype = c_int32

        d.Close_Dev.argtypes = [POINTER(c_uint8), c_int32]
        d.Close_Dev.restype = c_int32

        # Device info
        d.Check_HeadSerial.argtypes = [c_uint8, c_char_p, c_int32]
        d.Check_HeadSerial.restype = c_int32

        # Memory writing
        d.Write_FMemBMPPath.argtypes = [c_uint8, c_char_p, c_uint32]
        d.Write_FMemBMPPath.restype = c_int32

        d.Write_FMemArray.argtypes = [c_uint8, POINTER(c_uint8), c_int32, c_uint32, c_uint32, c_uint32]
        d.Write_FMemArray.restype = c_int32

        d.Change_DispSlot.argtypes = [c_uint8, c_uint32]
        d.Change_DispSlot.restype = c_int32

        # Temperature
        d.Check_Temp.argtypes = [c_uint8, POINTER(c_double), POINTER(c_double)]
        d.Check_Temp.restype = c_int32

        # Memory reading
        d.Check_FMem_Slot.argtypes = [c_uint8, c_uint32, c_uint32, c_uint32, c_uint32, POINTER(c_uint8)]
        d.Check_FMem_Slot.restype = c_int32

        d.Check_Disp_IMG.argtypes = [c_uint8, c_uint32, c_uint32, c_uint32, POINTER(c_uint8)]
        d.Check_Disp_IMG.restype = c_int32

        # Mode
        d.Mode_Select.argtypes = [c_uint8, c_uint8]
        d.Mode_Select.restype = c_int32

        d.Mode_Check.argtypes = [c_uint8, POINTER(c_uint32)]
        d.Mode_Check.restype = c_int32

        # LED / IO
        d.Check_LED.argtypes = [c_uint8, POINTER(c_uint32)]
        d.Check_LED.restype = c_int32

        d.Check_IO.argtypes = [c_uint8, POINTER(c_uint32)]
        d.Check_IO.restype = c_int32

        # Reboot
        d.Reboot.argtypes = [c_uint8]
        d.Reboot.restype = None

        # SD card
        d.Write_SDBMPPath.argtypes = [c_uint8, c_char_p, c_uint32]
        d.Write_SDBMPPath.restype = c_int32

        d.Write_SDArray.argtypes = [c_uint8, POINTER(c_uint8), c_int32, c_uint32, c_uint32, c_uint32]
        d.Write_SDArray.restype = c_int32

        d.Check_SD_Slot.argtypes = [c_uint8, c_uint32, c_uint32, c_uint32, c_uint32, POINTER(c_uint8)]
        d.Check_SD_Slot.restype = c_int32

        d.Upload_from_SD_to_FMem.argtypes = [c_uint8, c_uint32, c_uint32]
        d.Upload_from_SD_to_FMem.restype = c_int32

    # --- Resource Management ---
    def open(self):
        num = self._dll.Open_Dev(self._bIDList, self._max_devices)
        if num <= 0:
            raise SLMError("No SLM devices found")
        self._num_devices = num
        self._open = True
        return num

    def close(self):
        if self._open:
            self._dll.Close_Dev(self._bIDList, self._num_devices)
            self._open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # --- Device Methods ---
    def get_head_serial(self, device_index=0):
        buf = ctypes.create_string_buffer(11)
        bID = self._bID(device_index)
        if self._dll.Check_HeadSerial(bID, buf, 11) != 1:
            raise SLMError("Failed to read head serial")
        return buf.value.decode("ascii")

    def write_frame_bmp(self, path, slot, device_index=0):
        bID = self._bID(device_index)
        if self._dll.Write_FMemBMPPath(bID, path.encode("utf-8"), slot) != 1:
            raise SLMError("Failed to write frame BMP")

    def write_frame_array(self, array, xpix, ypix, slot, device_index=0):
        bID = self._bID(device_index)
        arr = (c_uint8 * len(array))(*array)
        if self._dll.Write_FMemArray(bID, arr, len(array), xpix, ypix, slot) != 1:
            raise SLMError("Failed to write frame array")

    def change_display_slot(self, slot, device_index=0):
        bID = self._bID(device_index)
        if self._dll.Change_DispSlot(bID, slot) != 1:
            raise SLMError("Failed to change display slot")

    def get_temperature(self, device_index=0):
        head, cb = c_double(), c_double()
        bID = self._bID(device_index)
        if self._dll.Check_Temp(bID, ctypes.byref(head), ctypes.byref(cb)) != 1:
            raise SLMError("Failed to read temperature")
        return head.value, cb.value

    def read_frame_memory(self, slot, xpix, ypix, device_index=0):
        arrsize = xpix * ypix
        arr = (c_uint8 * arrsize)()
        bID = self._bID(device_index)
        if self._dll.Check_FMem_Slot(bID, arrsize, xpix, ypix, slot, arr) != 1:
            raise SLMError("Failed to read frame memory")
        return list(arr)

    def read_display_image(self, xpix, ypix, device_index=0):
        arrsize = xpix * ypix
        arr = (c_uint8 * arrsize)()
        bID = self._bID(device_index)
        if self._dll.Check_Disp_IMG(bID, arrsize, xpix, ypix, arr) != 1:
            raise SLMError("Failed to read display image")
        return list(arr)

    def set_mode(self, mode, device_index=0):
        bID = self._bID(device_index)
        if self._dll.Mode_Select(bID, mode) != 1:
            raise SLMError("Failed to set mode")

    def get_mode(self, device_index=0):
        mode = c_uint32()
        bID = self._bID(device_index)
        if self._dll.Mode_Check(bID, ctypes.byref(mode)) != 1:
            raise SLMError("Failed to check mode")
        return mode.value

    def get_led_status(self, device_index=0):
        status = c_uint32()
        bID = self._bID(device_index)
        if self._dll.Check_LED(bID, ctypes.byref(status)) != 1:
            raise SLMError("Failed to check LED")
        return status.value

    def get_io_status(self, device_index=0):
        status = c_uint32()
        bID = self._bID(device_index)
        if self._dll.Check_IO(bID, ctypes.byref(status)) != 1:
            raise SLMError("Failed to check IO")
        return status.value

    def reboot(self, device_index=0):
        bID = self._bID(device_index)
        self._dll.Reboot(bID)

    def write_sd_bmp(self, path, slot, device_index=0):
        bID = self._bID(device_index)
        if self._dll.Write_SDBMPPath(bID, path.encode("utf-8"), slot) != 1:
            raise SLMError("Failed to write SD BMP")

    def write_sd_array(self, array, xpix, ypix, slot, device_index=0):
        bID = self._bID(device_index)
        arr = (c_uint8 * len(array))(*array)
        if self._dll.Write_SDArray(bID, arr, len(array), xpix, ypix, slot) != 1:
            raise SLMError("Failed to write SD array")

    def read_sd_slot(self, slot, xpix, ypix, device_index=0):
        arrsize = xpix * ypix
        arr = (c_uint8 * arrsize)()
        bID = self._bID(device_index)
        if self._dll.Check_SD_Slot(bID, arrsize, xpix, ypix, slot, arr) != 1:
            raise SLMError("Failed to read SD slot")
        return list(arr)

    def upload_sd_to_fmem(self, sd_slot, fmem_slot, device_index=0):
        bID = self._bID(device_index)
        if self._dll.Upload_from_SD_to_FMem(bID, sd_slot, fmem_slot) != 1:
            raise SLMError("Failed to upload SD to FMem")

    # --- Helpers ---
    def _bID(self, device_index):
        if not self._open:
            raise SLMError("Device not open")
        if device_index >= self._num_devices:
            raise SLMError("Invalid device index")
        return self._bIDList[device_index]
