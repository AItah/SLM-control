# SLM-control

## installation
- install packages
1. run prepare_requirements_for_pip.py
```
python prepare_requirements_for_pip.py --mode LATEST
```
optional to use different modes: EQUAL or MIN

2. in the CLI (while in the venv) run: </br>
```
pip install -r pack_ready_4pip.txt
```

3. review .. \Docs\L55-0491_USB_Control_SDK.pdf

generally:
To control the SLM, the code loads the Hamamatsu SDK DLL via ctypes and expects the SDK files to be present locally.
DLL used: hpkSLMdaLV.dll (loaded in slm_cls.py); default location is /SLM_SDK/hpkSLMdaLV.dll .
Companion files in the same folder: hpkSLMdaLV.ini (and the other SDK support files shipped there).
You need the Hamamatsu SLM driver/SDK installed so the device is recognized by Windows; otherwise Open_Dev will fail with “No SLM devices found.”
Use 64-bit Python to match the 64-bit DLL (the one in src/SLM_SDK).
</br>





