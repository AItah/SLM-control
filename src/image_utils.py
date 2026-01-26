from __future__ import annotations

import traceback
from typing import Tuple

import numpy as np
from PIL import Image
from PySide6 import QtGui


def to_grayscale_qpixmap(arr_2d: np.ndarray) -> QtGui.QPixmap:
    arr = np.ascontiguousarray(arr_2d.astype(np.uint8))
    h, w = arr.shape
    qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qimg.copy())


def image_path_to_array(path: str, size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(size, resample=Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def exception_to_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
