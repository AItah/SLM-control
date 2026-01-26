from __future__ import annotations

from PySide6 import QtCore

from slm_params import SlmParams


class SlmParamsStore(QtCore.QObject):
    changed = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._params: SlmParams | None = None

    def set_params(self, params: SlmParams) -> None:
        self._params = params
        self.changed.emit(params)

    def get(self) -> SlmParams | None:
        return self._params
