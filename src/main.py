from __future__ import annotations

import sys

from PySide6 import QtWidgets

from slm_control_window import SlmControlWindow
from slm_params_window import SlmParamsWindow
from slm_store import SlmParamsStore


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)

    store = SlmParamsStore()
    params_window = SlmParamsWindow(store)
    control_window = SlmControlWindow(store)

    params_window.show()
    control_window.show()

    app.aboutToQuit.connect(control_window.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
