from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SlmParams:
    nx: int
    ny: int
    px_side_m: float
    py_side_m: float
    fill_factor_percent: float | None
    c2pi2unit: int
    source_path: Path


def load_slm_params(path: str | Path) -> SlmParams:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    fill = data.get("Fill_factor_percent")
    if fill is None:
        fill = data.get("fill_factor_percent")

    return SlmParams(
        nx=int(data["Nx"]),
        ny=int(data["Ny"]),
        px_side_m=float(data["px_side_m"]),
        py_side_m=float(data["py_side_m"]),
        fill_factor_percent=float(fill) if fill is not None else None,
        c2pi2unit=int(data["c2pi2unit"]),
        source_path=path,
    )
