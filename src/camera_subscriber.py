from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import numpy as np
from PIL import Image

try:
    import zmq
except Exception as exc:  # pragma: no cover - optional dependency
    zmq = None
    _ZMQ_IMPORT_ERROR = exc
else:
    _ZMQ_IMPORT_ERROR = None


@dataclass
class CameraSettings:
    endpoint: str = "tcp://127.0.0.1:5555"
    topic: str = "camera"
    status_topic: str = "camera/status"
    show_status: bool = True
    bind: bool = False
    rcv_hwm: int = 2
    poll_ms: int = 50


class CameraSubscriber:
    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self._context = None
        self._socket = None
        self._poller = None

    @property
    def available(self) -> bool:
        return zmq is not None

    @property
    def import_error(self) -> Optional[Exception]:
        return _ZMQ_IMPORT_ERROR

    def start(self) -> None:
        if zmq is None:
            raise RuntimeError(f"pyzmq is required: {_ZMQ_IMPORT_ERROR}")
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, max(1, int(self.settings.rcv_hwm)))
        if self.settings.bind:
            self._socket.bind(self.settings.endpoint)
        else:
            self._socket.connect(self.settings.endpoint)

        self._socket.setsockopt_string(zmq.SUBSCRIBE, self.settings.topic)
        if self.settings.show_status and self.settings.status_topic:
            self._socket.setsockopt_string(zmq.SUBSCRIBE, self.settings.status_topic)

        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        self._poller = None

    def poll(self) -> Optional[tuple[str, dict[str, Any], bytes]]:
        if self._socket is None or self._poller is None:
            return None
        events = dict(self._poller.poll(max(1, int(self.settings.poll_ms))))
        if self._socket not in events:
            return None
        parts = self._socket.recv_multipart()
        if len(parts) < 3:
            return None
        topic = parts[0].decode("utf-8", errors="ignore")
        try:
            metadata = json.loads(parts[1].decode("utf-8"))
        except Exception:
            metadata = {}
        payload = parts[2]
        return topic, metadata, payload


def _normalize_format(fmt: str) -> str:
    fmt = fmt.lower().strip()
    aliases = {
        "gray": "gray8",
        "grey": "gray8",
        "mono": "gray8",
        "mono8": "gray8",
        "l8": "gray8",
        "rgb": "rgb8",
        "bgr": "bgr8",
    }
    return aliases.get(fmt, fmt)


def decode_frame(metadata: dict[str, Any], payload: bytes) -> tuple[np.ndarray, str]:
    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))
    fmt = _normalize_format(str(metadata.get("format", "gray8")))
    compressed = bool(metadata.get("compressed", False))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame size in metadata.")

    if compressed:
        img = Image.open(BytesIO(payload))
        if fmt == "gray8":
            img = img.convert("L")
            return np.array(img), "gray8"
        img = img.convert("RGB")
        return np.array(img), "rgb8"

    expected_gray = width * height
    expected_color = width * height * 3

    if fmt == "gray8":
        if len(payload) == expected_gray:
            img = np.frombuffer(payload, dtype=np.uint8).reshape(height, width)
            return img, "gray8"
        if len(payload) == expected_color:
            img = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
            return img, "bgr8"
        raise ValueError(f"Payload size mismatch: {len(payload)} != {expected_gray}")

    if fmt in ("rgb8", "bgr8"):
        if len(payload) == expected_color:
            img = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
            return img, fmt
        if len(payload) == expected_gray:
            img = np.frombuffer(payload, dtype=np.uint8).reshape(height, width)
            return img, "gray8"
        raise ValueError(f"Payload size mismatch: {len(payload)} != {expected_color}")

    if len(payload) == expected_color:
        img = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
        return img, "bgr8"
    if len(payload) == expected_gray:
        img = np.frombuffer(payload, dtype=np.uint8).reshape(height, width)
        return img, "gray8"

    raise ValueError(f"Unsupported pixel format '{fmt}'.")
