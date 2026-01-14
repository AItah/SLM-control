import argparse
import os

INPUT_FILE = "requirements.txt"
OUTPUT_FILE = "pack_ready_4pip.txt"

MODE_LATEST = "LATEST"
MODE_EQUAL = "EQUAL"
MODE_MIN = "MIN"
VALID_MODES = (MODE_LATEST, MODE_EQUAL, MODE_MIN)

_VERSION_OPERATORS = ("===", "==", "~=", ">=", "<=", "!=", ">", "<")


def _split_marker(spec: str):
    if ";" in spec:
        req, marker = spec.split(";", 1)
        return req.strip(), marker.strip()
    return spec.strip(), ""


def _find_operator(req: str):
    for op in _VERSION_OPERATORS:
        idx = req.find(op)
        if idx != -1:
            return op
    return None


def _strip_version_spec(spec: str) -> str:
    req, marker = _split_marker(spec)
    name = req
    for op in _VERSION_OPERATORS:
        idx = req.find(op)
        if idx != -1:
            name = req[:idx].strip()
            break

    if marker:
        return f"{name}; {marker}"
    return name


def _normalize_line(line: str, mode: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return line
    if stripped.startswith("-"):
        return line
    if " @ " in stripped or "://" in stripped:
        return line

    if "#" in line:
        body, comment = line.split("#", 1)
    else:
        body, comment = line, ""

    body = body.strip()
    comment_suffix = f"#{comment}" if comment else ""

    op = _find_operator(body)
    if op:
        if mode == MODE_LATEST:
            return f"{_strip_version_spec(body)}{comment_suffix}\n"
        if mode == MODE_MIN and op in ("==", "==="):
            req, marker = _split_marker(body)
            name = req.split(op, 1)[0].strip()
            version = req.split(op, 1)[1].strip()
            marker_suffix = f"; {marker}" if marker else ""
            return f"{name}>={version}{marker_suffix}{comment_suffix}\n"
        return f"{body}{comment_suffix}\n"

    req_part, marker = _split_marker(body)
    marker_suffix = f"; {marker}" if marker else ""

    parts = req_part.split()
    if len(parts) >= 2:
        name, version = parts[0], parts[1]
        if mode == MODE_LATEST:
            return f"{name}{marker_suffix}{comment_suffix}\n"
        if mode == MODE_MIN:
            return f"{name}>={version}{marker_suffix}{comment_suffix}\n"
        return f"{name}=={version}{marker_suffix}{comment_suffix}\n"

    return f"{req_part}{marker_suffix}{comment_suffix}\n"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a pip-ready requirements file from a space-separated list."
    )
    parser.add_argument("--input", default=INPUT_FILE, help="Path to requirements file.")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Path to output file.")
    parser.add_argument(
        "--mode",
        default=MODE_EQUAL,
        type=str.upper,
        choices=VALID_MODES,
        help="Version mode: LATEST, EQUAL, or MIN.",
    )
    return parser.parse_args()


def prepare_requirements(input_path=INPUT_FILE, output_path=OUTPUT_FILE, mode=MODE_EQUAL):
    if not os.path.exists(input_path):
        raise SystemExit(f"Requirements file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    normalized = [_normalize_line(line, mode) for line in lines]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.writelines(normalized)

    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    args = _parse_args()
    prepare_requirements(input_path=args.input, output_path=args.output, mode=args.mode)
