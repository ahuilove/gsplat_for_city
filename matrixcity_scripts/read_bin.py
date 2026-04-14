#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read COLMAP `images.bin` files.

This script parses the binary image records written by COLMAP and prints a
human-readable summary. It can also export the parsed content to JSON.

Example:
    python matrixcity_scripts/read_bin.py /path/to/sparse/0/images.bin
    python matrixcity_scripts/read_bin.py /path/to/images.bin --json out.json
"""

import argparse
import json
import struct
from pathlib import Path
from typing import BinaryIO, Dict, List


def read_next_bytes(fid: BinaryIO, num_bytes: int, fmt: str):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(
            f"Unexpected end of file: expected {num_bytes} bytes, got {len(data)}"
        )
    return struct.unpack("<" + fmt, data)


def read_c_string(fid: BinaryIO) -> str:
    """Read a null-terminated UTF-8 string."""
    chars: List[bytes] = []
    while True:
        char = fid.read(1)
        if char == b"":
            raise EOFError("Unexpected end of file while reading string")
        if char == b"\x00":
            return b"".join(chars).decode("utf-8")
        chars.append(char)


def read_images_binary(path: Path) -> Dict[int, dict]:
    """
    Parse COLMAP images.bin.

    Format per image:
    - image_id: int32
    - qvec: 4 * float64
    - tvec: 3 * float64
    - camera_id: int32
    - name: null-terminated string
    - num_points2D: uint64
    - points2D: repeated (x: float64, y: float64, point3D_id: int64)
    """
    images = {}
    with path.open("rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            image_props = read_next_bytes(fid, 64, "idddddddi")
            image_id = image_props[0]
            qvec = list(image_props[1:5])
            tvec = list(image_props[5:8])
            camera_id = image_props[8]
            name = read_c_string(fid)

            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            xys = []
            point3d_ids = []
            for _ in range(num_points2d):
                x, y, point3d_id = read_next_bytes(fid, 24, "ddq")
                xys.append([x, y])
                point3d_ids.append(point3d_id)

            images[image_id] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "num_points2D": num_points2d,
                "xys": xys,
                "point3D_ids": point3d_ids,
            }
    return images


def summarize(images: Dict[int, dict], max_items: int) -> None:
    """Print a concise summary to stdout."""
    print(f"Loaded {len(images)} image records.")
    for image_id in sorted(images)[:max_items]:
        image = images[image_id]
        print(
            f"[{image_id}] name={image['name']} camera_id={image['camera_id']} "
            f"qvec={image['qvec']} tvec={image['tvec']} "
            f"points2D={image['num_points2D']}"
        )
    if len(images) > max_items:
        print(f"... truncated {len(images) - max_items} additional records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read COLMAP images.bin")
    parser.add_argument("images_bin", type=Path, help="Path to COLMAP images.bin")
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="Optional path to save parsed output as JSON",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Number of records to print in the console summary",
    )
    args = parser.parse_args()

    if not args.images_bin.exists():
        raise FileNotFoundError(f"images.bin not found: {args.images_bin}")

    images = read_images_binary(args.images_bin)
    summarize(images, max_items=args.max_items)

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        with args.json_path.open("w", encoding="utf-8") as f:
            json.dump(images, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON to {args.json_path}")


if __name__ == "__main__":
    main()
