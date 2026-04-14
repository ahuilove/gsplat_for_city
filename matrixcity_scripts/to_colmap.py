#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert MatrixCity small_city aerial train block_1 to a COLMAP-style dataset.

Output:
    /home/zhanqh/dataset/city/Matrixcity/small_city/block_1_colmap/
        images/
        sparse/0/cameras.bin
        sparse/0/images.bin
        sparse/0/points3D.bin
        sparse/0/project.ini

Notes:
- 0000.png is skipped because it has no pose.
- MatrixCity rot_mat appears scaled by 0.01; this script detects that and multiplies by 100.
- MatrixCity pose is NeRF/OpenGL-style c2w; converted here to COLMAP/OpenCV-style w2c.
- Images are symlinked by default to save space. Falls back to copy if symlink fails.
"""

import json
import math
import os
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from plyfile import PlyData
except ImportError as e:
    raise ImportError(
        "Please install plyfile first: pip install plyfile"
    ) from e


# =========================
# Paths
# =========================
BLOCK_DIR = Path("/home/zhanqh/dataset/city/Matrixcity/small_city/aerial/train/block_1")
TRANSFORMS_PATH = BLOCK_DIR / "transforms.json"
PLY_PATH = Path("/home/zhanqh/dataset/city/Matrixcity/small_city_pointcloud/point_cloud/aerial/Block_A.ply")
OUT_DIR = Path("/home/zhanqh/dataset/city/Matrixcity/small_city/block_1_colmap")

# =========================
# Options
# =========================
USE_SYMLINK = True          # symlink images instead of copying
MAX_POINTS = 200000         # set to None to keep all points
RANDOM_SEED = 42


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion [qw, qx, qy, qz]."""
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0.0, 0.0, 0.0],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0.0, 0.0],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0.0],
        [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[:, np.argmax(eigvals)]
    qvec = qvec[[3, 0, 1, 2]]  # -> qw, qx, qy, qz
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if USE_SYMLINK:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def load_image_size(image_path: Path):
    with Image.open(image_path) as im:
        w, h = im.size
    return w, h


def maybe_fix_scaled_rotation(c2w: np.ndarray) -> np.ndarray:
    """
    MatrixCity docs/snippet indicate:
    - Rotation block may be scaled by 0.01, needs to be multiplied by 100.
    - Translation vector needs to be multiplied by 0.01 to match depth/point cloud units (cm).
    """
    c2w = c2w.copy()
    
    # Fix the rotation matrix
    c2w[:3, :3] *= 100.0  # Rotation matrix needs to be multiplied by 100
    
    # Fix the translation vector (convert from meters to centimeters)
    c2w[:3, 3] *= 0.01  # Translation needs to be multiplied by 0.01
    
    return c2w


def matrixcity_c2w_to_colmap_w2c(c2w_nerf: np.ndarray) -> np.ndarray:
    """
    MatrixCity uses NeRF/OpenGL-style camera coordinates:
        x right, y up, z backward
    COLMAP uses OpenCV-style camera coordinates:
        x right, y down, z forward

    Convert c2w_nerf -> c2w_cv by flipping camera Y and Z axes, then invert.
    """
    c2w_cv = c2w_nerf.copy()
    c2w_cv[:3, 1:3] *= -1.0
    w2c = np.linalg.inv(c2w_cv)
    return w2c


def load_ply_points_and_colors(ply_path: Path):
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]

    pts = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)

    names = vertex.data.dtype.names
    if all(k in names for k in ("red", "green", "blue")):
        cols = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    else:
        cols = np.full((pts.shape[0], 3), 255, dtype=np.uint8)

    return pts, cols


def write_project_ini(path: Path, image_dir: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("[General]\n")
        f.write(f"image_path={image_dir}\n")


def collect_image_records(frames_sorted, image_dir: Path):
    records = []
    for image_id, frame in enumerate(frames_sorted, start=1):
        frame_index = int(frame["frame_index"])
        image_name = f"{frame_index:04d}.png"
        src_image = BLOCK_DIR / image_name
        dst_image = image_dir / image_name

        if not src_image.exists():
            print(f"[Warning] Missing image for frame_index={frame_index}: {src_image}")
            continue

        link_or_copy(src_image, dst_image)

        if "rot_mat" in frame:
            c2w = np.array(frame["rot_mat"], dtype=np.float64)
        elif "transform_matrix" in frame:
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        else:
            print(
                f"[Warning] Frame {frame_index} has no rot_mat/transform_matrix. Skipped."
            )
            continue

        c2w = maybe_fix_scaled_rotation(c2w)
        w2c = matrixcity_c2w_to_colmap_w2c(c2w)
        qvec = rotmat2qvec(w2c[:3, :3]).astype(np.float64)
        tvec = w2c[:3, 3].astype(np.float64)

        records.append(
            {
                "image_id": image_id,
                "camera_id": 1,
                "name": image_name,
                "qvec": qvec,
                "tvec": tvec,
            }
        )
    return records


def write_cameras_bin(
    path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float
):
    with open(path, "wb") as f:
        f.write(struct.pack("L", 1))
        f.write(struct.pack("IiLL", 1, 1, width, height))
        f.write(np.array([fx, fy, cx, cy], dtype=np.float64).tobytes())
    print(f"[Info] Wrote cameras.bin: {path}")


def write_images_bin(path: Path, image_records):
    image_struct = struct.Struct("<I 4d 3d I")
    with open(path, "wb") as f:
        f.write(struct.pack("L", len(image_records)))
        for record in image_records:
            f.write(
                image_struct.pack(
                    record["image_id"],
                    *record["qvec"].tolist(),
                    *record["tvec"].tolist(),
                    record["camera_id"],
                )
            )
            f.write(record["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("Q", 0))
    print(f"[Info] Wrote {len(image_records)} posed images to {path}")


def write_points3d_bin(path: Path, ply_path: Path, max_points=None, seed=42):
    pts, cols = load_ply_points_and_colors(ply_path)
    print(f"[Info] Loaded {len(pts)} points from {ply_path}")

    if max_points is not None and len(pts) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]
        print(f"[Info] Randomly downsampled point cloud to {len(pts)} points")

    point_struct = struct.Struct("<Q 3d 3B d Q")
    with open(path, "wb") as f:
        f.write(struct.pack("L", len(pts)))
        for i, (p, c) in enumerate(zip(pts, cols), start=1):
            f.write(
                point_struct.pack(
                    i,
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    int(c[0]),
                    int(c[1]),
                    int(c[2]),
                    1.0,
                    0,
                )
            )
    print(f"[Info] Wrote points3D.bin: {path}")


def main():
    if not BLOCK_DIR.exists():
        raise FileNotFoundError(f"BLOCK_DIR not found: {BLOCK_DIR}")
    if not TRANSFORMS_PATH.exists():
        raise FileNotFoundError(f"transforms.json not found: {TRANSFORMS_PATH}")
    if not PLY_PATH.exists():
        raise FileNotFoundError(f"PLY not found: {PLY_PATH}")

    images_out = OUT_DIR / "images"
    sparse_out = OUT_DIR / "sparse" / "0"
    ensure_dir(images_out)
    ensure_dir(sparse_out)

    with open(TRANSFORMS_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Infer image size from first posed image that exists.
    frames = meta["frames"]
    frames_sorted = sorted(frames, key=lambda x: int(x["frame_index"]))

    first_existing_image = None
    for frame in frames_sorted:
        idx = int(frame["frame_index"])
        cand = BLOCK_DIR / f"{idx:04d}.png"
        if cand.exists():
            first_existing_image = cand
            break
    if first_existing_image is None:
        raise RuntimeError("No posed image found in block_1.")

    width, height = load_image_size(first_existing_image)

    # Intrinsics
    if "fl_x" in meta:
        fx = float(meta["fl_x"])
    else:
        camera_angle_x = float(meta["camera_angle_x"])
        fx = 0.5 * width / math.tan(0.5 * camera_angle_x)

    if "fl_y" in meta:
        fy = float(meta["fl_y"])
    else:
        fy = fx

    cx = float(meta.get("cx", width / 2.0))
    cy = float(meta.get("cy", height / 2.0))

    print(f"[Info] Image size: {width} x {height}")
    print(f"[Info] Intrinsics: fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")

    image_records = collect_image_records(frames_sorted, images_out)
    write_project_ini(sparse_out / "project.ini", images_out)
    write_cameras_bin(sparse_out / "cameras.bin", width, height, fx, fy, cx, cy)
    write_images_bin(sparse_out / "images.bin", image_records)
    write_points3d_bin(
        sparse_out / "points3D.bin",
        PLY_PATH,
        max_points=MAX_POINTS,
        seed=RANDOM_SEED,
    )

    print(f"\n[Done] COLMAP-style dataset created at:\n{OUT_DIR}")
    print("\nYou can now point gsplat to this folder, e.g. use factor=1 first.")


if __name__ == "__main__":
    main()
