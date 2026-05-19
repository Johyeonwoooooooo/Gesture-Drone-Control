"""
voxel_io.py
===========
Load per-room point-cloud NPY files and convert them to voxel grids.

Expected folder layout
----------------------
<data_dir>/
    00800_TEEsavR23oF_000_002/
        coord.npy       ← (N, 3) float32 xyz coordinates  [required]
        color.npy       ← (N, 3) uint8  RGB               [optional]
        instance.npy    ← (N,)   int    instance ids       [optional]
        normal.npy      ← (N, 3) float  surface normals    [optional]
        segment.npy     ← (N,)   int    semantic labels    [optional]
    00800_TEEsavR23oF_000_003/
        ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GridMeta:
    """A 3-D occupancy voxel grid with its world-space origin and resolution."""

    grid: np.ndarray          # shape (Nx, Ny, Nz), dtype uint8; 0=free, 1=occupied
    x_min: float
    y_min: float
    z_min: float
    resolution: float
    room_id: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape  # type: ignore[return-value]

    @property
    def free_ratio(self) -> float:
        return float(np.mean(self.grid == 0))

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.x_min, self.y_min, self.z_min], dtype=float)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def to_voxel(
        self,
        point: np.ndarray | tuple[float, float, float],
    ) -> tuple[int, int, int]:
        x, y, z = point
        return (
            round((x - self.x_min) / self.resolution),
            round((y - self.y_min) / self.resolution),
            round((z - self.z_min) / self.resolution),
        )

    def to_world(self, voxel: tuple[int, int, int]) -> np.ndarray:
        ix, iy, iz = voxel
        return np.array(
            [
                self.x_min + ix * self.resolution,
                self.y_min + iy * self.resolution,
                self.z_min + iz * self.resolution,
            ],
            dtype=float,
        )

    def at(self, ix: int, iy: int, iz: int) -> int:
        """Return occupancy (1) or free (0); out-of-bounds → occupied."""
        nx, ny, nz = self.shape
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            return 1
        return int(self.grid[ix, iy, iz])

    def __repr__(self) -> str:
        nx, ny, nz = self.shape
        return (
            f"GridMeta(room={self.room_id!r}, shape=({nx},{ny},{nz}), "
            f"res={self.resolution:.3f}, free={self.free_ratio*100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Map from the sub-ID suffix (e.g. "002") to the folder-name infix
_SUFFIX_RE_PART = "00800_TEEsavR23oF"


def _find_room_dir(data_dir: Path, room_id: str) -> Optional[Path]:
    """
    Locate the folder that corresponds to *room_id* (e.g. ``"sub002"``).

    Accepts directories whose name ends with the zero-padded room number,
    e.g.  ``00800_TEEsavR23oF_000_002``  for ``sub002``.
    """
    number = room_id.replace("sub", "")  # "002"
    for candidate in data_dir.iterdir():
        if not candidate.is_dir():
            continue
        if candidate.name.endswith(f"_{number}"):
            return candidate
    return None


def _inflate(grid: np.ndarray, margin: int) -> np.ndarray:
    """Morphological dilation with a cubic structuring element of radius *margin*."""
    if margin <= 0:
        return grid
    occupied = np.argwhere(grid == 1)
    inflated = grid.copy()
    nx, ny, nz = grid.shape
    for ix, iy, iz in occupied:
        x0, x1 = max(0, ix - margin), min(nx, ix + margin + 1)
        y0, y1 = max(0, iy - margin), min(ny, iy + margin + 1)
        z0, z1 = max(0, iz - margin), min(nz, iz + margin + 1)
        inflated[x0:x1, y0:y1, z0:z1] = 1
    return inflated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_room(
    data_dir: Path,
    room_id: str,
    resolution: float = 0.12,
    margin: int = 1,
    sample_step: int = 1,
) -> GridMeta:
    """
    Load the ``coord.npy`` of *room_id* and return a :class:`GridMeta`.

    Parameters
    ----------
    data_dir:
        Root folder that contains one sub-directory per room.
    room_id:
        Room identifier, e.g. ``"sub002"``.
    resolution:
        Voxel edge length in metres.
    margin:
        Obstacle inflation radius in voxels (robot body clearance).
    sample_step:
        Use every *sample_step*-th point to save RAM.  ``1`` = all points.
    """
    room_dir = _find_room_dir(data_dir, room_id)
    if room_dir is None:
        raise FileNotFoundError(
            f"No directory found for room {room_id!r} under {data_dir}"
        )

    coord_path = room_dir / "coord.npy"
    if not coord_path.exists():
        raise FileNotFoundError(f"coord.npy not found in {room_dir}")

    points: np.ndarray = np.load(coord_path).astype(float)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    points = points[:, :3]                    # keep only XYZ
    points = points[::max(1, sample_step)]    # down-sample

    return voxelize(points, resolution=resolution, margin=margin, room_id=room_id)


def voxelize(
    points: np.ndarray,
    resolution: float = 0.12,
    margin: int = 1,
    room_id: str = "",
) -> GridMeta:
    """
    Convert a raw (N, 3) point cloud into a :class:`GridMeta`.

    Points are rounded to the nearest voxel centre; occupied voxels are
    then inflated by *margin* voxels in each direction.
    """
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    shape = np.ceil((xyz_max - xyz_min) / resolution).astype(int) + 3

    grid = np.zeros(tuple(shape), dtype=np.uint8)
    ijk = np.rint((points - xyz_min) / resolution).astype(int)
    valid = np.all((ijk >= 0) & (ijk < shape), axis=1)
    idx = ijk[valid]
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    grid = _inflate(grid, margin)

    return GridMeta(
        grid=grid,
        x_min=float(xyz_min[0]),
        y_min=float(xyz_min[1]),
        z_min=float(xyz_min[2]),
        resolution=resolution,
        room_id=room_id,
    )


def load_all_rooms(
    data_dir: Path,
    room_ids: list[str],
    resolution: float = 0.12,
    margin: int = 1,
    sample_step: int = 1,
) -> dict[str, GridMeta]:
    """Load multiple rooms and return a ``{room_id: GridMeta}`` mapping."""
    grids: dict[str, GridMeta] = {}
    for rid in room_ids:
        try:
            grids[rid] = load_room(
                data_dir, rid,
                resolution=resolution,
                margin=margin,
                sample_step=sample_step,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(str(exc)) from exc
    return grids
