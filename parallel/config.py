"""
config.py
=========
Central place for every tuneable constant.
Edit here; nothing else needs to change.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Voxelisation
# ---------------------------------------------------------------------------
RESOLUTION: float = 0.12      # metres per voxel edge
MARGIN: int = 1               # obstacle inflation radius (voxels)
POINT_SAMPLE_STEP: int = 1    # use every N-th point when building the grid
                               # (increase if RAM is tight)

# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------
ASTAR_MAX_ITER: int = 500_000  # safety cap on expanded nodes

# ---------------------------------------------------------------------------
# Room graph
# ---------------------------------------------------------------------------
# Maps each room sub-ID to the room IDs it shares a doorway / opening with.
# Derived from the TEEsavR23oF.glb bounding-box analysis; update if you add
# finer door-centroid labels later.
ROOM_GRAPH: dict[str, list[str]] = {
    "sub002": ["sub004", "sub005", "sub007", "sub009"],
    "sub003": ["sub004", "sub005"],
    "sub004": ["sub002", "sub003", "sub005"],
    "sub005": ["sub002", "sub003", "sub004", "sub006", "sub009", "sub011"],
    "sub006": ["sub005", "sub007", "sub008", "sub011"],
    "sub007": ["sub002", "sub006", "sub008", "sub009"],
    "sub008": ["sub006", "sub007"],
    "sub009": ["sub002", "sub005", "sub007"],
    "sub010": ["sub002", "sub005"],   # 2nd floor; connects via stairs
    "sub011": ["sub005", "sub006", "sub010"],
}

# World-space waypoints used to stitch adjacent room segments together.
# Key: frozenset({roomA, roomB}) → (x, y, z) midpoint of the shared doorway.
# Approximated from GLB bounding-box centroids; replace with door-detection
# results when available.
DOOR_WAYPOINTS: dict[frozenset, tuple[float, float, float]] = {
    frozenset({"sub002", "sub004"}): ( 0.10, 2.93, 1.20),
    frozenset({"sub002", "sub005"}): (-2.37, 4.75, 1.20),
    frozenset({"sub002", "sub007"}): (-5.21, 3.60, 1.20),
    frozenset({"sub002", "sub009"}): (-3.73, 3.55, 1.20),
    frozenset({"sub003", "sub004"}): ( 1.49, 4.82, 1.20),
    frozenset({"sub003", "sub005"}): (-0.60, 6.76, 1.20),
    frozenset({"sub004", "sub005"}): (-0.85, 4.82, 1.20),
    frozenset({"sub005", "sub006"}): (-3.74, 6.55, 1.20),
    frozenset({"sub005", "sub009"}): (-3.73, 5.55, 1.20),
    frozenset({"sub005", "sub011"}): (-2.37, 8.38, 1.20),
    frozenset({"sub006", "sub007"}): (-5.21, 5.33, 1.20),
    frozenset({"sub006", "sub008"}): (-6.10, 7.14, 1.20),
    frozenset({"sub006", "sub011"}): (-5.02, 8.38, 1.20),
    frozenset({"sub007", "sub008"}): (-6.10, 4.38, 1.20),
    frozenset({"sub007", "sub009"}): (-4.47, 4.62, 1.20),
    frozenset({"sub002", "sub010"}): (-4.55, 2.30, 4.31),
    frozenset({"sub005", "sub010"}): (-2.37, 6.13, 4.31),
    frozenset({"sub010", "sub011"}): (-5.27, 8.38, 4.31),
}

# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------
# None → use os.cpu_count() at runtime
WORKER_COUNT: int | None = None

# ---------------------------------------------------------------------------
# Benchmark / plotting
# ---------------------------------------------------------------------------
BENCHMARK_RUNS: int = 5
PLOT_DPI: int = 180
PALETTE: dict[str, str] = {
    "sequential": "#0094d4",
    "parallel":   "#7c3aed",
    "speedup":    "#e05c00",
}

