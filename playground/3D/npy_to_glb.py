"""Convert a Pointcept-style npy point cloud (coord/color/normal) to a GLB mesh.

Usage:
    python npy_to_glb.py --input <dir with coord.npy/color.npy/normal.npy> --output <out.glb>

Pipeline: load npy -> Poisson reconstruction -> density/bbox cleanup ->
vertex color transfer -> GLB export (vertex colors as COLOR_0).
"""

import argparse
import os

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default=r"C:\Users\yuni2\Downloads\00001_UVdNNRcVyV1_000_002-20260714T021023Z-1-001\00001_UVdNNRcVyV1_000_002",
    help="Directory containing coord.npy, color.npy, normal.npy",
)
parser.add_argument(
    "--output",
    default=os.path.join(
        os.path.dirname(__file__), "..", "tello_simulator", "Assets", "UVdNNRcVyV1_000_002.glb"
    ),
    help="Output .glb path",
)
parser.add_argument("--depth", type=int, default=9, help="Poisson octree depth")
args = parser.parse_args()

coord = np.load(os.path.join(args.input, "coord.npy")).astype(np.float64)
color = np.load(os.path.join(args.input, "color.npy"))  # uint8 0-255
normal = np.load(os.path.join(args.input, "normal.npy")).astype(np.float64)
print(f"Loaded {len(coord)} points")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coord)
pcd.normals = o3d.utility.Vector3dVector(normal)

print(f"Running Poisson reconstruction (depth={args.depth})...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=args.depth
)
print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

# Poisson closes surfaces far from the data; drop low-density vertices and
# anything outside the point cloud bounds.
densities = np.asarray(densities)
mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
bbox = pcd.get_axis_aligned_bounding_box()
bbox = o3d.geometry.AxisAlignedBoundingBox(
    bbox.min_bound - 0.05, bbox.max_bound + 0.05
)
mesh = mesh.crop(bbox)
mesh.remove_unreferenced_vertices()
mesh.remove_degenerate_triangles()
print(f"After cleanup: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

print("Transferring colors from point cloud...")
tree = cKDTree(coord)
_, idx = tree.query(np.asarray(mesh.vertices), k=1, workers=-1)
vertex_colors = color[idx]

out = trimesh.Trimesh(
    vertices=np.asarray(mesh.vertices),
    faces=np.asarray(mesh.triangles),
    vertex_colors=vertex_colors,
    process=False,
)
out_path = os.path.abspath(args.output)
out.export(out_path)
print(f"Saved {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
