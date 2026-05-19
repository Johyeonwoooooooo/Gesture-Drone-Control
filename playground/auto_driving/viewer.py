"""
[2단계] .npy 파일을 읽어서 즉시 3D 뷰어 실행
파일 선택창 → 바로 뷰어 오픈
"""

import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import sys
import os

def pick_npy_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="voxel .npy 파일을 선택하세요",
        filetypes=[("Voxel 파일", "*.npy"), ("모든 파일", "*.*")]
    )
    root.destroy()
    return path

def visualize(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    colors = np.tile([0.78, 0.25, 0.18], (len(pts), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Voxel Viewer", width=1400, height=900)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.15, 0.15, 0.15])
    opt.point_size = 2.0

    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    ctr.set_front([0.5, -0.8, 0.6])
    ctr.set_up([0, 0, 1])

    vis.run()
    vis.destroy_window()

def main():
    # 인자로 npy 경로 받거나, 없으면 파일 선택창
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        npy_path = sys.argv[1]
    else:
        npy_path = pick_npy_file()

    if not npy_path:
        print("파일이 선택되지 않았습니다.")
        return

    print(f"로딩 중: {npy_path}")
    pts = np.load(npy_path)
    print(f"포인트 수: {len(pts):,}  → 뷰어 실행")
    visualize(pts)

if __name__ == "__main__":
    main()
