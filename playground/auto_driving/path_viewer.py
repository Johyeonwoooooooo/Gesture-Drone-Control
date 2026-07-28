"""
Fast 3D RRT* Viewer using Open3D
GPU accelerated point cloud visualization

Requirements:
    pip install open3d numpy
"""

import os
import re
import numpy as np
import open3d as o3d
from tkinter import Tk, filedialog


class FastPathViewer3D:

    def __init__(self):
        self.path = None
        self.coord = None
        self.segment = None
        self.room_name = None

    def load_path_file(self, path_file):

        print(f"Loading path: {path_file}")

        # room id 추출
        match = re.search(r'path_(\d{3})\.txt', os.path.basename(path_file))

        if match:
            room_id = match.group(1)
            self.room_name = f'00800_TEEsavR23oF_000_{room_id}'

        path_points = []

        with open(path_file, 'r') as f:
            for line in f:

                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                xyz = [float(v.strip()) for v in line.split(',')]

                if len(xyz) == 3:
                    path_points.append(xyz)

        self.path = np.array(path_points, dtype=np.float32)

        print(f"Loaded {len(self.path)} waypoints")

    def load_room_data(self):

        if self.room_name is None:
            return False

        room_folder = f'./compressed_npy/{self.room_name}'

        coord_file = f'{room_folder}/coord.npy'
        segment_file = f'{room_folder}/segment.npy'

        if not os.path.exists(coord_file):
            print("Room data not found")
            return False

        print("Loading point cloud...")

        self.coord = np.load(coord_file).astype(np.float32)
        self.segment = np.load(segment_file)

        print(f"Loaded {len(self.coord):,} points")

        return True

    def create_pointcloud(self, downsample_rate=3):

        indices = np.arange(0, len(self.coord), downsample_rate)

        pts = self.coord[indices]
        seg = self.segment[indices]

        colors = np.zeros((len(pts), 3), dtype=np.float32)

        # floor
        floor_mask = seg == 0
        colors[floor_mask] = [0.7, 0.7, 0.7]

        # wall
        wall_mask = seg == 1
        colors[wall_mask] = [0.5, 0.3, 0.2]

        # obstacle
        obstacle_mask = seg == 2
        colors[obstacle_mask] = [1.0, 0.0, 0.0]

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def create_path_lines(self):

        points = self.path

        lines = []

        for i in range(len(points) - 1):
            lines.append([i, i + 1])

        line_set = o3d.geometry.LineSet()

        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # 파란 경로
        colors = [[0, 0, 1] for _ in lines]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def create_waypoints(self):

        spheres = []

        for i, p in enumerate(self.path):

            # 시작점
            if i == 0:
                color = [0, 1, 0]
                radius = 0.12

            # 목표점
            elif i == len(self.path) - 1:
                color = [1, 0, 0]
                radius = 0.12

            else:
                color = [0, 0, 1]
                radius = 0.06

            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)

            mesh.translate(p)
            mesh.paint_uniform_color(color)

            spheres.append(mesh)

        return spheres

    def show(self):

        geometries = []

        # point cloud
        if self.coord is not None:
            pcd = self.create_pointcloud(downsample_rate=2)
            geometries.append(pcd)

        # path line
        path_lines = self.create_path_lines()
        geometries.append(path_lines)

        # waypoint spheres
        geometries.extend(self.create_waypoints())

        print()
        print("Controls:")
        print("  Left Drag  : Rotate")
        print("  Right Drag : Move")
        print("  Wheel      : Zoom")
        print()

        o3d.visualization.draw_geometries(
            geometries,
            window_name='Fast 3D RRT* Viewer',
            width=1400,
            height=900
        )


def select_file():

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title='Select Path TXT',
        initialdir='./output',
        filetypes=[('Text Files', '*.txt')]
    )

    root.destroy()

    return file_path


def main():

    file_path = select_file()

    if not file_path:
        print("No file selected")
        return

    viewer = FastPathViewer3D()

    viewer.load_path_file(file_path)

    viewer.load_room_data()

    viewer.show()


if __name__ == '__main__':
    main()