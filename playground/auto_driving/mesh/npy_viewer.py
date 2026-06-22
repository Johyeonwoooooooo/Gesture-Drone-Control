"""
Point Cloud 3D Viewer
----------------------
py 파일 실행하면 폴더 선택 창이 뜹니다.

마우스 조작:
    - 왼쪽 드래그: 회전
    - 오른쪽 드래그: 이동
    - 스크롤: 줌
    - R: 뷰 초기화
    - Q: 종료
"""

import numpy as np
import open3d as o3d
import os
import sys
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# ─────────────────────────────────────────
# 포인트클라우드 로드
# ─────────────────────────────────────────

def load_scene(folder_path):
    coord_path  = os.path.join(folder_path, "coord.npy")
    color_path  = os.path.join(folder_path, "color.npy")
    normal_path = os.path.join(folder_path, "normal.npy")

    if not os.path.exists(coord_path):
        return None

    coords = np.load(coord_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))

    if os.path.exists(color_path):
        colors = np.load(color_path).astype(np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    else:
        z = coords[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-8)
        colormap = np.stack([z_norm, 1 - z_norm, np.ones_like(z_norm) * 0.5], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colormap)

    if os.path.exists(normal_path):
        normals = np.load(normal_path).astype(np.float64)
        pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


def load_scenes(folders, progress_callback=None):
    combined = o3d.geometry.PointCloud()
    for i, folder in enumerate(folders):
        pcd = load_scene(folder)
        if pcd is not None:
            combined += pcd
        if progress_callback:
            progress_callback(i + 1, len(folders), os.path.basename(folder))
    return combined


# ─────────────────────────────────────────
# 뷰어
# ─────────────────────────────────────────

def visualize(pcd, title="Point Cloud Viewer"):
    if len(pcd.points) == 0:
        messagebox.showerror("오류", "포인트가 없습니다.")
        return

    n = len(pcd.points)
    point_size = max(0.5, min(3.0, 500000 / n))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"{title}  |  포인트 수: {n:,}", width=1280, height=800)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.05, 0.05, 0.1])
    opt.show_coordinate_frame = True

    def reset_view(vis):
        vis.reset_view_point(True)
        return False

    vis.register_key_callback(ord("R"), reset_view)
    vis.run()
    vis.destroy_window()


# ─────────────────────────────────────────
# GUI
# ─────────────────────────────────────────

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Point Cloud Viewer")
        self.root.geometry("560x500")
        self.root.resizable(False, False)
        self.root.configure(bg="#0e0e16")

        self.selected_folder = None
        self.scene_folders = []
        self.check_vars = []

        self._build_ui()
        self.root.mainloop()

    def _build_ui(self):
        # 타이틀
        tk.Label(
            self.root, text="Point Cloud 3D Viewer",
            font=("Courier New", 16, "bold"),
            fg="#7dd3fc", bg="#0e0e16"
        ).pack(pady=(20, 4))

        tk.Label(
            self.root, text="폴더를 선택하면 씬 목록이 표시됩니다",
            font=("Courier New", 9),
            fg="#64748b", bg="#0e0e16"
        ).pack(pady=(0, 16))

        # 폴더 선택 버튼
        tk.Button(
            self.root, text="📁  폴더 선택",
            font=("Courier New", 11, "bold"),
            bg="#1e3a5f", fg="#7dd3fc",
            activebackground="#2a4a7f", activeforeground="#bae6fd",
            relief="flat", padx=20, pady=8, cursor="hand2",
            command=self._select_folder
        ).pack()

        self.folder_label = tk.Label(
            self.root, text="선택된 폴더 없음",
            font=("Courier New", 8),
            fg="#475569", bg="#0e0e16", wraplength=500
        )
        self.folder_label.pack(pady=(6, 12))

        # 씬 목록
        list_frame = tk.Frame(self.root, bg="#0e0e16")
        list_frame.pack(fill="both", expand=True, padx=24)

        tk.Label(
            list_frame, text="씬 목록",
            font=("Courier New", 9, "bold"),
            fg="#94a3b8", bg="#0e0e16", anchor="w"
        ).pack(fill="x")

        canvas_frame = tk.Frame(list_frame, bg="#141420", bd=1, relief="solid")
        canvas_frame.pack(fill="both", expand=True, pady=(4, 0))

        self.canvas = tk.Canvas(canvas_frame, bg="#141420", highlightthickness=0, height=180)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.check_frame = tk.Frame(self.canvas, bg="#141420")
        self.canvas.create_window((0, 0), window=self.check_frame, anchor="nw")
        self.check_frame.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        ))

        # 전체 선택/해제
        btn_frame = tk.Frame(self.root, bg="#0e0e16")
        btn_frame.pack(pady=(6, 0))

        tk.Button(
            btn_frame, text="전체 선택",
            font=("Courier New", 8), bg="#1e293b", fg="#94a3b8",
            relief="flat", padx=8, pady=4, cursor="hand2",
            command=self._select_all
        ).pack(side="left", padx=4)

        tk.Button(
            btn_frame, text="전체 해제",
            font=("Courier New", 8), bg="#1e293b", fg="#94a3b8",
            relief="flat", padx=8, pady=4, cursor="hand2",
            command=self._deselect_all
        ).pack(side="left", padx=4)

        self.count_label = tk.Label(
            btn_frame, text="",
            font=("Courier New", 8), fg="#64748b", bg="#0e0e16"
        )
        self.count_label.pack(side="left", padx=8)

        # 다운샘플 옵션
        opt_frame = tk.Frame(self.root, bg="#0e0e16")
        opt_frame.pack(pady=(10, 0))

        tk.Label(
            opt_frame, text="다운샘플 voxel 크기 (0 = 안 함):",
            font=("Courier New", 8), fg="#64748b", bg="#0e0e16"
        ).pack(side="left")

        self.voxel_var = tk.StringVar(value="0")
        tk.Entry(
            opt_frame, textvariable=self.voxel_var,
            font=("Courier New", 9), width=6,
            bg="#1e293b", fg="#e2e8f0", insertbackground="white",
            relief="flat"
        ).pack(side="left", padx=6)

        # 열기 버튼
        tk.Button(
            self.root, text="🚀  선택한 씬 열기",
            font=("Courier New", 12, "bold"),
            bg="#0f4c81", fg="#e0f2fe",
            activebackground="#1a6aad", activeforeground="white",
            relief="flat", padx=24, pady=10, cursor="hand2",
            command=self._open_viewer
        ).pack(pady=(12, 8))

        # 진행 바 + 상태
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            self.root, variable=self.progress_var,
            maximum=100, length=510
        )
        self.progress.pack(pady=(0, 4))

        self.status_label = tk.Label(
            self.root, text="",
            font=("Courier New", 8), fg="#64748b", bg="#0e0e16"
        )
        self.status_label.pack()

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Point Cloud 폴더 선택")
        if not folder:
            return

        self.selected_folder = folder
        self.folder_label.config(text=folder, fg="#7dd3fc")

        # 단일 씬인지 루트 폴더인지 판단
        has_coord = os.path.exists(os.path.join(folder, "coord.npy"))
        if has_coord:
            self.scene_folders = [folder]
        else:
            self.scene_folders = sorted([
                d for d in glob.glob(os.path.join(folder, "*"))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "coord.npy"))
            ])

        self._build_checklist()

    def _build_checklist(self):
        for widget in self.check_frame.winfo_children():
            widget.destroy()
        self.check_vars = []

        if not self.scene_folders:
            tk.Label(
                self.check_frame,
                text="coord.npy가 있는 씬 폴더를 찾지 못했습니다",
                font=("Courier New", 9), fg="#ef4444", bg="#141420"
            ).pack(pady=10)
            return

        for folder in self.scene_folders:
            var = tk.BooleanVar(value=True)
            self.check_vars.append(var)
            tk.Checkbutton(
                self.check_frame,
                text=os.path.basename(folder),
                variable=var,
                font=("Courier New", 9),
                fg="#cbd5e1", bg="#141420",
                selectcolor="#0f172a",
                activebackground="#141420",
                activeforeground="#7dd3fc",
                command=self._update_count
            ).pack(anchor="w", padx=8, pady=1)

        self._update_count()

    def _select_all(self):
        for var in self.check_vars:
            var.set(True)
        self._update_count()

    def _deselect_all(self):
        for var in self.check_vars:
            var.set(False)
        self._update_count()

    def _update_count(self):
        n = sum(v.get() for v in self.check_vars)
        self.count_label.config(text=f"{n}/{len(self.check_vars)}개 선택")

    def _open_viewer(self):
        selected = [f for f, v in zip(self.scene_folders, self.check_vars) if v.get()]
        if not selected:
            messagebox.showwarning("알림", "씬을 하나 이상 선택하세요.")
            return

        voxel = 0.0
        try:
            voxel = float(self.voxel_var.get())
        except ValueError:
            pass

        self.status_label.config(text="로딩 중...")
        self.progress_var.set(0)
        self.root.update()

        def progress_cb(done, total, name):
            self.progress_var.set(done / total * 100)
            self.status_label.config(text=f"로딩 중... {name} ({done}/{total})")
            self.root.update()

        pcd = load_scenes(selected, progress_callback=progress_cb)

        if voxel > 0:
            before = len(pcd.points)
            pcd = pcd.voxel_down_sample(voxel_size=voxel)
            self.status_label.config(
                text=f"다운샘플: {before:,} → {len(pcd.points):,} 포인트"
            )
        else:
            self.status_label.config(text=f"로드 완료: {len(pcd.points):,} 포인트")

        self.progress_var.set(100)
        self.root.update()

        # GUI 숨기고 뷰어 열기, 뷰어 닫으면 GUI 복원
        self.root.withdraw()
        visualize(pcd, title=f"{len(selected)}개 씬")
        self.root.deiconify()
        self.progress_var.set(0)
        self.status_label.config(text="뷰어 종료. 다시 선택하거나 닫으세요.")


if __name__ == "__main__":
    App()