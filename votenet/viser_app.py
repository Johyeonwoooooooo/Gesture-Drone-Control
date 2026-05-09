import os
import sys
import time
import threading
import argparse
from pathlib import Path

import numpy as np
import torch
import importlib
import clip
import viser

BASE_DIR = '/shareHost/minyoy/votenet'
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'scannet'))

from plyfile import PlyData
from pc_util import random_sampling
from ap_helper import parse_predictions
from scannet_detection_dataset import DC

# ── 기본 설정 ──────────────────────────────────────
DEFAULT_PLY_PATH = os.path.join(BASE_DIR, 'demo_files/input_pc_hm3d.ply')
DEFAULT_SCENE_DIR = os.path.join(BASE_DIR, 'demo_files')
CKPT_PATH = os.path.join(BASE_DIR, 'demo_files/pretrained_votenet_on_scannet.tar')
NUM_POINT = 40000
CONF_THRESH = 0.5
VISER_PORT = 8080
SPLAT_SIZE = 0.012
# ──────────────────────────────────────────────────

SCANNET_CLASSES = [DC.class2type[i] for i in range(DC.num_class)]

CLASS_COLORS = {
    'bed': (255, 75, 75),
    'chair': (75, 158, 255),
    'door': (75, 255, 111),
    'window': (255, 216, 75),
    'table': (255, 75, 240),
    'curtain': (75, 255, 240),
    'picture': (255, 151, 75),
    'cabinet': (166, 75, 255),
    'bookshelf': (75, 255, 166),
    'counter': (255, 75, 151),
    'desk': (151, 255, 75),
    'refrigerator': (75, 75, 255),
    'toilet': (255, 200, 200),
    'sink': (200, 200, 255),
    'bathtub': (200, 255, 200),
    'garbagebin': (220, 220, 120),
    'sofa': (255, 160, 200),
    'showercurtain': (160, 220, 255),
    'otherfurniture': (180, 180, 180),
}
DEFAULT_COLOR = (200, 200, 200)


def parse_args():
    parser = argparse.ArgumentParser(description='VoteNet + CLIP + Viser 3D scene search app')
    parser.add_argument('--ply_path', type=str, default=DEFAULT_PLY_PATH)
    parser.add_argument('--scene_dir', type=str, default=DEFAULT_SCENE_DIR,
                        help='Directory containing .ply files or subdirectories with input_pc.ply')
    parser.add_argument('--port', type=int, default=VISER_PORT)
    parser.add_argument('--num_point', type=int, default=NUM_POINT)
    parser.add_argument('--conf_thresh', type=float, default=CONF_THRESH)
    parser.add_argument('--splat_size', type=float, default=SPLAT_SIZE)
    return parser.parse_args()


def discover_scenes(scene_dir, default_ply):
    """Return {scene_name: ply_path}. Supports direct .ply files and */input_pc.ply."""
    scene_map = {}
    scene_dir = Path(scene_dir)

    if scene_dir.exists():
        for ply in sorted(scene_dir.glob('*.ply')):
            scene_map[ply.stem] = str(ply)
        for child in sorted(scene_dir.iterdir()):
            if child.is_dir():
                candidate = child / 'input_pc.ply'
                if candidate.exists():
                    scene_map[child.name] = str(candidate)

    default_ply = Path(default_ply)
    if default_ply.exists():
        scene_map.setdefault(default_ply.stem, str(default_ply))

    if not scene_map:
        # 경로가 아직 없어도 앱 실행 중 직접 확인할 수 있게 기본값은 넣어둔다.
        scene_map[default_ply.stem] = str(default_ply)

    return scene_map


def read_ply_xyzrgb(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    xyz = np.stack([np.array(pc['x']), np.array(pc['y']), np.array(pc['z'])], axis=1)
    try:
        rgb = np.stack([np.array(pc['red']), np.array(pc['green']), np.array(pc['blue'])], axis=1)
    except Exception:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    return xyz.astype(np.float32), rgb.astype(np.uint8)


def preprocess_point_cloud(xyz, num_point=40000):
    floor_height = np.percentile(xyz[:, 2], 0.99)
    height = xyz[:, 2] - floor_height
    pc = np.concatenate([xyz[:, :3], np.expand_dims(height, 1)], axis=1)
    pc = random_sampling(pc, num_point)
    return np.expand_dims(pc.astype(np.float32), 0)


def load_votenet(ckpt_path, device):
    MODEL = importlib.import_module('votenet')
    net = MODEL.VoteNet(
        num_proposal=256,
        input_feature_dim=1,
        vote_factor=1,
        sampling='seed_fps',
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
    ).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net


def text_to_class(query, clip_model, device):
    q_tok = clip.tokenize([query]).to(device)
    c_tok = clip.tokenize(SCANNET_CLASSES).to(device)
    with torch.no_grad():
        q_feat = clip_model.encode_text(q_tok)
        c_feat = clip_model.encode_text(c_tok)
    q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
    c_feat = c_feat / c_feat.norm(dim=-1, keepdim=True)
    sims = (q_feat @ c_feat.T).squeeze(0)
    best = sims.argmax().item()
    return best, SCANNET_CLASSES[best], sims[best].item()


def bbox_to_corners(center, size):
    cx, cy, cz = center
    dx, dy, dz = np.array(size) / 2
    return np.array([
        [cx - dx, cy - dy, cz - dz], [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz], [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz], [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz], [cx - dx, cy + dy, cz + dz],
    ], dtype=np.float32)


def bbox_to_lines(center, size):
    corners = bbox_to_corners(center, size)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return np.array([[corners[i], corners[j]] for i, j in edges], dtype=np.float32)


def get_center_size(bbox):
    if bbox.ndim > 1:
        center = bbox.mean(axis=0)
        size = bbox.max(axis=0) - bbox.min(axis=0)
    else:
        center = bbox[:3]
        size = bbox[3:6]
    return center.astype(np.float32), size.astype(np.float32)


def align_pc_to_bbox(xyz):
    """Match point cloud view coordinate system with VoteNet bbox visualization."""
    xyz_out = xyz.copy()
    xyz_out[:, 1] = -xyz[:, 2]
    xyz_out[:, 2] = xyz[:, 1]
    scene_center = xyz_out.mean(axis=0)
    xyz_out -= scene_center
    return xyz_out.astype(np.float32), scene_center.astype(np.float32)


def make_covariances(n, splat_size):
    s2 = splat_size ** 2
    cov = np.zeros((n, 3, 3), dtype=np.float32)
    cov[:, 0, 0] = s2
    cov[:, 1, 1] = s2
    cov[:, 2, 2] = s2
    return cov


def get_room_camera_pose(xyz_vis):
    xyz_min = xyz_vis.min(axis=0)
    xyz_max = xyz_vis.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    size = xyz_max - xyz_min

    eye_height = xyz_min[2] + size[2] * 0.45
    camera_pos = np.array([center[0], center[1], eye_height], dtype=np.float32)

    if size[0] > size[1]:
        look_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        look_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    camera_look_at = camera_pos + look_dir
    return camera_pos, camera_look_at


def roll_up_direction(camera_pos, camera_look_at, base_up=(0.0, 0.0, 1.0), direction='right'):
    forward = np.array(camera_look_at, dtype=np.float32) - np.array(camera_pos, dtype=np.float32)
    forward = forward / max(np.linalg.norm(forward), 1e-8)

    up = np.array(base_up, dtype=np.float32)
    up = up / max(np.linalg.norm(up), 1e-8)

    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-8)

    new_up = right if direction == 'right' else -right
    return tuple(new_up.astype(np.float32))


class ViserSceneApp:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')

        print('VoteNet 로드 중...')
        self.net = load_votenet(CKPT_PATH, self.device)
        print('CLIP 로드 중...')
        self.clip_model, _ = clip.load('ViT-B/32', device=self.device)
        print('모델 로드 완료!')

        self.server = viser.ViserServer(port=args.port)
        print(f'\n🌐 viser 웹앱: http://localhost:{args.port}')

        try:
            self.server.scene.world_axes.visible = False
        except Exception:
            pass

        self.scene_map = discover_scenes(args.scene_dir, args.ply_path)

        self.eval_config = {
            'remove_empty_box': True,
            'use_3d_nms': True,
            'nms_iou': 0.25,
            'use_old_type_nms': False,
            'cls_nms': False,
            'per_class_proposal': False,
            'conf_thresh': args.conf_thresh,
            'dataset_config': DC,
        }

        self.xyz = None
        self.rgb = None
        self.xyz_vis = None
        self.scene_center = None
        self.end_points = None
        self.detections = []
        self.camera_pos = None
        self.camera_look_at = None
        self.camera_up = None

        self.splat_handle = None
        self.bbox_handles = {}
        self.loading = False

        self.query_input = None
        self.search_btn = None
        self.result_md = None
        self.det_md = None
        self.scene_selector = None
        self.scene_path_input = None
        self.conf_slider = None
        self.splat_slider = None

        self._build_gui()
        self._register_client_handler()

    def _build_gui(self):
        with self.server.gui.add_folder('🗺️ 맵 선택'):
            scene_names = list(self.scene_map.keys())
            initial_scene = scene_names[0]
            self.scene_selector = self.server.gui.add_dropdown(
                'Scene',
                options=scene_names,
                initial_value=initial_scene,
            )
            self.scene_path_input = self.server.gui.add_text(
                'PLY 경로',
                initial_value=self.scene_map[initial_scene],
            )
            load_scene_btn = self.server.gui.add_button('맵 로드 / 교체')
            refresh_scene_btn = self.server.gui.add_button('Scene 목록 새로고침')

        with self.server.gui.add_folder('🔍 텍스트 쿼리'):
            self.query_input = self.server.gui.add_text('물체 입력', initial_value='find the bed')
            self.search_btn = self.server.gui.add_button('🔎 검색')
            self.result_md = self.server.gui.add_markdown('맵을 로드하면 결과가 표시됩니다.')

        with self.server.gui.add_folder('👁️ 보기'):
            show_all_btn = self.server.gui.add_button('전체 물체 보기')
            reset_view_btn = self.server.gui.add_button('방 중앙 시점으로 이동')
            self.conf_slider = self.server.gui.add_slider(
                'Confidence', min=0.1, max=1.0, step=0.05, initial_value=self.args.conf_thresh
            )
            self.splat_slider = self.server.gui.add_slider(
                'Splat 크기', min=0.002, max=0.05, step=0.001, initial_value=self.args.splat_size
            )

        with self.server.gui.add_folder('📋 감지 결과'):
            self.det_md = self.server.gui.add_markdown('아직 감지 결과가 없습니다.')

        @self.scene_selector.on_update
        def _(_event):
            selected = self.scene_selector.value
            self.scene_path_input.value = self.scene_map.get(selected, self.scene_path_input.value)

        @load_scene_btn.on_click
        def _(_event):
            self.load_scene(self.scene_path_input.value)

        @refresh_scene_btn.on_click
        def _(_event):
            self.scene_map = discover_scenes(self.args.scene_dir, self.args.ply_path)
            names = list(self.scene_map.keys())
            if names:
                self.scene_selector.options = names
                self.scene_selector.value = names[0]
                self.scene_path_input.value = self.scene_map[names[0]]
            self.result_md.content = f'Scene 목록을 갱신했습니다. {len(names)}개 발견.'

        @self.search_btn.on_click
        def _(_event):
            self.search_query()

        @show_all_btn.on_click
        def _(_event):
            self.show_all()
            self.result_md.content = '전체 물체를 표시합니다.'

        @reset_view_btn.on_click
        def _(_event):
            self.apply_room_camera_to_all_clients()
            self.result_md.content = '방 중앙 시점으로 이동했습니다.'

        @self.conf_slider.on_update
        def _(event):
            if self.end_points is None:
                return
            self.eval_config['conf_thresh'] = event.target.value
            self.detections = list(parse_predictions(self.end_points, self.eval_config)[0])
            self.show_all()
            self.update_detection_markdown()
            self.result_md.content = f'Confidence {event.target.value:.2f} → {len(self.detections)}개 감지'

        @self.splat_slider.on_update
        def _(event):
            if self.xyz_vis is not None and self.rgb is not None:
                self.render_splats(event.target.value)

    def _register_client_handler(self):
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            if self.camera_pos is None:
                return
            self.apply_room_camera(client)
            threading.Thread(
                target=self.apply_room_camera_repeated,
                args=(client,),
                daemon=True,
            ).start()

    def remove_scene_handles(self):
        if self.splat_handle is not None:
            try:
                self.splat_handle.remove()
            except Exception:
                pass
            self.splat_handle = None
        self.clear_bboxes()

    def load_scene(self, ply_path):
        if self.loading:
            self.result_md.content = '이미 맵을 로드 중입니다.'
            return

        self.loading = True
        try:
            ply_path = str(ply_path).strip()
            if not ply_path:
                self.result_md.content = 'PLY 경로가 비어 있습니다.'
                return
            if not os.path.exists(ply_path):
                self.result_md.content = f'❌ PLY 파일을 찾을 수 없습니다:\n`{ply_path}`'
                return

            self.result_md.content = f'🔄 맵 로드 중...\n`{ply_path}`'
            print(f'\n🔄 맵 로드 중: {ply_path}')

            self.remove_scene_handles()

            self.xyz, self.rgb = read_ply_xyzrgb(ply_path)
            pc = preprocess_point_cloud(self.xyz, self.args.num_point)
            pc_tensor = torch.from_numpy(pc).to(self.device)

            print('VoteNet 추론 중...')
            tic = time.time()
            with torch.no_grad():
                self.end_points = self.net({'point_clouds': pc_tensor})
            self.end_points['point_clouds'] = pc_tensor
            print(f'추론 시간: {time.time() - tic:.3f}초')

            self.detections = list(parse_predictions(self.end_points, self.eval_config)[0])
            print(f'총 감지된 물체: {len(self.detections)}개')

            self.xyz_vis, self.scene_center = align_pc_to_bbox(self.xyz)
            self.camera_pos, self.camera_look_at = get_room_camera_pose(self.xyz_vis)
            self.camera_up = roll_up_direction(
                self.camera_pos,
                self.camera_look_at,
                direction='right',
            )

            self.server.initial_camera.position = tuple(self.camera_pos)
            self.server.initial_camera.look_at = tuple(self.camera_look_at)
            self.server.initial_camera.up_direction = self.camera_up

            self.render_splats(self.splat_slider.value if self.splat_slider is not None else self.args.splat_size)
            self.show_all()
            self.update_detection_markdown()
            self.apply_room_camera_to_all_clients()
            self.force_camera_after_load()

            self.result_md.content = (
                f'✅ 맵 로드 완료\n\n'
                f'- 파일: `{ply_path}`\n'
                f'- 포인트 수: {len(self.xyz):,}\n'
                f'- 감지 물체: {len(self.detections)}개'
            )
        except Exception as exc:
            self.result_md.content = f'❌ 맵 로드 실패:\n`{exc}`'
            print(f'❌ 맵 로드 실패: {exc}')
        finally:
            self.loading = False

    def render_splats(self, splat_size):
        if self.xyz_vis is None or self.rgb is None:
            return

        if self.splat_handle is not None:
            try:
                self.splat_handle.remove()
            except Exception:
                pass
            self.splat_handle = None

        cov = make_covariances(len(self.xyz_vis), splat_size)
        rgbs_f = (self.rgb / 255.0).astype(np.float32)
        opacity = np.full((len(self.xyz_vis), 1), 0.95, dtype=np.float32)

        # 같은 name을 재사용하면 stale handle이 남는 경우가 있어 timestamp를 붙인다.
        self.splat_handle = self.server.scene.add_gaussian_splats(
            name=f'scene/splats_{time.time_ns()}',
            centers=self.xyz_vis,
            covariances=cov,
            rgbs=rgbs_f,
            opacities=opacity,
        )

    def clear_bboxes(self):
        for _name, handle in list(self.bbox_handles.items()):
            try:
                handle.remove()
            except Exception:
                pass
        self.bbox_handles.clear()

    def draw_bbox(self, idx, center, size, color_f, width, show_label=False, cls_name='', score=0.0):
        if self.scene_center is None:
            return

        c = center - self.scene_center
        lines = bbox_to_lines(c, size)
        colors_arr = np.tile(np.array(color_f, dtype=np.float32), (12, 2, 1))

        h = self.server.scene.add_line_segments(
            name=f'bbox/{idx}_{time.time_ns()}',
            points=lines,
            colors=colors_arr,
            line_width=width,
        )
        self.bbox_handles[f'bbox/{idx}'] = h

        if show_label:
            lh = self.server.scene.add_label(
                name=f'bbox/{idx}/label_{time.time_ns()}',
                text=f'{cls_name} ({score:.2f})',
                position=c.astype(np.float32),
            )
            self.bbox_handles[f'bbox/{idx}/label'] = lh

    def show_all(self):
        self.clear_bboxes()
        if not self.detections:
            return

        for i, (cls_id, bbox, score) in enumerate(self.detections):
            cls_name = DC.class2type[cls_id]
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            color_f = tuple(c / 255.0 for c in color)
            center, size = get_center_size(bbox)
            self.draw_bbox(
                i,
                center,
                size,
                color_f,
                2.0,
                show_label=True,
                cls_name=cls_name,
                score=score,
            )

    def highlight(self, cls_idx):
        self.clear_bboxes()
        if not self.detections:
            return

        for i, (cls_id, bbox, score) in enumerate(self.detections):
            cls_name = DC.class2type[cls_id]
            is_target = cls_id == cls_idx
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            color_f = tuple(c / 255.0 for c in color) if is_target else (0.25, 0.25, 0.25)
            width = 4.0 if is_target else 1.0
            center, size = get_center_size(bbox)
            self.draw_bbox(
                i,
                center,
                size,
                color_f,
                width,
                show_label=is_target,
                cls_name=cls_name,
                score=score,
            )

    def search_query(self):
        if not self.detections:
            self.result_md.content = '먼저 맵을 로드해야 합니다.'
            return

        query = self.query_input.value.strip()
        if not query:
            return

        self.result_md.content = f'🔄 검색 중: "{query}"'
        cls_idx, cls_name, sim = text_to_class(query, self.clip_model, self.device)
        matched = [(bbox, score) for cid, bbox, score in self.detections if cid == cls_idx]
        self.highlight(cls_idx)

        if matched:
            best_bbox, best_score = max(matched, key=lambda x: x[1])
            center, _ = get_center_size(best_bbox)
            self.result_md.content = (
                f'✅ **{cls_name}** (CLIP={sim:.3f})\n\n'
                f'- 감지 개수: {len(matched)}개\n'
                f'- 최고 score: {best_score:.3f}\n'
                f'- 원본 좌표: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})'
            )
        else:
            self.result_md.content = f'❌ **{cls_name}** 감지되지 않음 (CLIP={sim:.3f})'

    def update_detection_markdown(self):
        if not self.detections:
            self.det_md.content = '감지 결과가 없습니다.'
            return

        lines = []
        for cls_id, _bbox, score in self.detections:
            lines.append(f'- **{DC.class2type[cls_id]}** ({score:.2f})')
        self.det_md.content = '\n'.join(lines)

    def apply_room_camera(self, client: viser.ClientHandle):
        if self.camera_pos is None:
            return
        client.camera.position = tuple(self.camera_pos)
        client.camera.look_at = tuple(self.camera_look_at)
        client.camera.up_direction = self.camera_up

    def apply_room_camera_to_all_clients(self):
        for client in self.server.get_clients().values():
            try:
                self.apply_room_camera(client)
            except Exception:
                pass

    def apply_room_camera_repeated(self, client: viser.ClientHandle):
        # Viser client가 초기 카메라를 다시 덮는 경우가 있어서 여러 번 적용한다.
        for delay in (0.0, 0.2, 0.7, 1.5, 2.5):
            time.sleep(delay)
            try:
                self.apply_room_camera(client)
            except Exception:
                return

    def force_camera_after_load(self):
        def worker():
            for delay in (0.1, 0.3, 0.8, 1.6):
                time.sleep(delay)
                self.apply_room_camera_to_all_clients()

        threading.Thread(target=worker, daemon=True).start()

    def run(self):
        # 최초 맵 자동 로드
        self.load_scene(self.args.ply_path)
        print('✅ 실행 중. 종료하려면 Ctrl+C')
        while True:
            time.sleep(0.1)


def main():
    args = parse_args()
    app = ViserSceneApp(args)
    app.run()


if __name__ == '__main__':
    main()
