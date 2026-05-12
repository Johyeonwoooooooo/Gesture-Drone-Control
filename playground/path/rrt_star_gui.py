"""
RRT* Drone Path Planner - Interactive GUI
- 포인트 클라우드를 2D 탑뷰/사이드뷰로 시각화
- 마우스 클릭으로 시작/끝 위치 선택
- RRT* 알고리즘 자동 실행 및 경로 시각화
"""

import os
import sys
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

# ──────────────────────────────────────────
# rrt_star_drone 모듈 import
# ──────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

try:
    from rrt_star_drone import (
        load_pointcloud, build_obstacle_tree,
        plan_path, RRTStar, smooth_path, interpolate_path,
        find_nearest_free, OBSTACLE_RADIUS,
        _DEFAULT_COORD, _DEFAULT_COLOR, _DEFAULT_NORMAL
    )
    print("✓ rrt_star_drone 모듈 로드 성공")
except ImportError as e:
    print(f"✗ rrt_star_drone.py를 찾을 수 없습니다: {e}")
    print("  rrt_star_drone.py와 같은 폴더에 이 파일을 놓아주세요.")
    sys.exit(1)


# ──────────────────────────────────────────
# 색상 팔레트
# ──────────────────────────────────────────
COL = {
    'bg':        '#0d1117',
    'panel':     '#161b22',
    'accent':    '#58a6ff',
    'green':     '#3fb950',
    'red':       '#f85149',
    'yellow':    '#e3b341',
    'purple':    '#bc8cff',
    'text':      '#c9d1d9',
    'subtext':   '#8b949e',
    'grid':      '#21262d',
    'start':     '#00ff88',
    'goal':      '#ff6b6b',
    'path_raw':  '#4d9de0',
    'path_smo':  '#ff9f1c',
    'path_den':  '#e84855',
}

plt.rcParams.update({
    'figure.facecolor':  COL['bg'],
    'axes.facecolor':    COL['panel'],
    'axes.edgecolor':    COL['grid'],
    'axes.labelcolor':   COL['text'],
    'xtick.color':       COL['subtext'],
    'ytick.color':       COL['subtext'],
    'text.color':        COL['text'],
    'grid.color':        COL['grid'],
    'grid.linewidth':    0.5,
    'font.family':       'monospace',
})


# ──────────────────────────────────────────
# GUI 클래스
# ──────────────────────────────────────────

class DronePathPlannerGUI:
    def __init__(self):
        self.coord      = None
        self.color      = None
        self.kd_tree    = None
        self.bounds     = None

        self.start_pos  = None   # [x, y, z]
        self.goal_pos   = None   # [x, y, z]
        self.select_mode = 'start'  # 'start' | 'goal'

        self.result     = None
        self.is_running = False

        # Z 슬라이더 값 (선택 고도)
        self.z_value    = 1.5

        self._build_ui()
        self._load_data()

    # ── UI 구성 ──────────────────────────────

    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9), facecolor=COL['bg'])
        self.fig.canvas.manager.set_window_title('RRT* Drone Path Planner')

        gs = GridSpec(
            3, 3,
            figure=self.fig,
            left=0.05, right=0.98,
            top=0.93, bottom=0.12,
            hspace=0.35, wspace=0.3,
        )

        # 메인 탑뷰 (XY)
        self.ax_top  = self.fig.add_subplot(gs[0:2, 0:2])
        # 사이드뷰 (XZ)
        self.ax_side = self.fig.add_subplot(gs[2, 0:2])
        # 정보 패널
        self.ax_info = self.fig.add_subplot(gs[0:3, 2])

        self._style_axes()

        # 타이틀
        self.fig.text(
            0.5, 0.97,
            '✈  RRT* Drone Path Planner  ✈',
            ha='center', va='top', fontsize=14,
            color=COL['accent'], fontweight='bold', fontfamily='monospace',
        )

        # Z 고도 슬라이더
        ax_z = self.fig.add_axes([0.07, 0.04, 0.50, 0.025],
                                  facecolor=COL['panel'])
        self.slider_z = Slider(
            ax_z, 'Z 고도 (m)', 0.0, 5.0,
            valinit=self.z_value, color=COL['accent'],
        )
        self.slider_z.label.set_color(COL['text'])
        self.slider_z.valtext.set_color(COL['accent'])
        self.slider_z.on_changed(self._on_z_change)

        # 버튼들
        ax_run  = self.fig.add_axes([0.60, 0.04, 0.10, 0.04])
        ax_clr  = self.fig.add_axes([0.71, 0.04, 0.10, 0.04])
        ax_sav  = self.fig.add_axes([0.82, 0.04, 0.10, 0.04])
        ax_mod  = self.fig.add_axes([0.60, 0.005, 0.32, 0.03])
        for _ax in [ax_run, ax_clr, ax_sav, ax_mod]:
            _ax.set_facecolor(COL['panel'])

        self.btn_run = Button(ax_run, '▶ start', color=COL['green'],    hovercolor='#5dbb6a')
        self.btn_clr = Button(ax_clr, '✕ reset',    color='#30363d',       hovercolor='#3d444d')
        self.btn_sav = Button(ax_sav, '💾 save',      color='#30363d',       hovercolor='#3d444d')

        for b, lbl in [(self.btn_run, '▶ strart'),
                       (self.btn_clr, '✕ reset'),
                       (self.btn_sav, '💾 save')]:
            b.label.set_fontsize(9)
            b.label.set_color('white')

        self.btn_run.on_clicked(self._on_run)
        self.btn_clr.on_clicked(self._on_clear)
        self.btn_sav.on_clicked(self._on_save)

        # 선택 모드 라디오
        self.radio = RadioButtons(
            ax_mod,
            ('🟢  start', '🔴  goal'),
            activecolor=COL['accent'],
        )
        for lbl in self.radio.labels:
            lbl.set_color(COL['text'])
            lbl.set_fontsize(9)
        self.radio.on_clicked(self._on_mode_change)

        # 클릭 이벤트
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # 상태 텍스트
        self.status_text = self.fig.text(
            0.5, 0.955, 'data loading...',
            ha='center', fontsize=9, color=COL['yellow'],
        )

    def _style_axes(self):
        for ax, title in [
            (self.ax_top,  'TOP VIEW  (X-Y)'),
            (self.ax_side, 'SIDE VIEW  (X-Z)'),
            (self.ax_info, 'INFO'),
        ]:
            ax.set_facecolor(COL['panel'])
            ax.tick_params(colors=COL['subtext'], labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(COL['grid'])
            ax.set_title(title, color=COL['accent'], fontsize=9, pad=6)
            ax.grid(True, alpha=0.3)

        self.ax_top.set_xlabel('X (m)', fontsize=8)
        self.ax_top.set_ylabel('Y (m)', fontsize=8)
        self.ax_side.set_xlabel('X (m)', fontsize=8)
        self.ax_side.set_ylabel('Z (m)', fontsize=8)
        self.ax_info.axis('off')

    # ── 데이터 로드 ──────────────────────────

    def _load_data(self):
        self._set_status('load point cloud...', COL['yellow'])
        t = threading.Thread(target=self._load_data_thread, daemon=True)
        t.start()

    def _load_data_thread(self):
        try:
            self.coord, self.color, _ = load_pointcloud(
                _DEFAULT_COORD, _DEFAULT_COLOR, _DEFAULT_NORMAL
            )
            margin = 0.2
            self.bounds = [
                [self.coord[:, 0].min() - margin, self.coord[:, 0].max() + margin],
                [self.coord[:, 1].min() - margin, self.coord[:, 1].max() + margin],
                [self.coord[:, 2].min() + OBSTACLE_RADIUS,
                 self.coord[:, 2].max() - OBSTACLE_RADIUS],
            ]
            self.z_value = float(np.mean(self.bounds[2]))
            self.slider_z.set_val(self.z_value)
            self.slider_z.valmin = float(self.bounds[2][0])
            self.slider_z.valmax = float(self.bounds[2][1])

            self.kd_tree = build_obstacle_tree(self.coord)
            self._draw_pointcloud()
            self._update_info()
            self._set_status(
                f'✓ loaded  {len(self.coord):,}pts  |  top veiw select',
                COL['green'],
            )
        except FileNotFoundError as e:
            self._set_status(f'✗ no files: {e}', COL['red'])
        except Exception as e:
            self._set_status(f'✗ error: {e}', COL['red'])

    def _draw_pointcloud(self):
        # 탑뷰 (XY)
        self.ax_top.cla()
        self._style_axes()

        if self.color is not None:
            cols = np.clip(self.color, 0, 1)
        else:
            cols = COL['subtext']

        step = max(1, len(self.coord) // 80_000)
        c_sub = self.coord[::step]
        col_sub = cols[::step] if self.color is not None else cols

        self.ax_top.scatter(
            c_sub[:, 0], c_sub[:, 1],
            c=col_sub, s=0.3, alpha=0.6, linewidths=0,
        )
        self.ax_top.set_xlim(self.bounds[0])
        self.ax_top.set_ylim(self.bounds[1])

        # 사이드뷰 (XZ)
        self.ax_side.cla()
        self._style_axes()
        self.ax_side.scatter(
            c_sub[:, 0], c_sub[:, 2],
            c=col_sub, s=0.3, alpha=0.6, linewidths=0,
        )
        self.ax_side.set_xlim(self.bounds[0])
        self.ax_side.set_ylim(self.bounds[2])

        # Z 라인
        self.z_line_top  = self.ax_top.axhline(y=0, color=COL['yellow'], lw=0.6,
                                                 ls='--', alpha=0.0)  # top엔 불필요
        self.z_line_side = self.ax_side.axhline(
            y=self.z_value, color=COL['yellow'], lw=1.0, ls='--', alpha=0.7,
            label=f'Z={self.z_value:.2f}m',
        )
        self.ax_side.legend(fontsize=7, loc='upper right',
                            facecolor=COL['panel'], labelcolor=COL['text'])

        self.fig.canvas.draw_idle()

    # ── 이벤트 핸들러 ────────────────────────

    def _on_z_change(self, val):
        self.z_value = val
        if hasattr(self, 'z_line_side'):
            self.z_line_side.set_ydata([val, val])
            self.z_line_side.set_label(f'Z={val:.2f}m')
            self.ax_side.legend(fontsize=7, loc='upper right',
                                facecolor=COL['panel'], labelcolor=COL['text'])
            self.fig.canvas.draw_idle()

    def _on_mode_change(self, label):
        self.select_mode = 'start' if 'start' in label else 'goal'

    def _on_click(self, event):
        if event.inaxes != self.ax_top:
            return
        if event.button != 1:
            return
        if self.coord is None or self.is_running:
            return

        x, y = event.xdata, event.ydata
        z = self.z_value
        pos = np.array([x, y, z])

        # 장애물 안이면 자동 보정
        free = find_nearest_free(pos, self.kd_tree, self.bounds)
        if free is None:
            self._set_status('✗ space no enough', COL['red'])
            return

        if not np.allclose(free, pos):
            self._set_status(
                f'⚠ obstacles deteced → automatic: ({free[0]:.2f}, {free[1]:.2f}, {free[2]:.2f})',
                COL['yellow'],
            )
            pos = free

        if self.select_mode == 'start':
            self.start_pos = pos
            self._set_status(
                f'🟢 start: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  |  select goal',
                COL['green'],
            )
        else:
            self.goal_pos = pos
            self._set_status(
                f'🔴 goal: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  |  ▶ start button',
                COL['red'],
            )

        self._redraw_markers()

    def _on_run(self, event):
        if self.is_running:
            return
        if self.start_pos is None or self.goal_pos is None:
            self._set_status('✗ must all select', COL['red'])
            return
        if self.kd_tree is None:
            self._set_status('✗ data not loaded yet', COL['red'])
            return

        self.is_running = True
        self._set_status('⏳ RRT* ... (5000 iter)', COL['yellow'])
        self.btn_run.label.set_text('...')
        self.fig.canvas.draw_idle()

        t = threading.Thread(target=self._run_planner, daemon=True)
        t.start()

    def _run_planner(self):
        try:
            result = plan_path(
                self.start_pos.tolist(),
                self.goal_pos.tolist(),
                verbose=True,
            )
            self.result = result
            if result:
                self._set_status(
                    f'✓ complete!  distance: {result["total_dist"]:.3f}m  |  '
                    f'wavepoint: {len(result["smooth_path"])}개',
                    COL['green'],
                )
                self._draw_result()
            else:
                self._set_status(
                    '✗ change start, end point',
                    COL['red'],
                )
        except Exception as e:
            self._set_status(f'✗ 오류: {e}', COL['red'])
        finally:
            self.is_running = False
            self.btn_run.label.set_text('▶ start')
            self.fig.canvas.draw_idle()

    def _on_clear(self, event):
        if self.is_running:
            return
        self.start_pos = None
        self.goal_pos  = None
        self.result    = None
        self._draw_pointcloud()
        self._update_info()
        self._set_status('reset | select', COL['accent'])

    def _on_save(self, event):
        if self.result is None:
            self._set_status('✗ 저장할 경로가 없습니다. 먼저 경로를 탐색하세요.', COL['red'])
            return
        save_path = os.path.join(_SCRIPT_DIR, 'waypoints.npy')
        np.save(save_path, self.result['smooth_path'])
        self._set_status(f'✓ 저장 완료: {save_path}', COL['green'])

    # ── 시각화 ───────────────────────────────

    def _redraw_markers(self):
        """시작/목표 마커만 다시 그림 (포인트 클라우드 유지)"""
        # 기존 마커 제거
        for attr in ['_start_scatter_top', '_start_scatter_side',
                     '_goal_scatter_top',  '_goal_scatter_side',
                     '_start_ann_top', '_goal_ann_top']:
            art = getattr(self, attr, None)
            if art is not None:
                try:
                    art.remove()
                except:
                    pass
            setattr(self, attr, None)

        mk_kw = dict(zorder=10, edgecolors='white', linewidths=1.2)

        if self.start_pos is not None:
            s = self.start_pos
            self._start_scatter_top  = self.ax_top.scatter(
                s[0], s[1], c=COL['start'], s=180, marker='^', **mk_kw)
            self._start_scatter_side = self.ax_side.scatter(
                s[0], s[2], c=COL['start'], s=120, marker='^', **mk_kw)
            self._start_ann_top = self.ax_top.annotate(
                f' START\n ({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})',
                (s[0], s[1]), fontsize=7, color=COL['start'],
                path_effects=[pe.withStroke(linewidth=2, foreground=COL['bg'])],
            )

        if self.goal_pos is not None:
            g = self.goal_pos
            self._goal_scatter_top  = self.ax_top.scatter(
                g[0], g[1], c=COL['goal'], s=180, marker='*', **mk_kw)
            self._goal_scatter_side = self.ax_side.scatter(
                g[0], g[2], c=COL['goal'], s=120, marker='*', **mk_kw)
            self._goal_ann_top = self.ax_top.annotate(
                f' GOAL\n ({g[0]:.2f},{g[1]:.2f},{g[2]:.2f})',
                (g[0], g[1]), fontsize=7, color=COL['goal'],
                path_effects=[pe.withStroke(linewidth=2, foreground=COL['bg'])],
            )

        self.fig.canvas.draw_idle()

    def _draw_result(self):
        if self.result is None:
            return

        raw    = self.result['raw_path']
        smooth = self.result['smooth_path']
        dense  = self.result['dense_path']

        # 탑뷰 경로
        self.ax_top.plot(raw[:, 0],    raw[:, 1],    color=COL['path_raw'],
                         lw=0.8, alpha=0.4, ls='--', label='Raw')
        self.ax_top.plot(smooth[:, 0], smooth[:, 1], color=COL['path_smo'],
                         lw=2.0, alpha=0.9, label='Smooth')
        self.ax_top.scatter(smooth[:, 0], smooth[:, 1],
                            c=COL['path_smo'], s=25, zorder=8, edgecolors='white', lw=0.5)
        self.ax_top.legend(fontsize=7, loc='upper right',
                           facecolor=COL['panel'], labelcolor=COL['text'])

        # 사이드뷰 경로
        self.ax_side.plot(smooth[:, 0], smooth[:, 2], color=COL['path_smo'],
                          lw=2.0, alpha=0.9)
        self.ax_side.scatter(smooth[:, 0], smooth[:, 2],
                             c=COL['path_smo'], s=20, zorder=8, edgecolors='white', lw=0.5)

        self._redraw_markers()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _update_info(self):
        self.ax_info.cla()
        self.ax_info.axis('off')
        self.ax_info.set_title('INFO', color=COL['accent'], fontsize=9, pad=6)

        lines = []

        if self.bounds:
            lines += [
                ('SPACE BOUNDS', COL['accent'], True),
                (f"X: {self.bounds[0][0]:.2f} ~ {self.bounds[0][1]:.2f} m", COL['text'], False),
                (f"Y: {self.bounds[1][0]:.2f} ~ {self.bounds[1][1]:.2f} m", COL['text'], False),
                (f"Z: {self.bounds[2][0]:.2f} ~ {self.bounds[2][1]:.2f} m", COL['text'], False),
                ('', '', False),
            ]

        if self.coord is not None:
            lines += [
                ('POINT CLOUD', COL['accent'], True),
                (f"Points: {len(self.coord):,}", COL['text'], False),
                ('', '', False),
            ]

        lines += [
            ('POSITIONS', COL['accent'], True),
        ]
        if self.start_pos is not None:
            s = self.start_pos
            lines += [
                ('Start:', COL['start'], True),
                (f"  X={s[0]:.3f}", COL['text'], False),
                (f"  Y={s[1]:.3f}", COL['text'], False),
                (f"  Z={s[2]:.3f}", COL['text'], False),
            ]
        else:
            lines.append(('Start: (미선택)', COL['subtext'], False))

        if self.goal_pos is not None:
            g = self.goal_pos
            lines += [
                ('Goal:', COL['goal'], True),
                (f"  X={g[0]:.3f}", COL['text'], False),
                (f"  Y={g[1]:.3f}", COL['text'], False),
                (f"  Z={g[2]:.3f}", COL['text'], False),
            ]
        else:
            lines.append(('Goal:  (미선택)', COL['subtext'], False))

        if self.result:
            r = self.result
            lines += [
                ('', '', False),
                ('RESULT', COL['accent'], True),
                (f"Distance: {r['total_dist']:.3f} m", COL['green'], False),
                (f"Nodes:    {r['n_nodes']:,}",         COL['text'],  False),
                (f"Raw WPs:  {len(r['raw_path'])}",     COL['text'],  False),
                (f"Smooth WPs: {len(r['smooth_path'])}", COL['path_smo'], False),
                (f"Dense pts:  {len(r['dense_path'])}",  COL['text'],  False),
                ('', '', False),
                ('WAYPOINTS (smooth)', COL['accent'], True),
            ]
            for i, wp in enumerate(r['smooth_path']):
                lines.append(
                    (f"[{i:2d}] {wp[0]:6.2f} {wp[1]:6.2f} {wp[2]:5.2f}",
                     COL['path_smo'], False)
                )

        y = 0.98
        dy = 0.038
        for text, color, bold in lines:
            if text == '':
                y -= dy * 0.5
                continue
            self.ax_info.text(
                0.03, y, text,
                transform=self.ax_info.transAxes,
                fontsize=7.5 if bold else 7,
                color=color,
                fontweight='bold' if bold else 'normal',
                va='top', fontfamily='monospace',
            )
            y -= dy
            if y < 0.01:
                break

        self.fig.canvas.draw_idle()

    # ── 유틸 ─────────────────────────────────

    def _set_status(self, msg, color):
        self.status_text.set_text(msg)
        self.status_text.set_color(color)
        try:
            self.fig.canvas.draw_idle()
        except:
            pass

    def show(self):
        plt.show()


# ──────────────────────────────────────────
# Entry
# ──────────────────────────────────────────

if __name__ == '__main__':
    gui = DronePathPlannerGUI()
    gui.show()