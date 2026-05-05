"""
benchmark.py - 경로 탐색 알고리즘 체계적 평가

실행: python benchmark.py

평가 항목:
  1. 성공률 (Success Rate) - 경로를 찾았는가?
  2. 경로 길이 (Path Length) - 최적에 가까운가?
  3. 탐색 시간 (Computation Time) - 얼마나 빠른가?
  4. 경로 안전성 (Clearance) - 장애물에서 얼마나 떨어졌는가?
  5. 경로 부드러움 (Smoothness) - 급격한 방향전환이 있는가?

테스트 환경:
  - 장애물 밀도: sparse(10%) → dense(40%)
  - 맵 패턴: 랜덤, 미로형, 좁은 통로, 함정(local minima)
  - 시작/목표 거리: 가까운 곳 ~ 맵 대각선
"""

import numpy as np
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import math

from maps import GridMap2D
from astar import AStarPlanner
from rrt import RRTStarPlanner
from core import Map, Point, Planner


# ══════════════════════════════════════════
# 평가 메트릭
# ══════════════════════════════════════════

@dataclass
class TestResult:
    """단일 테스트 결과"""
    map_type: str           # 맵 종류
    obstacle_density: float # 장애물 비율
    algorithm: str          # 알고리즘 이름
    success: bool           # 경로 찾았는가
    path_length: float      # 경로 총 길이
    optimal_ratio: float    # 직선거리 대비 경로 비율 (1.0에 가까울수록 좋음)
    compute_time: float     # 탐색 시간 (초)
    num_waypoints: int      # waypoint 수
    min_clearance: float    # 장애물과의 최소 거리
    smoothness: float       # 방향 변화량 (작을수록 부드러움)
    start_goal_dist: float  # 시작-목표 직선거리


def compute_path_length(path: List[Point]) -> float:
    total = 0
    for i in range(len(path) - 1):
        total += math.sqrt(sum((a - b) ** 2 for a, b in zip(path[i], path[i+1])))
    return total


def compute_min_clearance(path: List[Point], map_: Map) -> float:
    """경로의 각 점에서 가장 가까운 장애물까지의 거리"""
    if not path:
        return 0

    min_dist = float('inf')
    # 맵 전체를 스캔하는 건 비효율적이니 주변만 탐색
    for point in path:
        for r in range(1, 10):
            found_obstacle = False
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    check = (point[0] + dx, point[1] + dy)
                    if len(point) == 3:
                        for dz in range(-r, r + 1):
                            check = (point[0] + dx, point[1] + dy, point[2] + dz)
                            if map_.is_in_bounds(check) and not map_.is_free(check):
                                d = math.sqrt(dx**2 + dy**2 + dz**2)
                                min_dist = min(min_dist, d)
                                found_obstacle = True
                    else:
                        if map_.is_in_bounds(check) and not map_.is_free(check):
                            d = math.sqrt(dx**2 + dy**2)
                            min_dist = min(min_dist, d)
                            found_obstacle = True
            if found_obstacle:
                break

    return min_dist if min_dist != float('inf') else -1


def compute_smoothness(path: List[Point]) -> float:
    """연속 세 점 사이의 각도 변화 합 (작을수록 부드러움)"""
    if len(path) < 3:
        return 0

    total_angle = 0
    for i in range(len(path) - 2):
        v1 = np.array(path[i+1]) - np.array(path[i])
        v2 = np.array(path[i+2]) - np.array(path[i+1])

        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        total_angle += abs(math.acos(cos_angle))

    return math.degrees(total_angle)


def straight_line_dist(a: Point, b: Point) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ══════════════════════════════════════════
# 다양한 맵 생성기
# ══════════════════════════════════════════

def make_random_map(size: int, density: float) -> GridMap2D:
    """랜덤 장애물 맵"""
    m = GridMap2D(size, size)
    m.grid = (np.random.random((size, size)) < density).astype(np.int8)
    # 모서리 근처는 비워둠 (시작/목표 배치용)
    m.grid[:5, :5] = 0
    m.grid[-5:, -5:] = 0
    return m


def make_maze_map(size: int) -> GridMap2D:
    """미로형 맵 (DFS 기반)"""
    m = GridMap2D(size, size)
    m.grid[:] = 1  # 전부 벽으로 시작

    # DFS로 미로 생성
    cell_size = 3
    cols = size // cell_size
    rows = size // cell_size
    visited = set()
    stack = [(0, 0)]
    visited.add((0, 0))

    while stack:
        cx, cy = stack[-1]
        # 현재 셀 열기
        for dx in range(cell_size):
            for dy in range(cell_size):
                px, py = cx * cell_size + dx, cy * cell_size + dy
                if 0 <= px < size and 0 <= py < size:
                    m.grid[py, px] = 0

        # 이웃 탐색
        neighbors = []
        for nx, ny in [(cx+1,cy), (cx-1,cy), (cx,cy+1), (cx,cy-1)]:
            if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in visited:
                neighbors.append((nx, ny))

        if neighbors:
            next_cell = neighbors[np.random.randint(len(neighbors))]
            visited.add(next_cell)
            stack.append(next_cell)
        else:
            stack.pop()

    return m


def make_narrow_passage_map(size: int) -> GridMap2D:
    """좁은 통로 맵 - 벽 사이에 1~2칸 통로"""
    m = GridMap2D(size, size)

    num_walls = 4
    for i in range(num_walls):
        wall_x = (i + 1) * size // (num_walls + 1)
        gap_y = np.random.randint(5, size - 5)
        gap_size = np.random.randint(1, 3)  # 통로 폭

        m.grid[:, wall_x:wall_x+2] = 1
        m.grid[gap_y:gap_y+gap_size, wall_x:wall_x+2] = 0

    return m


def make_trap_map(size: int) -> GridMap2D:
    """함정 맵 - Potential Field가 빠지기 쉬운 local minima"""
    m = GridMap2D(size, size)

    # 목표 앞에 U자형 장벽
    cx, cy = size // 2, size // 2
    wall_len = size // 3

    # U자 아래쪽
    m.grid[cy:cy+2, cx-wall_len//2:cx+wall_len//2] = 1
    # U자 왼쪽
    m.grid[cy:cy+wall_len, cx-wall_len//2:cx-wall_len//2+2] = 1
    # U자 오른쪽
    m.grid[cy:cy+wall_len, cx+wall_len//2-2:cx+wall_len//2] = 1

    return m


def make_cluttered_map(size: int, num_obstacles: int = 15) -> GridMap2D:
    """원형 장애물 여러 개 (가구/기둥 시뮬레이션)"""
    m = GridMap2D(size, size)
    for _ in range(num_obstacles):
        cx = np.random.randint(8, size - 8)
        cy = np.random.randint(8, size - 8)
        r = np.random.randint(2, 6)
        m.add_obstacle_circle(cx, cy, r)
    m.grid[:5, :5] = 0
    m.grid[-5:, -5:] = 0
    return m


# ══════════════════════════════════════════
# 유효한 시작/목표 찾기
# ══════════════════════════════════════════

def find_valid_endpoints(map_: GridMap2D, min_dist: float = 20) -> Optional[tuple]:
    """장애물 위가 아닌 시작/목표 쌍을 찾기"""
    for _ in range(200):
        sx = np.random.randint(2, map_.width // 4)
        sy = np.random.randint(2, map_.height // 4)
        gx = np.random.randint(map_.width * 3 // 4, map_.width - 2)
        gy = np.random.randint(map_.height * 3 // 4, map_.height - 2)

        if (map_.is_free((sx, sy)) and map_.is_free((gx, gy))
                and straight_line_dist((sx,sy), (gx,gy)) >= min_dist):
            return (sx, sy), (gx, gy)
    return None


# ══════════════════════════════════════════
# 벤치마크 실행
# ══════════════════════════════════════════

def run_single_test(planner: Planner, algo_name: str,
                    map_: GridMap2D, map_type: str,
                    density: float, start: Point, goal: Point) -> TestResult:
    """단일 테스트 실행"""
    sg_dist = straight_line_dist(start, goal)

    t0 = time.time()
    path = planner.plan(map_, start, goal)
    elapsed = time.time() - t0

    if path:
        pl = compute_path_length(path)
        return TestResult(
            map_type=map_type,
            obstacle_density=density,
            algorithm=algo_name,
            success=True,
            path_length=round(pl, 2),
            optimal_ratio=round(pl / sg_dist, 3) if sg_dist > 0 else 1.0,
            compute_time=round(elapsed, 4),
            num_waypoints=len(path),
            min_clearance=round(compute_min_clearance(path, map_), 2),
            smoothness=round(compute_smoothness(path), 2),
            start_goal_dist=round(sg_dist, 2),
        )
    else:
        return TestResult(
            map_type=map_type,
            obstacle_density=density,
            algorithm=algo_name,
            success=False,
            path_length=0, optimal_ratio=0, compute_time=round(elapsed, 4),
            num_waypoints=0, min_clearance=0, smoothness=0,
            start_goal_dist=round(sg_dist, 2),
        )


def run_benchmark(trials_per_config: int = 20, map_size: int = 60):
    """전체 벤치마크 실행"""

    planners = {
        'A*': AStarPlanner(),
        'RRT*': RRTStarPlanner(max_iter=5000, step_size=3.0, goal_sample_rate=0.1),
    }

    # 테스트 구성: (맵 생성 함수, 이름, 장애물 밀도)
    configs = [
        # 랜덤 맵 - 밀도별
        (lambda: make_random_map(map_size, 0.10), "random", 0.10),
        (lambda: make_random_map(map_size, 0.20), "random", 0.20),
        (lambda: make_random_map(map_size, 0.30), "random", 0.30),
        (lambda: make_random_map(map_size, 0.40), "random", 0.40),
        # 패턴 맵
        (lambda: make_maze_map(map_size), "maze", 0.50),
        (lambda: make_narrow_passage_map(map_size), "narrow_passage", 0.15),
        (lambda: make_trap_map(map_size), "trap", 0.10),
        (lambda: make_cluttered_map(map_size, 15), "cluttered", 0.20),
    ]

    results: List[TestResult] = []
    total_tests = len(configs) * trials_per_config * len(planners)
    test_count = 0

    print(f"\n{'='*65}")
    print(f"  Pathfinding Benchmark")
    print(f"  Map size: {map_size}x{map_size}  |  Trials per config: {trials_per_config}")
    print(f"  Total tests: {total_tests}")
    print(f"{'='*65}\n")

    for map_gen, map_type, density in configs:
        print(f"  [{map_type}] density={density:.0%} ", end='', flush=True)

        for trial in range(trials_per_config):
            map_ = map_gen()
            endpoints = find_valid_endpoints(map_)
            if endpoints is None:
                continue
            start, goal = endpoints

            for algo_name, planner in planners.items():
                result = run_single_test(planner, algo_name, map_, map_type, density, start, goal)
                results.append(result)
                test_count += 1

        print(f"  ✓ ({test_count}/{total_tests})")

    return results


# ══════════════════════════════════════════
# 결과 분석 & 리포트
# ══════════════════════════════════════════

def analyze_results(results: List[TestResult]):
    """결과 통계 분석"""

    print(f"\n{'='*80}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*80}")

    # 알고리즘별 전체 통계
    for algo in ['A*', 'RRT*']:
        algo_results = [r for r in results if r.algorithm == algo]
        successes = [r for r in algo_results if r.success]
        success_rate = len(successes) / len(algo_results) * 100 if algo_results else 0

        print(f"\n  ── {algo} Overall ──")
        print(f"  Success Rate:     {success_rate:.1f}% ({len(successes)}/{len(algo_results)})")

        if successes:
            avg_time = np.mean([r.compute_time for r in successes])
            avg_ratio = np.mean([r.optimal_ratio for r in successes])
            avg_clearance = np.mean([r.min_clearance for r in successes])
            avg_smooth = np.mean([r.smoothness for r in successes])

            print(f"  Avg Time:         {avg_time:.4f}s")
            print(f"  Avg Path Ratio:   {avg_ratio:.3f}x (1.0 = straight line)")
            print(f"  Avg Clearance:    {avg_clearance:.2f} cells")
            print(f"  Avg Smoothness:   {avg_smooth:.1f}° total turning")

    # 맵 타입별 비교
    print(f"\n{'='*80}")
    print(f"  BY MAP TYPE")
    print(f"{'='*80}")

    map_types = sorted(set(r.map_type for r in results))

    header = f"  {'Map Type':<18} {'Algo':<6} {'Success':>8} {'Time':>10} {'Ratio':>8} {'Clear':>8} {'Smooth':>8}"
    print(header)
    print(f"  {'-'*76}")

    for mt in map_types:
        for algo in ['A*', 'RRT*']:
            mr = [r for r in results if r.map_type == mt and r.algorithm == algo]
            succ = [r for r in mr if r.success]
            sr = len(succ) / len(mr) * 100 if mr else 0

            if succ:
                at = np.mean([r.compute_time for r in succ])
                ar = np.mean([r.optimal_ratio for r in succ])
                ac = np.mean([r.min_clearance for r in succ])
                asm = np.mean([r.smoothness for r in succ])
                print(f"  {mt:<18} {algo:<6} {sr:>7.0f}% {at:>9.4f}s {ar:>7.3f}x {ac:>7.2f} {asm:>7.1f}°")
            else:
                print(f"  {mt:<18} {algo:<6} {sr:>7.0f}%       -         -       -       -")

    # 밀도별 성공률 변화
    print(f"\n{'='*80}")
    print(f"  SUCCESS RATE BY DENSITY (random maps)")
    print(f"{'='*80}")

    random_results = [r for r in results if r.map_type == 'random']
    densities = sorted(set(r.obstacle_density for r in random_results))

    for d in densities:
        for algo in ['A*', 'RRT*']:
            dr = [r for r in random_results if r.obstacle_density == d and r.algorithm == algo]
            sr = sum(r.success for r in dr) / len(dr) * 100 if dr else 0
            bar = '█' * int(sr / 5) + '░' * (20 - int(sr / 5))
            print(f"  {d:.0%} {algo:<6} {bar} {sr:.0f}%")

    return results


def save_results(results: List[TestResult], filename: str = "benchmark_results.json"):
    """결과를 JSON으로 저장"""
    data = [asdict(r) for r in results]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {filename}")


# ══════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════

def plot_results(results: List[TestResult], save_path: str = "benchmark_charts.png"):
    """결과 차트 생성"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 밀도별 성공률
    ax = axes[0, 0]
    random_results = [r for r in results if r.map_type == 'random']
    densities = sorted(set(r.obstacle_density for r in random_results))

    for algo in ['A*', 'RRT*']:
        rates = []
        for d in densities:
            dr = [r for r in random_results if r.obstacle_density == d and r.algorithm == algo]
            rates.append(sum(r.success for r in dr) / len(dr) * 100 if dr else 0)
        ax.plot([d*100 for d in densities], rates, 'o-', label=algo, linewidth=2)

    ax.set_xlabel('Obstacle Density (%)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate vs Obstacle Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # 2. 맵 타입별 성공률
    ax = axes[0, 1]
    map_types = sorted(set(r.map_type for r in results))
    x = np.arange(len(map_types))
    width = 0.35

    for i, algo in enumerate(['A*', 'RRT*']):
        rates = []
        for mt in map_types:
            mr = [r for r in results if r.map_type == mt and r.algorithm == algo]
            rates.append(sum(r.success for r in mr) / len(mr) * 100 if mr else 0)
        ax.bar(x + i * width, rates, width, label=algo)

    ax.set_xlabel('Map Type')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Map Type')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(map_types, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 탐색 시간 비교
    ax = axes[1, 0]
    for algo in ['A*', 'RRT*']:
        succ = [r for r in results if r.algorithm == algo and r.success]
        if succ:
            times = [r.compute_time for r in succ]
            ax.hist(times, bins=30, alpha=0.6, label=f'{algo} (avg={np.mean(times):.4f}s)')
    ax.set_xlabel('Computation Time (s)')
    ax.set_ylabel('Count')
    ax.set_title('Computation Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 경로 최적성 비교
    ax = axes[1, 1]
    for algo in ['A*', 'RRT*']:
        succ = [r for r in results if r.algorithm == algo and r.success]
        if succ:
            ratios = [r.optimal_ratio for r in succ]
            ax.hist(ratios, bins=30, alpha=0.6, label=f'{algo} (avg={np.mean(ratios):.3f}x)')
    ax.set_xlabel('Path Length / Straight Line Distance')
    ax.set_ylabel('Count')
    ax.set_title('Path Optimality (closer to 1.0 = better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Charts saved to {save_path}")
    plt.close()


# ══════════════════════════════════════════

if __name__ == '__main__':
    results = run_benchmark(trials_per_config=20, map_size=60)
    analyze_results(results)
    save_results(results)
    plot_results(results)