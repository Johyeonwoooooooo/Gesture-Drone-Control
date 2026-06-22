"""
RRT* Drone Path Planner
- Input: Point Cloud (coord.npy, color.npy, normal.npy)
- Drone size: 20cm x 20cm
- Algorithm: RRT* (Rapidly-exploring Random Tree Star)
"""

import os
import time
import numpy as np
from scipy.spatial import KDTree

# ──────────────────────────────────────────
# File paths (relative to this file)
# ──────────────────────────────────────────
_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_COORD  = os.path.join(_BASE_DIR, 'map_data', 'coord.npy')
_DEFAULT_COLOR  = os.path.join(_BASE_DIR, 'map_data', 'color.npy')
_DEFAULT_NORMAL = os.path.join(_BASE_DIR, 'map_data', 'normal.npy')

# ──────────────────────────────────────────
# 1. Parameters
# ──────────────────────────────────────────
DRONE_HALF_SIZE = 0.10
DRONE_RADIUS    = DRONE_HALF_SIZE * np.sqrt(2) * 1.3  # ~0.185m
SAFETY_MARGIN   = 0.05
OBSTACLE_RADIUS = DRONE_RADIUS + SAFETY_MARGIN         # ~0.235m

MAX_ITER      = 5000
STEP_SIZE     = 0.30
GOAL_RADIUS   = 0.25
REWIRE_RADIUS = 0.80
GOAL_BIAS     = 0.10

# ──────────────────────────────────────────
# 2. Point cloud loader
# ──────────────────────────────────────────

def load_pointcloud(coord_path, color_path=None, normal_path=None):
    coord  = np.load(coord_path).astype(np.float32)
    color  = np.load(color_path).astype(np.float32) / 255.0 if color_path else None
    normal = np.load(normal_path).astype(np.float32) if normal_path else None
    return coord, color, normal


def build_obstacle_tree(coord):
    print(f"  Points: {len(coord):,}")
    tree = KDTree(coord)
    print("  KD-Tree built")
    return tree


# ──────────────────────────────────────────
# 3. Collision check
# ──────────────────────────────────────────

def is_point_free(point, kd_tree, radius=OBSTACLE_RADIUS):
    indices = kd_tree.query_ball_point(point, radius)
    return len(indices) == 0


def is_edge_free(p1, p2, kd_tree, n_checks=None, radius=OBSTACLE_RADIUS):
    dist = np.linalg.norm(p2 - p1)
    if n_checks is None:
        n_checks = max(3, int(dist / (radius * 0.5)))
    for t in np.linspace(0, 1, n_checks):
        p = p1 + t * (p2 - p1)
        if not is_point_free(p, kd_tree, radius):
            return False
    return True


def find_nearest_free(point, kd_tree, bounds, radius=OBSTACLE_RADIUS, max_attempts=500):
    """
    If the given point is inside an obstacle,
    search nearby until a free position is found.
    Samples random offsets in increasing radius.
    """
    if is_point_free(point, kd_tree, radius):
        return point.copy()

    print(f"  Position {np.round(point,3)} is inside obstacle. Searching nearby free space...")

    rng = np.random.default_rng(42)
    for step in np.linspace(radius, 1.5, max_attempts):
        offset    = rng.uniform(-step, step, size=3)
        candidate = point + offset
        # clamp to bounds
        candidate[0] = np.clip(candidate[0], bounds[0][0], bounds[0][1])
        candidate[1] = np.clip(candidate[1], bounds[1][0], bounds[1][1])
        candidate[2] = np.clip(candidate[2], bounds[2][0], bounds[2][1])
        if is_point_free(candidate, kd_tree, radius):
            print(f"  -> Adjusted to {np.round(candidate,3)} (offset {step:.3f}m)")
            return candidate

    print("  -> Could not find free space near this position.")
    return None


# ──────────────────────────────────────────
# 4. RRT* Algorithm
# ──────────────────────────────────────────

class Node:
    __slots__ = ['pos', 'parent', 'cost']
    def __init__(self, pos, parent=None, cost=0.0):
        self.pos    = np.array(pos, dtype=np.float64)
        self.parent = parent
        self.cost   = cost


class RRTStar:
    def __init__(self, start, goal, kd_tree, bounds,
                 step_size=STEP_SIZE, max_iter=MAX_ITER,
                 goal_radius=GOAL_RADIUS, rewire_radius=REWIRE_RADIUS,
                 goal_bias=GOAL_BIAS, obstacle_radius=OBSTACLE_RADIUS):
        self.start         = np.array(start, dtype=np.float64)
        self.goal          = np.array(goal,  dtype=np.float64)
        self.kd_tree       = kd_tree
        self.bounds        = bounds
        self.step_size     = step_size
        self.max_iter      = max_iter
        self.goal_radius   = goal_radius
        self.rewire_radius = rewire_radius
        self.goal_bias     = goal_bias
        self.obs_radius    = obstacle_radius

        self.nodes           = [Node(self.start, parent=None, cost=0.0)]
        self._node_positions = [self.start.copy()]
        self.best_goal_node  = None
        self.best_cost       = np.inf

    def _sample_random(self):
        if np.random.rand() < self.goal_bias:
            return self.goal.copy()
        return np.array([
            np.random.uniform(self.bounds[0][0], self.bounds[0][1]),
            np.random.uniform(self.bounds[1][0], self.bounds[1][1]),
            np.random.uniform(self.bounds[2][0], self.bounds[2][1]),
        ])

    def _nearest(self, pos):
        dists = np.linalg.norm(np.array(self._node_positions) - pos, axis=1)
        idx   = int(np.argmin(dists))
        return self.nodes[idx], idx

    def _steer(self, from_pos, to_pos):
        diff = to_pos - from_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return from_pos.copy()
        if dist <= self.step_size:
            return to_pos.copy()
        return from_pos + (diff / dist) * self.step_size

    def _near_nodes(self, pos, radius):
        arr   = np.array(self._node_positions)
        dists = np.linalg.norm(arr - pos, axis=1)
        return np.where(dists <= radius)[0].tolist()

    def _cost(self, pos1, pos2):
        return float(np.linalg.norm(pos1 - pos2))

    def plan(self, verbose=True):
        start_time = time.time()
        found_iter = None

        for i in range(self.max_iter):
            q_rand          = self._sample_random()
            nearest_node, _ = self._nearest(q_rand)
            q_new           = self._steer(nearest_node.pos, q_rand)

            if not is_point_free(q_new, self.kd_tree, self.obs_radius):
                continue
            if not is_edge_free(nearest_node.pos, q_new, self.kd_tree, radius=self.obs_radius):
                continue

            near_idxs   = self._near_nodes(q_new, self.rewire_radius)
            best_parent = nearest_node
            best_cost   = nearest_node.cost + self._cost(nearest_node.pos, q_new)

            for idx in near_idxs:
                n = self.nodes[idx]
                c = n.cost + self._cost(n.pos, q_new)
                if c < best_cost:
                    if is_edge_free(n.pos, q_new, self.kd_tree, radius=self.obs_radius):
                        best_cost   = c
                        best_parent = n

            new_node = Node(q_new, parent=best_parent, cost=best_cost)
            self.nodes.append(new_node)
            self._node_positions.append(q_new.copy())

            for idx in near_idxs:
                n        = self.nodes[idx]
                new_cost = new_node.cost + self._cost(new_node.pos, n.pos)
                if new_cost < n.cost:
                    if is_edge_free(new_node.pos, n.pos, self.kd_tree, radius=self.obs_radius):
                        n.parent = new_node
                        n.cost   = new_cost

            dist_to_goal = self._cost(q_new, self.goal)
            if dist_to_goal <= self.goal_radius:
                total_cost = new_node.cost + dist_to_goal
                if total_cost < self.best_cost:
                    self.best_goal_node = Node(self.goal, parent=new_node, cost=total_cost)
                    self.best_cost      = total_cost
                    if found_iter is None:
                        found_iter = i
                        if verbose:
                            print(f"  Path found! (iter {i}, cost={total_cost:.3f}m)")

            if verbose and (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                status  = f"cost={self.best_cost:.3f}m" if self.best_cost < np.inf else "not found yet"
                print(f"  iter {i+1:5d} | nodes {len(self.nodes):5d} | {status} | {elapsed:.1f}s")

        if verbose:
            elapsed = time.time() - start_time
            print(f"\n  Done: {elapsed:.2f}s | total nodes {len(self.nodes)}")

        return self._extract_path()

    def _extract_path(self):
        if self.best_goal_node is None:
            return None
        path = []
        node = self.best_goal_node
        while node is not None:
            path.append(node.pos.copy())
            node = node.parent
        path.reverse()
        return np.array(path)


# ──────────────────────────────────────────
# 5. Path post-processing
# ──────────────────────────────────────────

def smooth_path(path, kd_tree, obs_radius=OBSTACLE_RADIUS):
    if path is None or len(path) < 3:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if is_edge_free(path[i], path[j], kd_tree, radius=obs_radius):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return np.array(smoothed)


def interpolate_path(path, resolution=0.05):
    if path is None or len(path) < 2:
        return path
    dense = [path[0]]
    for i in range(len(path) - 1):
        seg_len = np.linalg.norm(path[i + 1] - path[i])
        n       = max(2, int(seg_len / resolution))
        for t in np.linspace(0, 1, n)[1:]:
            dense.append(path[i] + t * (path[i + 1] - path[i]))
    return np.array(dense)


# ──────────────────────────────────────────
# 6. Main interface
# ──────────────────────────────────────────

def plan_path(start, goal,
              coord_path=_DEFAULT_COORD,
              color_path=_DEFAULT_COLOR,
              normal_path=_DEFAULT_NORMAL,
              max_iter=MAX_ITER, verbose=True):
    """
    Drone path planning main function

    Parameters
    ----------
    start : (x, y, z)  start position (m)
    goal  : (x, y, z)  goal position (m)

    Returns
    -------
    dict or None
        'raw_path'    : raw RRT* path       (N, 3)
        'smooth_path' : shortcut path       (M, 3)
        'dense_path'  : interpolated path   (K, 3)
        'total_dist'  : total distance (m)
        'n_nodes'     : number of nodes explored
        'actual_start': actual start used (may be adjusted)
        'actual_goal' : actual goal used  (may be adjusted)
    """
    start = np.array(start, dtype=np.float64)
    goal  = np.array(goal,  dtype=np.float64)

    print("=" * 55)
    print("  RRT* Drone Path Planning")
    print("=" * 55)
    print(f"  Start: {start}")
    print(f"  Goal : {goal}")
    print(f"  Obstacle clearance: {OBSTACLE_RADIUS:.3f}m")
    print()

    print("[1/4] Loading point cloud...")
    coord, color, normal = load_pointcloud(coord_path, color_path, normal_path)

    margin = 0.2
    bounds = [
        [coord[:, 0].min() - margin, coord[:, 0].max() + margin],
        [coord[:, 1].min() - margin, coord[:, 1].max() + margin],
        [coord[:, 2].min() + OBSTACLE_RADIUS, coord[:, 2].max() - OBSTACLE_RADIUS],
    ]
    print(f"  Space X: {bounds[0][0]:.2f} ~ {bounds[0][1]:.2f} m")
    print(f"  Space Y: {bounds[1][0]:.2f} ~ {bounds[1][1]:.2f} m")
    print(f"  Space Z: {bounds[2][0]:.2f} ~ {bounds[2][1]:.2f} m")

    # Warn if out of bounds
    for name, pt in [("Start", start), ("Goal", goal)]:
        for axis, (lo, hi) in zip(['X', 'Y', 'Z'], bounds):
            val = pt[['X', 'Y', 'Z'].index(axis)]
            if not (lo <= val <= hi):
                print(f"  WARNING: {name} {axis}={val:.2f} is out of range ({lo:.2f}~{hi:.2f})")

    print("\n[2/4] Building KD-Tree...")
    kd_tree = build_obstacle_tree(coord)

    # Auto-adjust start/goal if inside obstacle
    actual_start = find_nearest_free(start, kd_tree, bounds)
    actual_goal  = find_nearest_free(goal,  kd_tree, bounds)

    if actual_start is None:
        print("  ERROR: Cannot find free space near start position.")
        return None
    if actual_goal is None:
        print("  ERROR: Cannot find free space near goal position.")
        return None

    if not np.allclose(actual_start, start):
        print(f"  Start adjusted: {np.round(start,3)} -> {np.round(actual_start,3)}")
    else:
        print("  Start position is free")

    if not np.allclose(actual_goal, goal):
        print(f"  Goal  adjusted: {np.round(goal,3)} -> {np.round(actual_goal,3)}")
    else:
        print("  Goal  position is free")

    print(f"\n[3/4] Running RRT* (max {max_iter} iterations)...")
    planner  = RRTStar(actual_start, actual_goal, kd_tree, bounds, max_iter=max_iter)
    raw_path = planner.plan(verbose=verbose)

    if raw_path is None:
        print("\n  Path not found.")
        print("  Hint: increase max_iter or check start/goal positions.")
        return None

    print("\n[4/4] Optimizing path...")
    smooth = smooth_path(raw_path, kd_tree)
    dense  = interpolate_path(smooth, resolution=0.05)

    total_dist = sum(
        np.linalg.norm(smooth[i + 1] - smooth[i]) for i in range(len(smooth) - 1)
    )

    print(f"\n{'=' * 55}")
    print(f"  Path planning complete!")
    print(f"  Raw waypoints : {len(raw_path)}")
    print(f"  After smooth  : {len(smooth)}")
    print(f"  After interp  : {len(dense)}")
    print(f"  Total distance: {total_dist:.3f}m")
    print(f"{'=' * 55}")

    print("\nWaypoints (smoothed):")
    for idx, wp in enumerate(smooth):
        print(f"  [{idx:2d}] x={wp[0]:7.3f}, y={wp[1]:7.3f}, z={wp[2]:7.3f}")

    return {
        'raw_path':    raw_path,
        'smooth_path': smooth,
        'dense_path':  dense,
        'total_dist':  total_dist,
        'n_nodes':     len(planner.nodes),
        'actual_start': actual_start,
        'actual_goal':  actual_goal,
    }


# ──────────────────────────────────────────
# Example
# ──────────────────────────────────────────
if __name__ == '__main__':
    # Space range: X[-12.86 ~ -4.38], Y[-1.16 ~ 5.50], Z[0.0 ~ 3.73]
    START = [-12.0, 0.0, 1.5]
    GOAL  = [-5.0,  4.0, 1.5]

    result = plan_path(START, GOAL, max_iter=5000)

    if result:
        save_path = os.path.join(_BASE_DIR, 'waypoints.npy')
        np.save(save_path, result['smooth_path'])
        print(f"\nWaypoints saved: {save_path}")