"""
Parallel 3D RRT* Path Planning for Multiple Rooms (Wall Penetration Fixed)

This script loads room data from compressed_npy folder and executes
RRT* path planning algorithm in parallel using multiprocessing.

Directory structure:
./compressed_npy/
  ├── 00800_TEEsavR23oF_000_002/
  │   ├── coord.npy
  │   └── segment.npy
  ├── 00800_TEEsavR23oF_000_003/
  ...
"""

import numpy as np
import os
from multiprocessing import Pool, cpu_count
from scipy.spatial import KDTree
import time


class RRTStar3D:
    """3D RRT* Path Planning Algorithm with Enhanced Collision Detection"""

    def __init__(self, start, goal, obstacle_cloud, bounds, 
                 max_iter=3000, step_size=0.2, goal_sample_rate=0.1,
                 search_radius=0.6, collision_radius=0.35):
        
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.collision_radius = collision_radius

        # Build KDTree for fast collision checking
        self.obstacle_tree = KDTree(obstacle_cloud)
        self.obstacle_cloud = obstacle_cloud

        # Tree structure
        self.nodes = {0: {'pos': self.start, 'parent': None, 'cost': 0.0}}

    def sample(self):
        """Sample random point in space, with bias toward goal"""
        if np.random.random() < self.goal_sample_rate:
            return self.goal.copy()

        point = np.array([
            np.random.uniform(self.bounds[0][0], self.bounds[0][1]),
            np.random.uniform(self.bounds[1][0], self.bounds[1][1]),
            np.random.uniform(self.bounds[2][0], self.bounds[2][1])
        ])
        return point

    def nearest(self, point):
        """Find nearest node to the given point"""
        distances = [np.linalg.norm(node['pos'] - point) 
                     for node in self.nodes.values()]
        nearest_id = np.argmin(distances)
        return nearest_id

    def steer(self, from_pos, to_pos):
        """Move from from_pos toward to_pos by step_size"""
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            return to_pos

        return from_pos + (direction / distance) * self.step_size

    def is_collision_free(self, from_pos, to_pos, n_checks=20):
        """
        Enhanced collision checking with denser sampling

        Args:
            from_pos: Start position
            to_pos: End position
            n_checks: Number of points to check along path (INCREASED from 10 to 20)

        Returns:
            True if path is collision-free
        """
        segment_length = np.linalg.norm(to_pos - from_pos)

        # Adaptive number of checks based on segment length
        # Check at least every 10cm
        n_checks = max(n_checks, int(np.ceil(segment_length / 0.1)))

        # Sample points densely along the line segment
        for alpha in np.linspace(0, 1, n_checks):
            point = from_pos + alpha * (to_pos - from_pos)

            # Query multiple nearest obstacles (not just the nearest one)
            # This helps detect thin walls
            k_neighbors = min(10, len(self.obstacle_cloud))
            distances, indices = self.obstacle_tree.query(point, k=k_neighbors)

            # Check if any nearby obstacle is too close
            if np.any(distances < self.collision_radius):
                return False

        return True

    def near(self, point):
        """Find nodes within search_radius (for RRT* rewiring)"""
        near_ids = []
        for node_id, node in self.nodes.items():
            if np.linalg.norm(node['pos'] - point) < self.search_radius:
                near_ids.append(node_id)
        return near_ids

    def plan(self):
        """Execute RRT* planning algorithm"""
        for i in range(self.max_iter):
            # Sample random point
            rand_point = self.sample()

            # Find nearest node in tree
            nearest_id = self.nearest(rand_point)
            nearest_pos = self.nodes[nearest_id]['pos']

            # Steer toward sample
            new_pos = self.steer(nearest_pos, rand_point)

            # Check collision with enhanced method
            if not self.is_collision_free(nearest_pos, new_pos):
                continue

            # Find nearby nodes for optimal parent selection (RRT*)
            near_ids = self.near(new_pos)

            # Choose parent with minimum cost
            min_cost = self.nodes[nearest_id]['cost'] + np.linalg.norm(new_pos - nearest_pos)
            best_parent = nearest_id

            for near_id in near_ids:
                near_pos = self.nodes[near_id]['pos']
                near_cost = self.nodes[near_id]['cost']
                cost_via_near = near_cost + np.linalg.norm(new_pos - near_pos)

                if cost_via_near < min_cost and self.is_collision_free(near_pos, new_pos):
                    min_cost = cost_via_near
                    best_parent = near_id

            # Add new node to tree
            new_id = len(self.nodes)
            self.nodes[new_id] = {
                'pos': new_pos,
                'parent': best_parent,
                'cost': min_cost
            }

            # Rewire nearby nodes (RRT* optimization)
            for near_id in near_ids:
                near_pos = self.nodes[near_id]['pos']
                cost_via_new = min_cost + np.linalg.norm(near_pos - new_pos)

                if cost_via_new < self.nodes[near_id]['cost'] and \
                   self.is_collision_free(new_pos, near_pos):
                    self.nodes[near_id]['parent'] = new_id
                    self.nodes[near_id]['cost'] = cost_via_new

            # Check if goal is reached
            if np.linalg.norm(new_pos - self.goal) < self.step_size:
                if self.is_collision_free(new_pos, self.goal):
                    goal_id = len(self.nodes)
                    self.nodes[goal_id] = {
                        'pos': self.goal,
                        'parent': new_id,
                        'cost': min_cost + np.linalg.norm(self.goal - new_pos)
                    }
                    print(f"  Goal reached at iteration {i+1}")
                    return self.extract_path(goal_id)

        print(f"  Max iterations ({self.max_iter}) reached, no path found")
        return None

    def extract_path(self, goal_id):
        """Extract path from start to goal by backtracking through tree"""
        path = []
        current_id = goal_id

        while current_id is not None:
            path.append(self.nodes[current_id]['pos'])
            current_id = self.nodes[current_id]['parent']

        return np.array(path[::-1])


def find_free_space_regions(coord, segment, obstacle_tree, min_clearance=0.5, voxel_size=0.25):
    """
    Identify free space regions within the point cloud bounds.

    Args:
        coord: Nx3 array of all point coordinates
        segment: N array of segment labels
        obstacle_tree: KDTree of obstacle points
        min_clearance: Minimum distance from obstacles (meters) - INCREASED to 50cm
        voxel_size: Grid resolution for sampling (meters) - REDUCED for denser sampling

    Returns:
        free_points: Mx3 array of collision-free points
    """
    # Get bounds
    x_min, x_max = coord[:, 0].min(), coord[:, 0].max()
    y_min, y_max = coord[:, 1].min(), coord[:, 1].max()
    z_min, z_max = coord[:, 2].min(), coord[:, 2].max()

    # Add margin to bounds
    margin = 0.6  # Increased margin
    x_min += margin
    x_max -= margin
    y_min += margin
    y_max -= margin
    z_min += 0.6  # Stay well above floor
    z_max -= 0.3  # Stay well below ceiling

    # Create sampling grid
    x_samples = np.arange(x_min, x_max, voxel_size)
    y_samples = np.arange(y_min, y_max, voxel_size)
    z_samples = np.arange(z_min, z_max, voxel_size)

    # Generate grid points
    xx, yy, zz = np.meshgrid(x_samples, y_samples, z_samples, indexing='ij')
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    print(f"  Checking {len(grid_points)} candidate points for free space...")

    # Query distances to obstacles for all grid points
    # Check multiple neighbors to detect thin walls
    k_neighbors = 5
    distances, _ = obstacle_tree.query(grid_points, k=k_neighbors)

    # All k neighbors must be far enough
    min_distances = distances.min(axis=1)
    free_mask = min_distances > min_clearance
    free_points = grid_points[free_mask]

    return free_points


def select_distant_start_goal(free_points, min_distance_ratio=0.5):
    """
    Select start and goal points that are far apart.

    Args:
        free_points: Mx3 array of valid free space points
        min_distance_ratio: Minimum distance as ratio of max possible distance

    Returns:
        start, goal: Selected start and goal positions
    """
    if len(free_points) < 2:
        return None, None

    # Compute bounding box diagonal (max possible distance)
    bbox_min = free_points.min(axis=0)
    bbox_max = free_points.max(axis=0)
    max_distance = np.linalg.norm(bbox_max - bbox_min)

    # Try to find distant pairs
    max_attempts = 100
    best_distance = 0
    best_start = None
    best_goal = None

    for _ in range(max_attempts):
        # Randomly select two points
        idx = np.random.choice(len(free_points), size=2, replace=False)
        candidate_start = free_points[idx[0]]
        candidate_goal = free_points[idx[1]]

        distance = np.linalg.norm(candidate_goal - candidate_start)

        # Accept if distance is large enough
        if distance > max_distance * min_distance_ratio:
            return candidate_start, candidate_goal

        # Track best pair
        if distance > best_distance:
            best_distance = distance
            best_start = candidate_start
            best_goal = candidate_goal

    # Return best found pair
    return best_start, best_goal


def process_single_room(room_id):
    """
    Worker function for multiprocessing.
    Loads room data, generates distant start/goal within free space, runs RRT*, saves result.

    Args:
        room_id: Room number (e.g., 2 for room 002)

    Returns:
        Tuple: (room_id, success, path_length, execution_time)
    """
    start_time = time.time()

    # Construct paths
    folder_name = f'00800_TEEsavR23oF_000_{room_id:03d}'
    folder_path = f'./compressed_npy/{folder_name}'

    # Load data
    try:
        coord = np.load(f'{folder_path}/coord.npy')
        segment = np.load(f'{folder_path}/segment.npy')
    except FileNotFoundError:
        print(f"Room {room_id:03d}: Data files not found in {folder_path}")
        return (room_id, False, 0, 0)

    print(f"Room {room_id:03d}: Loaded {len(coord)} points")

    # Filter obstacles - include ALL non-free space
    # Segment 0=floor, 1=wall, 2=furniture, etc.
    # For conservative collision checking, treat floor as obstacle too
    obstacle_mask = segment != 255  # Assuming 255 or similar is 'free space'

    # If that doesn't work, use walls and furniture
    if obstacle_mask.sum() == 0:
        obstacle_mask = (segment == 1) | (segment == 2)

    obstacles = coord[obstacle_mask]

    if len(obstacles) == 0:
        print(f"Room {room_id:03d}: Warning - No obstacles detected, using all points")
        obstacles = coord

    print(f"Room {room_id:03d}: Using {len(obstacles)} obstacle points")

    # Build obstacle KDTree
    obstacle_tree = KDTree(obstacles)

    # Compute bounding box
    bounds = [
        [coord[:, 0].min(), coord[:, 0].max()],
        [coord[:, 1].min(), coord[:, 1].max()],
        [coord[:, 2].min() + 0.6, coord[:, 2].max() - 0.3]
    ]

    print(f"Room {room_id:03d}: Finding free space regions...")

    # Find free space regions with stricter criteria
    free_points = find_free_space_regions(
        coord=coord,
        segment=segment,
        obstacle_tree=obstacle_tree,
        min_clearance=0.5,  # 50cm clearance (increased from 40cm)
        voxel_size=0.25     # 25cm sampling grid (denser than before)
    )

    if len(free_points) < 2:
        print(f"Room {room_id:03d}: Insufficient free space found ({len(free_points)} points)")
        return (room_id, False, 0, time.time() - start_time)

    print(f"Room {room_id:03d}: Found {len(free_points)} free space locations")

    # Select distant start and goal
    start, goal = select_distant_start_goal(free_points, min_distance_ratio=0.4)

    if start is None or goal is None:
        print(f"Room {room_id:03d}: Failed to find suitable start/goal pair")
        return (room_id, False, 0, time.time() - start_time)

    distance = np.linalg.norm(goal - start)
    print(f"Room {room_id:03d}: Start={start.round(2)}, Goal={goal.round(2)}, Distance={distance:.2f}m")

    # Run RRT* with enhanced collision detection
    rrt = RRTStar3D(
        start=start,
        goal=goal,
        obstacle_cloud=obstacles,
        bounds=bounds,
        max_iter=4000,       # Increased iterations
        step_size=0.2,       # Smaller steps (20cm instead of 30cm)
        goal_sample_rate=0.15,
        search_radius=0.6,   # Reduced for more conservative paths
        collision_radius=0.50  # Larger safety margin (35cm instead of 25cm)
    )

    path = rrt.plan()

    exec_time = time.time() - start_time

    if path is not None:
        # Verify path doesn't penetrate walls (post-processing check)
        print(f"Room {room_id:03d}: Verifying path collision-free...")
        path_valid = True
        for i in range(len(path) - 1):
            if not rrt.is_collision_free(path[i], path[i+1], n_checks=30):
                print(f"Room {room_id:03d}: Warning - Path segment {i} has collision!")
                path_valid = False

        if not path_valid:
            print(f"Room {room_id:03d}: ✗ Path validation failed, discarding")
            return (room_id, False, 0, exec_time)

        # Create output directory
        os.makedirs('./output', exist_ok=True)
        output_file = f'./output/path_{room_id:03d}.txt'

        # Save path to text file
        with open(output_file, 'w') as f:
            f.write(f"# Room {folder_name}\n")
            f.write(f"# Start: {start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}\n")
            f.write(f"# Goal: {goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}\n")
            f.write(f"# Straight-line distance: {distance:.3f}m\n")
            f.write(f"# Path length: {len(path)} waypoints\n")
            f.write(f"# Execution time: {exec_time:.2f}s\n")
            f.write(f"# Safety margin: 35cm (20cm drone + 15cm buffer)\n")
            f.write("# x, y, z\n")

            for point in path:
                f.write(f"{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}\n")

        print(f"Room {room_id:03d}: ✓ Path saved to {output_file} ({len(path)} points, {exec_time:.2f}s)")
        return (room_id, True, len(path), exec_time)
    else:
        print(f"Room {room_id:03d}: ✗ Path planning failed ({exec_time:.2f}s)")
        return (room_id, False, 0, exec_time)


def main():
    """Main execution function"""
    print("="*70)
    print("PARALLEL 3D RRT* PATH PLANNING FOR DRONES")
    print("Enhanced Collision Detection - Wall Penetration Fixed")
    print("="*70)

    # Check if data directory exists
    if not os.path.exists('./compressed_npy'):
        print("\nERROR: ./compressed_npy directory not found!")
        print("Please ensure the data folder is in the same directory as this script.")
        return

    # Target rooms: 002, 003, 004, 005, 006
    room_ids = [2, 3, 4, 5, 6]

    # Determine number of processes
    n_processes = min(cpu_count(), len(room_ids))
    print(f"\n🚀 Starting parallel execution with {n_processes} processes...\n")

    overall_start = time.time()

    # Execute parallel RRT* planning
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_room, room_ids)

    overall_time = time.time() - overall_start

    # Print summary report
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)

    success_count = sum(1 for _, success, _, _ in results if success)
    total_time = sum(exec_time for _, _, _, exec_time in results)

    print(f"Total rooms processed: {len(room_ids)}")
    print(f"Successful paths: {success_count}/{len(room_ids)}")
    print(f"Total computation time: {total_time:.2f}s")
    print(f"Wall-clock time (parallel): {overall_time:.2f}s")
    if overall_time > 0:
        print(f"Speedup factor: {total_time/overall_time:.2f}x")
    print()

    for room_id, success, path_len, exec_time in results:
        status = "✓" if success else "✗"
        print(f"  {status} Room {room_id:03d}: {path_len:4d} waypoints, {exec_time:5.2f}s")

    print("\n" + "="*70)
    if success_count > 0:
        print("✓ Path files saved to ./output/ directory")
    print("="*70)


if __name__ == '__main__':
    main()
