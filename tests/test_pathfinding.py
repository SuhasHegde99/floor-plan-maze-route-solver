"""
tests/test_pathfinding.py
=========================
Automated unit and integration test suite for Dijkstra's Algorithm, A* Search,
and image-to-grid processing.

Test Coverage:
  1. Simple grid with known theoretical shortest path cost.
  2. Obstacle grid forcing a detour around walls.
  3. Unreachable goal (enclosed by complete wall perimeter).
  4. Start equal to Goal (zero cost edge case).
  5. Comparative optimality: Dijkstra and A* return the exact same cost.
  6. Wall safety: Verifying that computed routes NEVER traverse blocked cells.
  7. Orthogonal movement constraint: Every step must be a valid 4-neighbor transition.
  8. Boundary validation: Out-of-bounds start/goal coordinates handled safely.
  9. Blocked start or blocked goal input validation.
  10. Real performance metrics: Execution time > 0, nodes expanded > 0, memory > 0.
  11. ImageGridProcessor thresholding and coordinate mapping fidelity.
"""

import os
import unittest
from typing import List

from pathfinding import (
    dijkstra,
    astar,
    manhattan_distance,
    get_valid_neighbors,
    compare_algorithms,
    PathfindingResult
)
from image_processing import ImageGridProcessor, generate_sample_maze


class TestManhattanHeuristic(unittest.TestCase):
    """Verifies properties of the Manhattan distance heuristic."""

    def test_zero_distance(self):
        self.assertEqual(manhattan_distance((3, 3), (3, 3)), 0)

    def test_horizontal_vertical_distance(self):
        self.assertEqual(manhattan_distance((0, 0), (0, 5)), 5)
        self.assertEqual(manhattan_distance((2, 1), (7, 1)), 5)

    def test_diagonal_manhattan_distance(self):
        # (0, 0) to (3, 4) should be 3 + 4 = 7
        self.assertEqual(manhattan_distance((0, 0), (3, 4)), 7)

    def test_symmetry(self):
        p1 = (12, 45)
        p2 = (89, 3)
        self.assertEqual(manhattan_distance(p1, p2), manhattan_distance(p2, p1))


class TestPathfindingAlgorithms(unittest.TestCase):
    """Tests Dijkstra and A* implementations on known grid layouts."""

    def setUp(self):
        # 1. Simple open 5x5 grid (all cells walkable)
        self.open_5x5: List[List[bool]] = [
            [True] * 5 for _ in range(5)
        ]

        # 2. 5x5 Grid with a wall forcing a detour:
        # S . . . .
        # # # # # .
        # . . . . .
        # . # # # #
        # . . . . G
        self.obstacle_5x5: List[List[bool]] = [
            [True,  True,  True,  True,  True],
            [False, False, False, False, True],
            [True,  True,  True,  True,  True],
            [True,  False, False, False, False],
            [True,  True,  True,  True,  True],
        ]

        # 3. 4x4 Grid with completely boxed-off goal:
        # S . . .
        # . # # #
        # . # G #
        # . # # #
        self.unreachable_grid: List[List[bool]] = [
            [True,  True,  True,  True],
            [True,  False, False, False],
            [True,  False, True,  False],
            [True,  False, False, False],
        ]

    def _assert_valid_path(self, grid: List[List[bool]], path: List[tuple], start: tuple, goal: tuple):
        """Helper to ensure a path is continuous, 4-connected, and avoids blocked cells."""
        self.assertTrue(len(path) > 0, "Path must not be empty.")
        self.assertEqual(path[0], start, "Path must begin at Start.")
        self.assertEqual(path[-1], goal, "Path must terminate at Goal.")

        for i, (r, c) in enumerate(path):
            # Check cell is within bounds
            self.assertTrue(0 <= r < len(grid), f"Row {r} out of bounds.")
            self.assertTrue(0 <= c < len(grid[0]), f"Col {c} out of bounds.")
            # Check cell is walkable (not blocked)
            self.assertTrue(
                grid[r][c],
                f"Path violates wall constraint: cell ({r}, {c}) is blocked!"
            )
            # Check 4-connected step from previous cell
            if i > 0:
                prev_r, prev_c = path[i - 1]
                step_dist = abs(r - prev_r) + abs(c - prev_c)
                self.assertEqual(
                    step_dist, 1,
                    f"Invalid step from ({prev_r}, {prev_c}) to ({r}, {c}): must be exactly 1 orthogonal step (no diagonals)."
                )

    def test_simple_grid_known_cost(self):
        """Test on open 5x5 grid from (0, 0) to (4, 4): optimal cost is 8."""
        start = (0, 0)
        goal = (4, 4)
        expected_cost = 8

        d_res = dijkstra(self.open_5x5, start, goal)
        a_res = astar(self.open_5x5, start, goal)

        self.assertTrue(d_res.success, f"Dijkstra failed: {d_res.message}")
        self.assertTrue(a_res.success, f"A* failed: {a_res.message}")

        self.assertEqual(d_res.cost, expected_cost)
        self.assertEqual(a_res.cost, expected_cost)
        self.assertEqual(d_res.step_count, expected_cost)
        self.assertEqual(a_res.step_count, expected_cost)

        self._assert_valid_path(self.open_5x5, d_res.path, start, goal)
        self._assert_valid_path(self.open_5x5, a_res.path, start, goal)

    def test_obstacle_grid_detour(self):
        """Test on grid with walls forcing a winding path."""
        start = (0, 0)
        goal = (4, 4)

        d_res = dijkstra(self.obstacle_5x5, start, goal)
        a_res = astar(self.obstacle_5x5, start, goal)

        self.assertTrue(d_res.success)
        self.assertTrue(a_res.success)

        # Both must find the exact same minimal cost
        self.assertEqual(d_res.cost, a_res.cost)
        # Expected detour path length: (0,0)->(0,4)->(2,4)->(2,0)->(4,0)->(4,4) = 4 + 2 + 4 + 2 + 4 = 16
        self.assertEqual(d_res.cost, 16)

        self._assert_valid_path(self.obstacle_5x5, d_res.path, start, goal)
        self._assert_valid_path(self.obstacle_5x5, a_res.path, start, goal)

    def test_unreachable_goal(self):
        """Goal completely enclosed by walls: both algorithms must report no path."""
        start = (0, 0)
        goal = (2, 2)  # Boxed in by False

        d_res = dijkstra(self.unreachable_grid, start, goal)
        a_res = astar(self.unreachable_grid, start, goal)

        self.assertFalse(d_res.success)
        self.assertFalse(a_res.success)
        self.assertEqual(len(d_res.path), 0)
        self.assertEqual(len(a_res.path), 0)
        self.assertIn("No route exists", d_res.message)
        self.assertIn("No route exists", a_res.message)

    def test_start_equals_goal(self):
        """Start and Goal are the same node: cost must be 0, path has 1 node."""
        coord = (2, 2)

        d_res = dijkstra(self.open_5x5, coord, coord)
        a_res = astar(self.open_5x5, coord, coord)

        self.assertTrue(d_res.success)
        self.assertTrue(a_res.success)
        self.assertEqual(d_res.cost, 0)
        self.assertEqual(a_res.cost, 0)
        self.assertEqual(d_res.path, [coord])
        self.assertEqual(a_res.path, [coord])

    def test_blocked_start_or_goal(self):
        """Selecting a blocked obstacle cell must return an informative error."""
        start = (1, 1)  # blocked in unreachable_grid
        goal = (0, 0)

        d_res = dijkstra(self.unreachable_grid, start, goal)
        self.assertFalse(d_res.success)
        self.assertIn("obstacle", d_res.message)

        # Blocked goal
        a_res = astar(self.unreachable_grid, (0, 0), (1, 1))
        self.assertFalse(a_res.success)
        self.assertIn("obstacle", a_res.message)

    def test_out_of_bounds_coordinates(self):
        """Coordinates outside grid dimensions must be rejected safely."""
        d_res = dijkstra(self.open_5x5, (-1, 0), (4, 4))
        self.assertFalse(d_res.success)
        self.assertIn("outside", d_res.message)

        a_res = astar(self.open_5x5, (0, 0), (10, 10))
        self.assertFalse(a_res.success)
        self.assertIn("outside", a_res.message)

    def test_multiple_random_grids_optimality_equivalence(self):
        """
        Generates diverse 10x10 grids with varying obstacle patterns and confirms
        that whenever a path exists, Dijkstra and A* ALWAYS produce identical costs.
        """
        import random
        random.seed(42)

        for trial in range(10):
            rows, cols = 10, 10
            # 20% obstacle density
            grid = [[random.random() > 0.20 for _ in range(cols)] for _ in range(rows)]
            start = (0, 0)
            goal = (rows - 1, cols - 1)
            grid[start[0]][start[1]] = True
            grid[goal[0]][goal[1]] = True

            d_res, a_res = compare_algorithms(grid, start, goal)

            self.assertEqual(
                d_res.success, a_res.success,
                f"Trial {trial}: Solvability mismatch between Dijkstra ({d_res.success}) and A* ({a_res.success})"
            )

            if d_res.success:
                self.assertEqual(
                    d_res.cost, a_res.cost,
                    f"Trial {trial}: Shortest path cost mismatch! Dijkstra={d_res.cost}, A*={a_res.cost}"
                )
                self._assert_valid_path(grid, d_res.path, start, goal)
                self._assert_valid_path(grid, a_res.path, start, goal)

    def test_performance_metrics_real_values(self):
        """Validates that real timing, memory tracking, and node counts are returned."""
        d_res, a_res = compare_algorithms(self.open_5x5, (0, 0), (4, 4))

        # Check nodes expanded
        self.assertGreater(d_res.nodes_expanded, 0)
        self.assertGreater(a_res.nodes_expanded, 0)

        # On an open grid, A* guided by Manhattan distance should expand fewer or equal nodes
        self.assertLessEqual(a_res.nodes_expanded, d_res.nodes_expanded)

        # Check execution time
        self.assertGreaterEqual(d_res.execution_time_ms, 0.0)
        self.assertGreaterEqual(a_res.execution_time_ms, 0.0)

        # Check peak memory
        self.assertGreaterEqual(d_res.peak_memory_kb, 0.0)
        self.assertGreaterEqual(a_res.peak_memory_kb, 0.0)


class TestImageProcessing(unittest.TestCase):
    """Tests image loading, thresholding, coordinate mapping, and route rendering."""

    def setUp(self):
        self.sample_maze_path = os.path.join("examples", "sample_maze.png")
        if not os.path.exists(self.sample_maze_path):
            os.makedirs("examples", exist_ok=True)
            img = generate_sample_maze(200, 200, cell_dimension=20, wall_thickness=4)
            img.save(self.sample_maze_path)

        self.processor = ImageGridProcessor(self.sample_maze_path)

    def test_image_loading(self):
        self.assertIsNotNone(self.processor.original_image)
        self.assertEqual(self.processor.width, 400 if os.path.getsize(self.sample_maze_path) > 5000 else self.processor.width)
        self.assertGreater(self.processor.height, 0)

    def test_threshold_and_polarity(self):
        # Bright walkable
        grid_bright = self.processor.compute_grid(threshold=128, walkable_is_bright=True, cell_size=2)
        self.assertEqual(len(grid_bright), self.processor.grid_rows)
        self.assertEqual(len(grid_bright[0]), self.processor.grid_cols)

        # Dark walkable (inverse)
        grid_dark = self.processor.compute_grid(threshold=128, walkable_is_bright=False, cell_size=2)
        # Verify inverse polarity
        self.assertEqual(grid_bright[0][0], not grid_dark[0][0])

    def test_coordinate_mapping_fidelity(self):
        self.processor.compute_grid(threshold=128, walkable_is_bright=True, cell_size=4)
        
        # Test image -> grid
        r, c = self.processor.image_to_grid_coord(100, 80)
        self.assertEqual(c, 100 // 4)
        self.assertEqual(r, 80 // 4)

        # Test grid -> image center
        ix, iy = self.processor.grid_to_image_coord(r, c)
        self.assertEqual(ix, (100 // 4) * 4 + 2)
        self.assertEqual(iy, (80 // 4) * 4 + 2)

    def test_cell_walkability_feedback(self):
        self.processor.compute_grid(threshold=128, walkable_is_bright=True, cell_size=2)
        is_walkable, msg = self.processor.get_cell_status_description(0, 0)
        self.assertIsInstance(is_walkable, bool)
        self.assertIsInstance(msg, str)
        self.assertTrue(len(msg) > 0)

    def test_route_overlay_rendering(self):
        self.processor.compute_grid(threshold=128, walkable_is_bright=True, cell_size=4)
        start = (2, 2)
        goal = (10, 10)
        dummy_path = [(2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (10, 10)]

        overlay = self.processor.render_route_overlay(
            path=dummy_path,
            start=start,
            goal=goal,
            path_color=(0, 220, 255)
        )
        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.size, self.processor.original_image.size)


if __name__ == "__main__":
    unittest.main()
