"""
tests/test_gui.py
=================
Automated verification of the Tkinter desktop GUI workflow and usability improvements:
  - Default brightness threshold is 100.
  - Default walkability mode is Bright / Light areas.
  - Advanced Settings section toggle and dark mode polarity selection.
  - Threshold separation and pixel count impact.
  - Recomputation clears blocked points and requires valid points before solving.
  - Execution-time comparison formatting (faster vs slower calculation).
  - Solved route rendering and export functionality.
"""

import os
import sys
import unittest
from unittest.mock import patch
import tkinter as tk
from tkinter import messagebox

from app import FloorPlanApp
from pathfinding import PathfindingResult


class TestGUIWorkflow(unittest.TestCase):
    def setUp(self):
        # Patch messagebox dialogs so automated tests never block on modal windows
        self.patch_warning = patch.object(messagebox, "showwarning", return_value=None)
        self.patch_info = patch.object(messagebox, "showinfo", return_value=None)
        self.patch_error = patch.object(messagebox, "showerror", return_value=None)
        self.mock_warning = self.patch_warning.start()
        self.mock_info = self.patch_info.start()
        self.mock_error = self.patch_error.start()

        self.root = tk.Tk()
        # Withdraw the window so it does not pop up during automated testing,
        # while Tkinter fully initializes all widgets, canvas, styles, and events.
        self.root.withdraw()
        self.app = FloorPlanApp(self.root)

    def tearDown(self):
        self.patch_warning.stop()
        self.patch_info.stop()
        self.patch_error.stop()
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_gui_defaults(self):
        """Verifies default brightness threshold is 100 and default polarity is bright."""
        self.assertEqual(int(float(self.app.slider_threshold.get())), 100)
        self.assertEqual(self.app.thresh_val_label.cget("text"), "100")
        self.assertEqual(self.app.var_polarity.get(), "bright")
        self.assertIn("Bright", self.app.lbl_active_mode.cget("text"))

    def test_advanced_settings_toggle_and_polarity(self):
        """Verifies Advanced Settings toggles open/closed and polarity changes update the grid."""
        # Starts collapsed
        self.assertFalse(self.app.advanced_visible)
        
        # Toggle open
        self.app._toggle_advanced_settings()
        self.assertTrue(self.app.advanced_visible)
        self.assertIn("▲", self.app.btn_toggle_adv.cget("text"))

        # Switch to dark polarity
        self.app.var_polarity.set("dark")
        self.app._on_polarity_changed()
        self.assertIn("Dark", self.app.lbl_active_mode.cget("text"))
        self.assertFalse(self.app.processor.walkable_is_bright)

        # Toggle close
        self.app._toggle_advanced_settings()
        self.assertFalse(self.app.advanced_visible)

        # Switch back to bright
        self.app.var_polarity.set("bright")
        self.app._on_polarity_changed()
        self.assertTrue(self.app.processor.walkable_is_bright)

    def test_threshold_impact_on_walkable_pixels(self):
        """Verifies higher threshold produces fewer or equal walkable cells in bright mode."""
        # Compute with threshold 50
        grid_50 = self.app.processor.compute_grid(threshold=50, walkable_is_bright=True, cell_size=2)
        count_50 = sum(sum(1 for cell in row if cell) for row in grid_50)

        # Compute with threshold 200
        grid_200 = self.app.processor.compute_grid(threshold=200, walkable_is_bright=True, cell_size=2)
        count_200 = sum(sum(1 for cell in row if cell) for row in grid_200)

        self.assertGreaterEqual(count_50, count_200, "Lower threshold must classify more or equal pixels as walkable.")

    def test_solve_requires_valid_points(self):
        """Verifies run_solvers refuses to run without valid, walkable Start and Goal points."""
        # 1. Missing points
        self.app.start_coord = None
        self.app.goal_coord = None
        self.app.run_solvers()
        self.assertIsNone(self.app.dijkstra_result)

        # 2. Blocked point
        self.app.start_coord = (0, 0)  # Wall corner in sample maze
        self.app.goal_coord = (10, 10)
        if not self.app.processor.is_cell_walkable(0, 0):
            self.app.run_solvers()
            self.assertIsNone(self.app.dijkstra_result)
            self.assertIsNone(self.app.start_coord)

    def test_execution_time_comparison_bug_fix(self):
        """
        Verifies execution-time comparison formatting:
        If Dijkstra takes 36.07 ms and A* takes 60.21 ms, comparison must say 'A* 24.14 ms slower', NOT faster.
        """
        # Case 1: A* is slower
        d_slow = PathfindingResult(
            algorithm="Dijkstra",
            success=True,
            path=[(0, 0), (0, 1)],
            cost=1,
            nodes_expanded=50,
            execution_time_ms=36.07,
            peak_memory_kb=100.0
        )
        a_slow = PathfindingResult(
            algorithm="A* Search",
            success=True,
            path=[(0, 0), (0, 1)],
            cost=1,
            nodes_expanded=40,
            execution_time_ms=60.21,
            peak_memory_kb=110.0
        )

        self.app._populate_results_table(d_slow, a_slow)
        items = self.app.tree_results.get_children()
        # Find time row
        time_row_vals = None
        for item in items:
            vals = self.app.tree_results.item(item, "values")
            if vals and vals[0] == "Time (ms)":
                time_row_vals = vals
                break

        self.assertIsNotNone(time_row_vals)
        self.assertEqual(time_row_vals[1], "36.07")
        self.assertEqual(time_row_vals[2], "60.21")
        self.assertEqual(time_row_vals[3], "A* 24.14 ms slower")
        self.assertNotIn("faster", time_row_vals[3])

        # Case 2: A* is faster
        d_fast = PathfindingResult(
            algorithm="Dijkstra",
            success=True,
            path=[(0, 0), (0, 1)],
            cost=1,
            nodes_expanded=100,
            execution_time_ms=75.50,
            peak_memory_kb=100.0
        )
        a_fast = PathfindingResult(
            algorithm="A* Search",
            success=True,
            path=[(0, 0), (0, 1)],
            cost=1,
            nodes_expanded=30,
            execution_time_ms=45.25,
            peak_memory_kb=95.0
        )

        self.app._populate_results_table(d_fast, a_fast)
        items = self.app.tree_results.get_children()
        for item in items:
            vals = self.app.tree_results.item(item, "values")
            if vals and vals[0] == "Time (ms)":
                time_row_vals = vals
                break

        self.assertIsNotNone(time_row_vals)
        self.assertEqual(time_row_vals[1], "75.50")
        self.assertEqual(time_row_vals[2], "45.25")
        self.assertEqual(time_row_vals[3], "A* 30.25 ms faster")

    def test_gui_solve_workflow(self):
        """Verifies end-to-end solving on sample maze with default threshold 100."""
        self.app._set_sample_default_points()
        self.app.run_solvers()

        self.assertIsNotNone(self.app.dijkstra_result)
        self.assertIsNotNone(self.app.astar_result)
        self.assertTrue(self.app.dijkstra_result.success)
        self.assertTrue(self.app.astar_result.success)
        self.assertEqual(self.app.dijkstra_result.cost, self.app.astar_result.cost)

    def test_gui_save_export(self):
        """Verifies exporting the solved route image."""
        self.app._set_sample_default_points()
        self.app.run_solvers()
        export_path = os.path.join("examples", "test_solved_export.png")
        if os.path.exists(export_path):
            os.remove(export_path)

        overlay = self.app._render_current_overlay()
        overlay.save(export_path)

        self.assertTrue(os.path.exists(export_path))
        self.assertGreater(os.path.getsize(export_path), 1000)

        if os.path.exists(export_path):
            os.remove(export_path)


if __name__ == "__main__":
    unittest.main()
