"""
app.py
======
Main Desktop GUI for the Floor Plan & Maze Route Solver.
Algorithms and Data Structures (ADSA) Project.

Features:
  - Interactive Tkinter desktop application.
  - Image upload (PNG, JPG, JPEG) and 1-click Sample Maze loader.
  - Grayscale conversion, adjustable brightness threshold, and bright/dark walkable polarity.
  - 4-directional grid graph generation with customizable downsampling for large plans.
  - Interactive click selection of Start and Goal points with wall collision detection.
  - Independent Dijkstra and A* pathfinding with Manhattan distance heuristic.
  - Real performance benchmark comparison (path cost, nodes expanded, execution time, peak memory).
  - High-visibility visual route rendering with export option.
"""

from __future__ import annotations
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple, List

from PIL import Image, ImageTk

from pathfinding import dijkstra, astar, PathfindingResult
from image_processing import ImageGridProcessor, generate_sample_maze

Coordinate = Tuple[int, int]  # (row, col)


class ToolTip:
    """Displays a clean hover tooltip for a given Tkinter widget."""
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window: Optional[tk.Toplevel] = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
            self.tip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                tw, text=self.text, justify=tk.LEFT,
                background="#0f172a", foreground="#f8fafc",
                relief=tk.SOLID, borderwidth=1,
                font=("Segoe UI", 8), padx=8, pady=5
            )
            label.pack()
        except Exception:
            pass

    def hide_tip(self, event=None):
        if self.tip_window:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None


class FloorPlanApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Floor Plan & Maze Route Solver — ADSA Course Project")
        self.root.geometry("1280x820")
        self.root.minsize(1024, 700)

        # Apply system styling
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Configure custom modern colors
        self.style.configure(".", font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        self.style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), foreground="#1a365d")
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        self.style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

        # Core State
        self.processor: ImageGridProcessor = ImageGridProcessor()
        self.current_pil_image: Optional[Image.Image] = None
        self.display_photo_image: Optional[ImageTk.PhotoImage] = None

        # Display zoom and offset tracking
        self.display_scale: float = 1.0
        self.display_offset_x: int = 0
        self.display_offset_y: int = 0

        # Points
        self.start_coord: Optional[Coordinate] = None  # (row, col) in grid
        self.goal_coord: Optional[Coordinate] = None   # (row, col) in grid
        self.next_selection_mode: str = "start"        # "start", "goal", or "done"

        # Search Results
        self.dijkstra_result: Optional[PathfindingResult] = None
        self.astar_result: Optional[PathfindingResult] = None
        self.active_route_view: str = "both"  # "dijkstra", "astar", "both"

        # Build UI layout
        self._build_menu()
        self._build_ui()

        # Load default sample maze on launch
        self._load_default_sample()

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image...", command=self.upload_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Load Sample Maze", command=self.load_sample_maze)
        file_menu.add_command(label="Save Solved Image...", command=self.save_solved_image, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Algorithm Explanation", command=self.show_algorithm_info)
        help_menu.add_command(label="Image Classification Limitations", command=self.show_limitations_info)
        help_menu.add_command(label="About", command=self.show_about_info)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self.upload_image())
        self.root.bind("<Control-s>", lambda e: self.save_solved_image())

    def _build_ui(self):
        # Main layout: Split into Left Control/Results Panel and Right Image Viewport
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # =========================================================================
        # LEFT SIDE: Controls & Performance Comparison
        # =========================================================================
        left_frame = ttk.Frame(main_paned, width=420)
        main_paned.add(left_frame, weight=0)

        # Top Header Banner
        header_frame = ttk.Frame(left_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=(0, 8))
        ttk.Label(header_frame, text="Floor Plan & Maze Route Solver", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(header_frame, text="Shortest-Path Navigation (Dijkstra vs A* Search)", font=("Segoe UI", 9, "italic")).pack(anchor=tk.W)

        # 1. Action & File Section
        action_group = ttk.LabelFrame(left_frame, text=" 1. Image Source ", padding=8)
        action_group.pack(fill=tk.X, padx=5, pady=4)

        btn_row = ttk.Frame(action_group)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="📁 Upload Image", command=self.upload_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_row, text="🎲 Load Sample Maze", command=self.load_sample_maze).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.lbl_image_info = ttk.Label(action_group, text="Image: None", foreground="#4a5568")
        self.lbl_image_info.pack(anchor=tk.W, pady=(4, 0))

        # 2. Image Processing & Grid Parameters Section
        prep_group = ttk.LabelFrame(left_frame, text=" 2. Grid & Walkability Settings ", padding=8)
        prep_group.pack(fill=tk.X, padx=5, pady=4)

        # Threshold Slider Header
        thresh_top = ttk.Frame(prep_group)
        thresh_top.pack(fill=tk.X)
        ttk.Label(thresh_top, text="Brightness Threshold:").pack(side=tk.LEFT)
        self.thresh_val_label = ttk.Label(thresh_top, text="100", font=("Segoe UI", 9, "bold"))
        self.thresh_val_label.pack(side=tk.RIGHT)

        self.slider_threshold = ttk.Scale(
            prep_group, from_=0, to=255, value=100, command=self._on_threshold_slider_change
        )
        self.slider_threshold.pack(fill=tk.X, pady=(2, 4))
        self.slider_threshold.bind("<ButtonRelease-1>", lambda e: self._apply_grid_settings())

        # Explanatory Help Text for Brightness Threshold
        thresh_help_text = (
            "💡 The brightness threshold separates walkable pixels from blocked pixels (0–255).\n"
            "• In Bright mode (default): Higher thresholds classify fewer pixels as walkable\n"
            "  (stricter/narrower corridors); lower thresholds classify more pixels as walkable."
        )
        lbl_thresh_help = ttk.Label(
            prep_group,
            text=thresh_help_text,
            font=("Segoe UI", 8),
            foreground="#334155",
            justify=tk.LEFT
        )
        lbl_thresh_help.pack(anchor=tk.W, pady=(0, 6))

        # Tooltips for Threshold
        ToolTip(
            self.slider_threshold,
            "Brightness Threshold (0–255):\n"
            "Separates walkable pixels from blocked obstacles.\n"
            "• Higher threshold = fewer walkable pixels (stricter)\n"
            "• Lower threshold = more walkable pixels (more permissive)"
        )
        ToolTip(self.thresh_val_label, "Current brightness cutoff threshold (default: 100).")

        # Active Walkable Mode Status Badge
        self.lbl_active_mode = ttk.Label(
            prep_group,
            text="Walkable Mode: Bright / Light areas (Default)",
            font=("Segoe UI", 8, "italic"),
            foreground="#047857"
        )
        self.lbl_active_mode.pack(anchor=tk.W, pady=(0, 4))

        # Advanced Settings Section (Expandable/Collapsible)
        self.advanced_visible = False
        self.btn_toggle_adv = ttk.Button(
            prep_group,
            text="⚙️ Advanced Settings (Polarity & Grid) ▼",
            command=self._toggle_advanced_settings
        )
        self.btn_toggle_adv.pack(fill=tk.X, pady=(2, 4))

        # Collapsible Frame for Advanced Settings
        self.frame_advanced = ttk.LabelFrame(prep_group, text=" Advanced Settings ", padding=6)
        # Note: frame_advanced is not packed initially; toggled via self._toggle_advanced_settings()

        # Walkability Polarity Radio buttons inside Advanced Settings
        ttk.Label(
            self.frame_advanced,
            text="Walkable Area Detection (Polarity):",
            font=("Segoe UI", 8, "bold")
        ).pack(anchor=tk.W, pady=(2, 2))

        self.var_polarity = tk.StringVar(value="bright")
        rb_bright = ttk.Radiobutton(
            self.frame_advanced,
            text="Bright / Light areas walkable (Default - Mazes & Floor Plans)",
            variable=self.var_polarity,
            value="bright",
            command=self._on_polarity_changed
        )
        rb_bright.pack(anchor=tk.W, pady=1)

        rb_dark = ttk.Radiobutton(
            self.frame_advanced,
            text="Dark / Black areas walkable (Inverted / Line Mazes)",
            variable=self.var_polarity,
            value="dark",
            command=self._on_polarity_changed
        )
        rb_dark.pack(anchor=tk.W, pady=1)

        # Grid Resolution (Cell Size Downsampling) inside Advanced Settings
        res_row = ttk.Frame(self.frame_advanced)
        res_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(res_row, text="Grid Resolution:").pack(side=tk.LEFT)
        self.var_cell_size = tk.StringVar(value="2")
        cell_combo = ttk.Combobox(
            res_row,
            textvariable=self.var_cell_size,
            values=["1 (Pixel-Level 1:1)", "2 (Standard 2x2)", "4 (Fast 4x4)", "8 (Overview 8x8)"],
            state="readonly",
            width=20
        )
        cell_combo.pack(side=tk.RIGHT)
        cell_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_grid_settings())

        btn_recalc = ttk.Button(prep_group, text="🔄 Apply Grid Settings", command=self._apply_grid_settings)
        btn_recalc.pack(fill=tk.X, pady=(4, 2))

        # 3. Navigation & Solving Section
        solve_group = ttk.LabelFrame(left_frame, text=" 3. Route Finding ", padding=8)
        solve_group.pack(fill=tk.X, padx=5, pady=4)

        pts_frame = ttk.Frame(solve_group)
        pts_frame.pack(fill=tk.X, pady=(0, 4))

        self.lbl_start_status = ttk.Label(pts_frame, text="Start: Not Set", foreground="#047857", font=("Segoe UI", 9, "bold"))
        self.lbl_start_status.pack(side=tk.LEFT, expand=True, anchor=tk.W)

        self.lbl_goal_status = ttk.Label(pts_frame, text="Goal: Not Set", foreground="#b91c1c", font=("Segoe UI", 9, "bold"))
        self.lbl_goal_status.pack(side=tk.RIGHT, expand=True, anchor=tk.E)

        btn_points_row = ttk.Frame(solve_group)
        btn_points_row.pack(fill=tk.X, pady=2)
        ttk.Button(btn_points_row, text="🎯 Set Start", command=lambda: self._set_selection_target("start")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_points_row, text="🏁 Set Goal", command=lambda: self._set_selection_target("goal")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_points_row, text="🧹 Clear Points", command=self.clear_points).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        self.btn_solve = ttk.Button(
            solve_group,
            text="🚀 Find Shortest Route (Dijkstra vs A*)",
            style="Accent.TButton",
            command=self.run_solvers
        )
        self.btn_solve.pack(fill=tk.X, pady=(6, 2))

        # 4. Performance Comparison Section
        results_group = ttk.LabelFrame(left_frame, text=" 4. Performance Benchmark (Real Measurements) ", padding=8)
        results_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=4)

        # Treeview Comparison Table
        columns = ("metric", "dijkstra", "astar", "comparison")
        self.tree_results = ttk.Treeview(results_group, columns=columns, show="headings", height=5)
        self.tree_results.heading("metric", text="Metric")
        self.tree_results.heading("dijkstra", text="Dijkstra")
        self.tree_results.heading("astar", text="A* Search")
        self.tree_results.heading("comparison", text="Comparison")

        self.tree_results.column("metric", width=120, anchor=tk.W)
        self.tree_results.column("dijkstra", width=85, anchor=tk.CENTER)
        self.tree_results.column("astar", width=85, anchor=tk.CENTER)
        self.tree_results.column("comparison", width=110, anchor=tk.CENTER)
        self.tree_results.pack(fill=tk.X, pady=(0, 4))

        # Route View Mode & Export
        view_row = ttk.Frame(results_group)
        view_row.pack(fill=tk.X, pady=2)
        ttk.Label(view_row, text="Route Display:").pack(side=tk.LEFT)
        self.var_view_mode = tk.StringVar(value="both")
        cb_view = ttk.Combobox(
            view_row,
            textvariable=self.var_view_mode,
            values=["Both Routes Overlaid", "Dijkstra Only", "A* Only", "Explored Nodes Frontier"],
            state="readonly",
            width=22
        )
        cb_view.pack(side=tk.RIGHT)
        cb_view.bind("<<ComboboxSelected>>", self._on_view_mode_changed)

        # Summary text box
        self.txt_summary = tk.Text(results_group, height=4, wrap=tk.WORD, font=("Segoe UI", 8), bg="#f8fafc", relief=tk.SOLID, bd=1)
        self.txt_summary.pack(fill=tk.BOTH, expand=True, pady=4)
        self.txt_summary.insert(tk.END, "Upload an image, set Start (Green) and Goal (Red), then click 'Find Shortest Route' to run real performance benchmarks.")
        self.txt_summary.config(state=tk.DISABLED)

        # Save Result Button
        ttk.Button(results_group, text="💾 Save Route Image...", command=self.save_solved_image).pack(fill=tk.X, pady=(2, 0))

        # =========================================================================
        # RIGHT SIDE: Image Viewport & Canvas
        # =========================================================================
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # Canvas toolbar / prompt
        canvas_header = ttk.Frame(right_frame)
        canvas_header.pack(fill=tk.X, pady=(0, 4))
        self.lbl_instruction = ttk.Label(
            canvas_header,
            text="👉 Click on the image to place START (Lime Green), then click to place GOAL (Red).",
            font=("Segoe UI", 10, "bold"),
            foreground="#1e40af"
        )
        self.lbl_instruction.pack(side=tk.LEFT)

        # Canvas with scrollbars
        canvas_container = ttk.Frame(right_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, bg="#1e293b", cursor="crosshair", highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        canvas_container.rowconfigure(0, weight=1)
        canvas_container.columnconfigure(0, weight=1)

        # Canvas Event Bindings
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self._refresh_canvas_display())

        # Status Bar at bottom
        self.status_var = tk.StringVar(value="Ready. Welcome to Floor Plan & Maze Route Solver.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, font=("Segoe UI", 8), padding=(6, 3))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # =========================================================================
    # CORE LOGIC: Image Loading, Grid Computing, Canvas Display
    # =========================================================================

    def _parse_cell_size(self) -> int:
        raw = self.var_cell_size.get()
        try:
            return int(raw.split()[0])
        except Exception:
            return 2

    def _load_default_sample(self):
        sample_path = os.path.join("examples", "sample_maze.png")
        if not os.path.exists(sample_path):
            os.makedirs("examples", exist_ok=True)
            img = generate_sample_maze(400, 400, cell_dimension=20, wall_thickness=4)
            img.save(sample_path)

        self.load_image_from_path(sample_path)
        # Pre-select standard start & goal coordinates for immediate demo convenience
        self._set_sample_default_points()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Floor Plan or Maze Image",
            filetypes=[
                ("Supported Images", "*.png;*.jpg;*.jpeg"),
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg;*.jpeg"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.load_image_from_path(file_path)

    def load_sample_maze(self):
        sample_path = os.path.join("examples", "sample_maze.png")
        if not os.path.exists(sample_path):
            os.makedirs("examples", exist_ok=True)
            img = generate_sample_maze(400, 400, cell_dimension=20, wall_thickness=4)
            img.save(sample_path)

        self.load_image_from_path(sample_path)
        self._set_sample_default_points()

    def _set_sample_default_points(self):
        """Places convenient default start & goal points on the sample maze."""
        if self.processor.grid_rows > 10 and self.processor.grid_cols > 10:
            # Entrance at top-left corridor, exit at bottom-right corridor
            s_candidate = (4, 4)
            g_candidate = (self.processor.grid_rows - 5, self.processor.grid_cols - 5)
            if self.processor.is_cell_walkable(*s_candidate) and self.processor.is_cell_walkable(*g_candidate):
                self.start_coord = s_candidate
                self.goal_coord = g_candidate
                self.next_selection_mode = "done"
                self._update_point_labels()
                self._refresh_canvas_display()
                self.status_var.set("Sample maze ready with default Start and Goal points. Click 'Find Shortest Route' to solve!")

    def load_image_from_path(self, file_path: str):
        try:
            self.processor.load_image(file_path)
            self.clear_points()
            self._apply_grid_settings()

            filename = os.path.basename(file_path)
            dim_text = f"{self.processor.width}x{self.processor.height} px"
            self.lbl_image_info.config(text=f"Image: {filename} ({dim_text})")
            self.status_var.set(f"Loaded '{filename}' successfully. Click image to set Start point.")
        except Exception as ex:
            messagebox.showerror("Image Load Error", f"Failed to load image:\n{str(ex)}")

    def _toggle_advanced_settings(self):
        if self.advanced_visible:
            self.frame_advanced.pack_forget()
            self.btn_toggle_adv.config(text="⚙️ Advanced Settings (Polarity & Grid) ▼")
            self.advanced_visible = False
        else:
            self.frame_advanced.pack(fill=tk.X, pady=(2, 4))
            self.btn_toggle_adv.config(text="⚙️ Advanced Settings (Polarity & Grid) ▲")
            self.advanced_visible = True

    def _on_polarity_changed(self):
        if self.var_polarity.get() == "bright":
            self.lbl_active_mode.config(
                text="Walkable Mode: Bright / Light areas (Default)",
                foreground="#047857"
            )
        else:
            self.lbl_active_mode.config(
                text="Walkable Mode: Dark / Black areas (Inverted)",
                foreground="#7c3aed"
            )
        self._apply_grid_settings()

    def _on_threshold_slider_change(self, val):
        thresh_int = int(float(val))
        self.thresh_val_label.config(text=str(thresh_int))

    def _apply_grid_settings(self):
        if self.processor.original_image is None:
            return

        thresh = int(float(self.slider_threshold.get()))
        polarity = (self.var_polarity.get() == "bright")
        cell_size = self._parse_cell_size()

        self.processor.compute_grid(threshold=thresh, walkable_is_bright=polarity, cell_size=cell_size)

        # Clear existing pathfinding results because grid geometry has changed
        self.dijkstra_result = None
        self.astar_result = None
        self._clear_results_table()

        # Check if existing start/goal points are still walkable
        start_cleared = False
        goal_cleared = False

        if self.start_coord:
            if not self.processor.is_cell_walkable(*self.start_coord):
                self.start_coord = None
                start_cleared = True

        if self.goal_coord:
            if not self.processor.is_cell_walkable(*self.goal_coord):
                self.goal_coord = None
                goal_cleared = True

        self._update_point_labels()
        self._refresh_canvas_display()

        status_msg = f"Grid updated: {self.processor.grid_rows}x{self.processor.grid_cols} cells (Thresh: {thresh}, Polarity: {'Bright' if polarity else 'Dark'})."
        if start_cleared and goal_cleared:
            status_msg += " Start and Goal points cleared (now blocked by obstacles)."
        elif start_cleared:
            status_msg += " Start point cleared (now blocked by obstacle)."
        elif goal_cleared:
            status_msg += " Goal point cleared (now blocked by obstacle)."

        self.status_var.set(status_msg)

    def _update_point_labels(self):
        if self.start_coord:
            self.lbl_start_status.config(text=f"Start: ({self.start_coord[0]}, {self.start_coord[1]})", foreground="#047857")
        else:
            self.lbl_start_status.config(text="Start: Not Set", foreground="#6b7280")

        if self.goal_coord:
            self.lbl_goal_status.config(text=f"Goal: ({self.goal_coord[0]}, {self.goal_coord[1]})", foreground="#b91c1c")
        else:
            self.lbl_goal_status.config(text="Goal: Not Set", foreground="#6b7280")

        if not self.start_coord:
            self.next_selection_mode = "start"
            self.lbl_instruction.config(text="👉 Click on the image to place START (Lime Green).", foreground="#1e40af")
        elif not self.goal_coord:
            self.next_selection_mode = "goal"
            self.lbl_instruction.config(text="👉 Click on the image to place GOAL (Red).", foreground="#991b1b")
        else:
            self.next_selection_mode = "done"
            self.lbl_instruction.config(text="✅ Points selected! Click 'Find Shortest Route' to execute Dijkstra & A*.", foreground="#065f46")

    def _set_selection_target(self, target: str):
        self.next_selection_mode = target
        if target == "start":
            self.lbl_instruction.config(text="👉 Click on the image to place START (Lime Green).", foreground="#1e40af")
        else:
            self.lbl_instruction.config(text="👉 Click on the image to place GOAL (Red).", foreground="#991b1b")

    def clear_points(self):
        self.start_coord = None
        self.goal_coord = None
        self.dijkstra_result = None
        self.astar_result = None
        self.next_selection_mode = "start"
        self._clear_results_table()
        self._update_point_labels()
        self._refresh_canvas_display()
        self.status_var.set("Cleared Start, Goal, and solution routes.")

    # =========================================================================
    # CANVAS CLICK & COORDINATE MAPPING
    # =========================================================================

    def _on_canvas_click(self, event):
        if self.processor.original_image is None or self.processor.grid_rows == 0:
            return

        # Canvas coordinates with scroll offset
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Convert canvas click to original image pixel coordinates
        img_x = (canvas_x - self.display_offset_x) / self.display_scale
        img_y = (canvas_y - self.display_offset_y) / self.display_scale

        if not (0 <= img_x < self.processor.width and 0 <= img_y < self.processor.height):
            return  # Clicked outside image region

        # Convert original image pixel to discrete grid coordinate (row, col)
        grid_row, grid_col = self.processor.image_to_grid_coord(img_x, img_y)

        # Validate walkability
        is_walkable, msg = self.processor.get_cell_status_description(grid_row, grid_col)
        if not is_walkable:
            messagebox.showwarning(
                "Blocked Obstacle Selected",
                f"Selected position ({int(img_x)}, {int(img_y)}) is classified as BLOCKED.\n\n"
                f"{msg}\n\n"
                "Tip: Click on a walkable floor/corridor or adjust the brightness threshold."
            )
            self.status_var.set(f"Blocked cell selected at grid row={grid_row}, col={grid_col}. Please choose a walkable cell.")
            return

        # Assign to Start or Goal
        if self.next_selection_mode == "start" or (self.start_coord is None):
            self.start_coord = (grid_row, grid_col)
            self.status_var.set(f"Start set at grid row={grid_row}, col={grid_col} (image x={int(img_x)}, y={int(img_y)}).")
            # Next automatically prompt for Goal
            if self.goal_coord is None:
                self.next_selection_mode = "goal"
            else:
                self.next_selection_mode = "done"
        else:
            self.goal_coord = (grid_row, grid_col)
            self.status_var.set(f"Goal set at grid row={grid_row}, col={grid_col} (image x={int(img_x)}, y={int(img_y)}).")
            self.next_selection_mode = "done"

        # Invalidate previous path results when points change
        self.dijkstra_result = None
        self.astar_result = None
        self._clear_results_table()

        self._update_point_labels()
        self._refresh_canvas_display()

    # =========================================================================
    # ROUTE RENDERING & CANVAS DRAWING
    # =========================================================================

    def _render_current_overlay(self) -> Image.Image:
        if self.processor.original_image is None:
            return Image.new("RGB", (400, 400), (30, 41, 59))

        path_to_draw = None
        visited_to_draw = None
        path_color = (0, 220, 255)  # Electric cyan

        mode = self.var_view_mode.get()

        if mode == "A* Only" and self.astar_result and self.astar_result.success:
            path_to_draw = self.astar_result.path
            path_color = (255, 140, 0)  # Bright amber/orange
        elif mode == "Dijkstra Only" and self.dijkstra_result and self.dijkstra_result.success:
            path_to_draw = self.dijkstra_result.path
            path_color = (0, 220, 255)  # Cyan
        elif mode == "Explored Nodes Frontier":
            if self.astar_result and self.astar_result.visited_cells:
                visited_to_draw = self.astar_result.visited_cells
            elif self.dijkstra_result and self.dijkstra_result.visited_cells:
                visited_to_draw = self.dijkstra_result.visited_cells
            if self.astar_result and self.astar_result.success:
                path_to_draw = self.astar_result.path
        else:
            # "Both Routes Overlaid" (default)
            if self.astar_result and self.astar_result.success:
                path_to_draw = self.astar_result.path
                path_color = (0, 240, 255)

        overlay = self.processor.render_route_overlay(
            path=path_to_draw,
            start=self.start_coord,
            goal=self.goal_coord,
            visited_cells=visited_to_draw,
            path_color=path_color,
            line_width=max(3, self.processor.cell_size)
        )
        return overlay

    def _refresh_canvas_display(self):
        if self.processor.original_image is None:
            return

        overlay_img = self._render_current_overlay()
        self.current_pil_image = overlay_img

        canvas_w = max(50, self.canvas.winfo_width())
        canvas_h = max(50, self.canvas.winfo_height())

        img_w, img_h = overlay_img.size

        # Fit image snugly within canvas while preserving aspect ratio
        scale_x = canvas_w / img_w
        scale_y = canvas_h / img_h
        scale = min(scale_x, scale_y, 1.0)  # Don't upscale past 100%

        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))

        self.display_scale = scale
        self.display_offset_x = (canvas_w - disp_w) // 2
        self.display_offset_y = (canvas_h - disp_h) // 2

        resized = overlay_img.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self.display_photo_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.display_offset_x,
            self.display_offset_y,
            anchor=tk.NW,
            image=self.display_photo_image
        )
        self.canvas.config(scrollregion=(0, 0, canvas_w, canvas_h))

    def _on_view_mode_changed(self, event=None):
        self._refresh_canvas_display()

    # =========================================================================
    # SOLVING & PERFORMANCE BENCHMARKING
    # =========================================================================

    def run_solvers(self):
        if self.processor.original_image is None:
            messagebox.showwarning("No Image", "Please upload an image or load a sample maze first.")
            return

        if self.start_coord is None:
            messagebox.showwarning("Missing Start Point", "Please click on the image to set a valid Start position (Lime Green).")
            return

        if self.goal_coord is None:
            messagebox.showwarning("Missing Goal Point", "Please click on the image to set a valid Goal position (Red).")
            return

        # Ensure both Start and Goal are walkable on the current grid before solving
        if not self.processor.is_cell_walkable(*self.start_coord):
            messagebox.showwarning(
                "Invalid Start Point",
                f"The selected Start position at grid ({self.start_coord[0]}, {self.start_coord[1]}) is currently BLOCKED by an obstacle/wall.\n\n"
                "Please click a valid walkable floor area or adjust the threshold/polarity settings before solving."
            )
            self.start_coord = None
            self._update_point_labels()
            self._refresh_canvas_display()
            return

        if not self.processor.is_cell_walkable(*self.goal_coord):
            messagebox.showwarning(
                "Invalid Goal Point",
                f"The selected Goal position at grid ({self.goal_coord[0]}, {self.goal_coord[1]}) is currently BLOCKED by an obstacle/wall.\n\n"
                "Please click a valid walkable floor area or adjust the threshold/polarity settings before solving."
            )
            self.goal_coord = None
            self._update_point_labels()
            self._refresh_canvas_display()
            return

        self.status_var.set("Running Dijkstra's Algorithm and A* Search...")
        self.root.update_idletasks()

        grid = self.processor.grid
        start = self.start_coord
        goal = self.goal_coord

        # Run Dijkstra
        d_res = dijkstra(grid, start, goal)
        self.dijkstra_result = d_res

        # Run A* Search
        a_res = astar(grid, start, goal)
        self.astar_result = a_res

        self._populate_results_table(d_res, a_res)
        self._refresh_canvas_display()

        if not d_res.success and not a_res.success:
            messagebox.showinfo(
                "No Route Exists",
                "Neither algorithm could find a path between Start and Goal.\n\n"
                "Reason: The Goal position is completely walled off from the Start position by obstacles.\n"
                "Try placing points in connected corridors or adjusting the threshold."
            )
            self.status_var.set("No route exists: Goal is disconnected from Start by obstacles.")
        else:
            self.status_var.set(
                f"Solved successfully! Optimal path cost: {a_res.cost} steps. "
                f"A* explored {a_res.nodes_expanded} nodes vs Dijkstra's {d_res.nodes_expanded} nodes."
            )

    def _clear_results_table(self):
        for item in self.tree_results.get_children():
            self.tree_results.delete(item)
        self.txt_summary.config(state=tk.NORMAL)
        self.txt_summary.delete("1.0", tk.END)
        self.txt_summary.insert(tk.END, "Points modified. Click 'Find Shortest Route' to execute benchmark.")
        self.txt_summary.config(state=tk.DISABLED)

    def _populate_results_table(self, d_res: PathfindingResult, a_res: PathfindingResult):
        self._clear_results_table()

        # Compute execution time comparison: A* relative to Dijkstra
        time_diff = d_res.execution_time_ms - a_res.execution_time_ms
        if abs(time_diff) < 0.01:
            comp_time = "Equal"
            time_summary = "both algorithms finished with approximately equal execution time"
        elif time_diff > 0:
            # Dijkstra took more time than A* => A* was faster
            comp_time = f"A* {time_diff:.2f} ms faster"
            time_summary = f"A* was {time_diff:.2f} ms faster than Dijkstra"
        else:
            # Dijkstra took less time than A* => A* was slower
            comp_time = f"A* {abs(time_diff):.2f} ms slower"
            time_summary = f"A* was {abs(time_diff):.2f} ms slower than Dijkstra"

        # Compute peak memory comparison: A* relative to Dijkstra
        mem_diff = d_res.peak_memory_kb - a_res.peak_memory_kb
        if abs(mem_diff) < 0.1:
            comp_mem = "Equal"
        elif mem_diff > 0:
            comp_mem = f"A* {mem_diff:.1f} KB less"
        else:
            comp_mem = f"A* {abs(mem_diff):.1f} KB more"

        # Compute nodes expanded comparison
        node_diff = d_res.nodes_expanded - a_res.nodes_expanded
        if node_diff == 0:
            comp_nodes = "Equal"
        elif node_diff > 0:
            pct_savings = (node_diff / max(1, d_res.nodes_expanded)) * 100.0
            comp_nodes = f"A* saved {pct_savings:.1f}% ({node_diff} fewer)"
        else:
            comp_nodes = f"A* expanded {-node_diff} more"

        if not d_res.success or not a_res.success:
            self.tree_results.insert("", tk.END, values=("Path Found?", "NO" if not d_res.success else "YES", "NO" if not a_res.success else "YES", "Disconnected"))
            self.tree_results.insert("", tk.END, values=("Nodes Visited", str(d_res.nodes_expanded), str(a_res.nodes_expanded), comp_nodes))
            self.tree_results.insert("", tk.END, values=("Time (ms)", f"{d_res.execution_time_ms:.2f}", f"{a_res.execution_time_ms:.2f}", comp_time))
            self.tree_results.insert("", tk.END, values=("Memory (KB)", f"{d_res.peak_memory_kb:.1f}", f"{a_res.peak_memory_kb:.1f}", comp_mem))
            
            self.txt_summary.config(state=tk.NORMAL)
            self.txt_summary.delete("1.0", tk.END)
            self.txt_summary.insert(tk.END, "⚠️ Goal is unreachable from Start due to obstacle barriers. Both algorithms exhausted connected components.")
            self.txt_summary.config(state=tk.DISABLED)
            return

        # 1. Path Cost
        cost_match = "Identical (Optimal)" if d_res.cost == a_res.cost else "Mismatch"
        self.tree_results.insert("", tk.END, values=("Path Cost (Steps)", str(d_res.cost), str(a_res.cost), cost_match))

        # 2. Nodes Expanded
        self.tree_results.insert("", tk.END, values=("Nodes Expanded", str(d_res.nodes_expanded), str(a_res.nodes_expanded), comp_nodes))

        # 3. Execution Time
        self.tree_results.insert("", tk.END, values=("Time (ms)", f"{d_res.execution_time_ms:.2f}", f"{a_res.execution_time_ms:.2f}", comp_time))

        # 4. Peak Memory
        self.tree_results.insert("", tk.END, values=("Peak Memory (KB)", f"{d_res.peak_memory_kb:.1f}", f"{a_res.peak_memory_kb:.1f}", comp_mem))

        # 5. Summary Text
        self.txt_summary.config(state=tk.NORMAL)
        self.txt_summary.delete("1.0", tk.END)
        summary = (
            f"• Both algorithms found the optimal shortest path of {a_res.cost} steps.\n"
            f"• Nodes: Dijkstra expanded {d_res.nodes_expanded} vs A* {a_res.nodes_expanded} ({comp_nodes}).\n"
            f"• Execution Time: {time_summary}.\n"
            f"• Note: Runtime & memory measurements come from real runs and naturally vary by machine and CPU load."
        )
        self.txt_summary.insert(tk.END, summary)
        self.txt_summary.config(state=tk.DISABLED)

    # =========================================================================
    # EXPORT & DIALOGS
    # =========================================================================

    def save_solved_image(self):
        if self.current_pil_image is None or (self.start_coord is None and self.goal_coord is None):
            messagebox.showwarning("Save Image", "No solved route or markers available to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Solved Route Image",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg;*.jpeg"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            try:
                # Render full resolution overlay
                full_res_overlay = self._render_current_overlay()
                full_res_overlay.save(file_path)
                messagebox.showinfo("Image Saved", f"Successfully saved route image to:\n{file_path}")
                self.status_var.set(f"Saved route image to {os.path.basename(file_path)}.")
            except Exception as ex:
                messagebox.showerror("Save Error", f"Failed to save image:\n{str(ex)}")

    def show_algorithm_info(self):
        msg = (
            "Algorithms and Data Structures Overview:\n\n"
            "1. Dijkstra's Algorithm:\n"
            "   - Uniform-cost search exploring outward radially from the Start node.\n"
            "   - Priority Queue ordered strictly by g(n) = exact accumulated path cost.\n"
            "   - Time Complexity: O(V log V + E) where V = R x C grid cells, E <= 4V.\n"
            "   - Guarantees the shortest path on any non-negative edge graph.\n\n"
            "2. A* Search:\n"
            "   - Informed search combining exact distance g(n) with heuristic estimate h(n).\n"
            "   - Priority Queue ordered by f(n) = g(n) + h(n).\n"
            "   - Heuristic used: Manhattan Distance h(n) = |r_n - r_g| + |c_n - c_g|.\n"
            "   - Admissibility: On a 4-connected grid with unit step cost, Manhattan distance\n"
            "     never overestimates true distance (h(n) <= h*(n)).\n"
            "   - Consistency: Triangle inequality holds (h(u) <= c(u,v) + h(v)).\n"
            "   - Guarantee: A* returns the exact same optimal cost as Dijkstra while\n"
            "     pruning unpromising search branches and expanding fewer nodes."
        )
        messagebox.showinfo("Algorithm Explanation", msg)

    def show_limitations_info(self):
        messagebox.showinfo("Image Classification Limitations", ImageGridProcessor.LIMITATIONS_EXPLANATION)

    def show_about_info(self):
        msg = (
            "Floor Plan & Maze Route Solver\n"
            "Algorithms and Data Structures (ADSA) Project\n\n"
            "Features:\n"
            "• Image-to-grid classification with threshold & polarity controls\n"
            "• Precise coordinate mapping between display, image, and grid\n"
            "• Dijkstra & A* pathfinding implementations\n"
            "• Real-time timing (ms) and peak memory (KB) benchmarking\n"
            "• High-resolution route export"
        )
        messagebox.showinfo("About", msg)


def main():
    root = tk.Tk()
    app = FloorPlanApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
