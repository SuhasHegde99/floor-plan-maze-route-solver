# Floor Plan & Maze Route Solver
**Algorithms and Data Structures (ADSA) Project**

An interactive desktop application built in Python and Tkinter to load 2D floor plans and maze images, classify them into walkable and obstacle grid graphs, and compute the shortest path using **Dijkstra's Algorithm** and **A\* Search**. The application benchmarks both algorithms side-by-side using real runtime, memory, and node expansion measurements.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites & System Requirements](#prerequisites--system-requirements)
3. [Quick Start (Windows)](#quick-start-windows)
4. [Manual Installation & Launch](#manual-installation--launch)
5. [User Guide & Features](#user-guide--features)
6. [Algorithm & Data Structure Details](#algorithm--data-structure-details)
   - [Dijkstra's Algorithm](#dijkstras-algorithm)
   - [A* Search & Manhattan Heuristic](#a-search--manhattan-heuristic)
   - [Heuristic Admissibility & Consistency](#heuristic-admissibility--consistency)
7. [Performance Benchmark Interpretation](#performance-benchmark-interpretation)
8. [Limitations of Image-Based Classification](#limitations-of-image-based-classification)
9. [Automated Testing](#automated-testing)
10. [Project File Structure](#project-file-structure)

---

## Project Overview

In pathfinding and robotics navigation, discrete 2D grid graphs are often derived from raster images such as architectural floor plans, blueprints, or overhead maze maps. 

This project bridges theoretical computer science concepts with applied image processing:
- Converts continuous images into discrete grid graphs with strict 4-directional connectivity (orthogonal steps only; no diagonal movements, no passing through walls).
- Solves for the optimal path using two fundamental shortest-path algorithms:
  - **Dijkstra's Algorithm** (Uniform-Cost Search)
  - **A\* Search** (Informed Search using the Manhattan-distance heuristic)
- Accurately tracks and compares empirical performance metrics (nodes expanded, execution time in milliseconds, peak Python heap memory in kilobytes).
- Visualizes the resulting shortest route overlaid onto the original image and exports high-resolution output files.

---

## Prerequisites & System Requirements

- **Operating System:** Windows 10 / 11 (also runs on macOS and Linux)
- **Python Version:** Python 3.9+ (tested on Python 3.15)
- **Tkinter:** Standard with official Python installers on Windows
- **Dependencies:**
  - `Pillow` (PIL) $\ge$ 10.0.0
  - `NumPy` $\ge$ 1.24.0

---

## Quick Start (Windows)

A turnkey Windows batch script (`run_windows.bat`) is included in the project root:

1. Double-click `run_windows.bat` in File Explorer, **or** open PowerShell in this directory and run:
   ```cmd
   .\run_windows.bat
   ```
2. The script will automatically:
   - Detect Python on your system.
   - Create or activate a project-local virtual environment (`.venv`).
   - Install dependencies from `requirements.txt`.
   - Run the automated test suite to verify code integrity.
   - Launch the graphical user interface (`app.py`).

---

## Manual Installation & Launch

If you prefer to run commands manually:

### 1. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(On Windows Command Prompt `cmd.exe`: `.venv\Scripts\activate.bat`)*

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Launch the Application
```powershell
python app.py
```

---

## User Guide & Features

```
+------------------------------------+-------------------------------------------+
| 1. Image Source                    | Canvas Viewport                           |
|    - [Upload Image] [Load Sample]  |                                           |
| 2. Grid & Walkability Settings     |  (Displays image, start [Green],          |
|    - Threshold Slider (0 - 255)    |   goal [Red], and the computed route)     |
|    - Bright / Dark Walkable Toggle |                                           |
|    - Grid Resolution (1x, 2x, 4x)  |                                           |
| 3. Route Finding                   |                                           |
|    - [Set Start] [Set Goal]        |                                           |
|    - [Find Shortest Route]         |                                           |
| 4. Performance Benchmark Table     |                                           |
|    - Metric | Dijkstra | A* | Comp |                                           |
|    - [Save Route Image]            |                                           |
+------------------------------------+-------------------------------------------+
```

### Step-by-Step Workflow:
1. **Load Image:**
   - Click **🎲 Load Sample Maze** to immediately test with the included solvable maze (`examples/sample_maze.png`), or click **📁 Upload Image** to pick any PNG, JPG, or JPEG file.
2. **Adjust Walkability Settings:**
   - **Brightness Threshold:** Controls the threshold between walls and corridors (default: 100).
     - *Threshold Concept:* Separates walkable pixels from blocked pixels (0–255).
     - *Bright mode (default):* Higher thresholds classify fewer pixels as walkable (stricter / narrower paths); lower thresholds classify more pixels as walkable (more permissive).
   - **Advanced Settings (Expandable):**
     - Click **⚙️ Advanced Settings (Polarity & Grid)** to reveal walkability polarity and resolution controls.
     - *Bright / Light areas walkable (Default):* For standard mazes and blueprints where corridors are white/light gray and walls are black/dark.
     - *Dark / Black areas walkable:* For inverted mazes, dark floor plans, or line drawings.
   - **Grid Resolution:** Controls downsampling (1x = exact pixel-by-pixel, 2x = standard 2x2 cells, 4x = fast). Using 2x or 4x allows high-resolution photos (e.g. 2000x2000) to be solved within milliseconds.
3. **Select Start and Goal:**
   - Click directly on the image to set the **Start** point (marked with a bright Lime Green pin).
   - Click again to set the **Goal** point (marked with a bright Red pin).
   - If you click on a wall/obstacle, the application alerts you with the cell's status and prompts you to select a valid walkable location.
4. **Find Shortest Route:**
   - Click **🚀 Find Shortest Route (Dijkstra vs A\*)**.
   - Both algorithms run on the exact same grid graph.
   - The computed optimal path is highlighted in electric cyan / vivid color on the image.
5. **Inspect Performance & Toggle Views:**
   - View the benchmark table comparing Path Cost, Nodes Expanded, Real Execution Time (ms), and Peak Python Memory (KB).
   - Use the **Route Display** dropdown to toggle between *Both Routes Overlaid*, *Dijkstra Only*, *A\* Only*, or *Explored Nodes Frontier* (which shows all cells visited during search).
6. **Save Result:**
   - Click **💾 Save Route Image...** to export the high-resolution image with the rendered path and markers.

---

## Algorithm & Data Structure Details

### Graph Representation
The floor plan is modeled as an implicit or discrete 2D grid graph $G = (V, E)$:
- **Vertices ($V$):** Every walkable cell $(r, c)$ where $0 \le r < \text{rows}$ and $0 \le c < \text{cols}$ and $\text{grid}[r][c] == \text{True}$.
- **Edges ($E$):** Strictly orthogonal 4-directional connections:
  $$\text{Neighbors}(r, c) = \{(r-1, c), (r+1, c), (r, c-1), (r, c+1)\} \cap V$$
- **Edge Weight:** Uniform cost $c(u, v) = 1.0$ for every step. No diagonal movements are allowed, preventing paths from "cutting corners" across diagonal walls.

---

### Dijkstra's Algorithm
Dijkstra's Algorithm is a uniform-cost search algorithm that systematically discovers the shortest path from a source node to all reachable nodes by order of accumulated distance $g(n)$.

- **Data Structures:**
  - `heapq` (Min-Heap Priority Queue): Stores tuples $(g(u), \text{counter}, u)$.
  - `distances`: Hash map (`dict`) mapping each coordinate $(r, c) \to g(u)$ (shortest known distance from Start).
  - `predecessors`: Hash map mapping each node to its parent node for linear path reconstruction.
  - `visited_cells`: Hash set of closed/expanded nodes.
- **Time Complexity:**
  $$\mathcal{O}(|V| \log |V| + |E|)$$
  With $|E| \le 4|V|$, this evaluates to $\mathcal{O}(|V| \log |V|)$.
- **Space Complexity:**
  $\mathcal{O}(|V|)$ to store the distance dictionary, predecessor tree, and priority queue.

---

### A* Search & Manhattan Heuristic
A\* Search improves upon Dijkstra's Algorithm by using an evaluation function:
$$f(n) = g(n) + h(n)$$
where:
- $g(n)$ is the exact cost from Start to node $n$.
- $h(n)$ is the heuristic estimate of the remaining cost from $n$ to Goal.

In our 4-connected unit grid, the heuristic used is the **Manhattan Distance** ($L_1$ norm):
$$h(r_n, c_n) = |r_n - r_{\text{goal}}| + |c_n - c_{\text{goal}}|$$

#### Heuristic Admissibility & Consistency Proof
1. **Admissibility ($h(n) \le h^*(n)$):**
   - On a grid with only horizontal and vertical movements of cost 1, the theoretical minimum number of steps between $(r_1, c_1)$ and $(r_2, c_2)$ without any obstacles is $|r_1 - r_2| + |c_1 - c_2|$.
   - The presence of walls can only force detours, which increases the required steps; it can never decrease them.
   - Therefore, $h(n)$ never overestimates the true remaining cost $h^*(n)$. An admissible heuristic guarantees that A\* will never return a suboptimal path.
2. **Consistency / Monotonicity ($h(u) \le c(u, v) + h(v)$):**
   - For any two neighboring cells $u$ and $v$, the step cost $c(u, v) = 1$.
   - By the triangle inequality:
     $$|r_u - r_g| \le |r_u - r_v| + |r_v - r_g|$$
     $$|c_u - c_g| \le |c_u - c_v| + |c_v - c_g|$$
     $$h(u) \le (|r_u - r_v| + |c_u - c_v|) + h(v)$$
   - Since $u$ and $v$ are adjacent orthogonal neighbors, $|r_u - r_v| + |c_u - c_v| = 1 = c(u, v)$.
   - Hence, $h(u) \le 1 + h(v)$.
   - Because the heuristic is consistent, once a node is expanded by A\*, its shortest path is guaranteed to have been found; closed nodes never need to be re-opened.

---

## Performance Benchmark Interpretation

When you run both algorithms on the same maze, the application records live statistics:

| Metric | Dijkstra's Algorithm | A* Search | Analysis |
| :--- | :---: | :---: | :--- |
| **Path Cost (Steps)** | $K$ | $K$ | **Guaranteed Identical:** Both algorithms find the exact optimal shortest path. |
| **Nodes Expanded** | High ($N_D$) | Lower ($N_A$) | **Search Efficiency:** Dijkstra expands nodes radially in all directions. A* is guided toward the goal, pruning branches that lead away from the objective. |
| **Execution Time** | Measured (ms) | Measured (ms) | Measured via high-resolution `time.perf_counter()`. |
| **Peak Heap Memory** | Measured (KB) | Measured (KB) | Measured via Python's standard `tracemalloc` module. |

> [!NOTE]
> **Variation Note:** Execution time and memory measurements are derived from actual live runs on your CPU. They naturally fluctuate across runs due to background OS processes, CPU thread scheduling, cache warmth, and Python garbage collection cycles.

---

## Limitations of Image-Based Classification

While image-to-grid thresholding is fast and practical, real-world floor plans and photos have known physical limitations:

1. **Uneven Lighting & Shadows:**
   Photographs of physical maps or blueprints often have lighting gradients or shadows. A shadowed white corridor can have a lower brightness value than a brightly lit wall, causing false obstacles.
2. **Resolution & Thin Walls:**
   If a maze or floor plan contains 1-pixel or 2-pixel wide walls, aggressive downsampling or blurring will cause walls to vanish or break, creating false shortcuts.
3. **Architectural Annotations:**
   Blueprints contain room labels, dimension arrows, electrical lines, and door-swing arcs. Pixel thresholding treats all these non-structural lines as solid walls unless filtered.
4. **JPEG Ringing & Compression Noise:**
   Lossy JPEG compression introduces high-frequency ringing artifacts along sharp black-white boundaries, which can generate spurious 1-pixel blocked specks in corridors.
5. **Texture and Carpet Patterns:**
   Real architectural floor plans may use textured wood or hatched tiles that produce speckled walkable/blocked patterns unless pre-smoothed.

---

## Automated Testing

The automated test suite in `tests/` verifies the mathematical correctness of both algorithms and the image processing pipeline across 20 test cases.

### Running Tests
Using the virtual environment Python:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Test Suite Coverage:
- `test_simple_grid_known_cost`: Verifies both algorithms return the exact known theoretical shortest path cost (8 steps on a 5x5 open grid).
- `test_obstacle_grid_detour`: Verifies correct navigation around complex U-shaped detour walls.
- `test_unreachable_goal`: Validates safe failure reporting when the goal is boxed off by walls.
- `test_start_equals_goal`: Edge case test when Start and Goal coordinates coincide (cost = 0).
- `test_multiple_random_grids_optimality_equivalence`: Generates diverse random grid configurations and checks that Dijkstra and A* always produce identical path costs.
- `test_blocked_start_or_goal`: Rejects starting or ending inside a wall.
- `test_out_of_bounds_coordinates`: Boundary checking against negative or oversized coordinates.
- `test_performance_metrics_real_values`: Ensures nodes expanded, time (ms), and memory (KB) are positive, genuine measurements.
- `test_coordinate_mapping_fidelity`: Verifies coordinate transformations between canvas, image, and grid.
- `test_gui_workflow`: Verifies full Tkinter GUI window initialization, sample loading, solving, and export.

---

## Project File Structure

```
adsaaa/
├── app.py                      # Main Tkinter desktop GUI application
├── pathfinding.py              # Independent Dijkstra and A* Search implementations
├── image_processing.py         # Image-to-grid conversion, thresholding, & route overlays
├── requirements.txt            # Python dependencies (Pillow, numpy)
├── run_windows.bat             # Turnkey Windows launch script
├── README.md                   # Complete documentation and user manual
├── examples/
│   └── sample_maze.png         # Generated sample maze for immediate testing
└── tests/
    ├── test_pathfinding.py     # Algorithm, heuristic, and image unit tests
    └── test_gui.py             # Tkinter GUI integration and workflow tests
```
