"""
pathfinding.py
==============
Implements Dijkstra's Algorithm and A* Search for 2D grid graphs.

Both algorithms operate on a 2D boolean grid (rows x cols) where:
  - True  => Walkable cell (passable floor/corridor)
  - False => Blocked cell (obstacle/wall)

Movement Rules:
  - 4-directional movement only: Up, Down, Left, Right
  - Unit movement cost: 1.0 per step
  - No diagonal movement allowed
  - Blocked cells cannot be traversed

Theoretical Guarantees:
  - Dijkstra's Algorithm finds the guaranteed shortest path by exploring
    nodes in increasing order of distance from the source.
  - A* Search uses the Manhattan distance heuristic h(n) = |r_n - r_g| + |c_n - c_g|.
    Because movement is strictly 4-directional and edge weights are 1.0,
    the Manhattan distance is admissible (h(n) <= true remaining distance)
    and consistent (triangle inequality holds: h(u) <= c(u,v) + h(v)).
    Therefore, A* is mathematically guaranteed to return the optimal shortest path
    while typically expanding significantly fewer nodes than Dijkstra.
"""

from __future__ import annotations
import heapq
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional, Dict

Coordinate = Tuple[int, int]  # (row, col)

# 4-directional orthogonal neighbors: (delta_row, delta_col)
ORTHOGONAL_NEIGHBORS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),  # North / Up
    (1, 0),   # South / Down
    (0, -1),  # West / Left
    (0, 1),   # East / Right
)


@dataclass
class PathfindingResult:
    """Stores the execution output and performance statistics of a search algorithm."""
    algorithm: str
    success: bool
    path: List[Coordinate] = field(default_factory=list)
    cost: int = 0
    nodes_expanded: int = 0
    execution_time_ms: float = 0.0
    peak_memory_kb: float = 0.0
    visited_cells: Set[Coordinate] = field(default_factory=set)
    message: str = ""

    @property
    def step_count(self) -> int:
        """Returns the number of steps (path length - 1) or 0 if no path."""
        return max(0, len(self.path) - 1) if self.success else 0


def manhattan_distance(p1: Coordinate, p2: Coordinate) -> int:
    """
    Computes the Manhattan distance (L1 norm) between two grid points.
    
    Formula: |r1 - r2| + |c1 - c2|
    Admissibility: On a 4-connected grid with uniform edge weight 1,
    the shortest possible path length between two points without obstacles
    is exactly the Manhattan distance. Obstacles can only increase this distance,
    never decrease it. Hence h(n) <= h*(n) everywhere.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_valid_neighbors(cell: Coordinate, grid: List[List[bool]]) -> List[Coordinate]:
    """
    Retrieves all 4-directional walkable neighbors within the grid boundaries.
    
    Args:
        cell: (row, col) tuple
        grid: 2D boolean grid where True = walkable, False = wall
        
    Returns:
        List of valid neighbor (row, col) coordinates.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    r, c = cell
    neighbors = []
    
    for dr, dc in ORTHOGONAL_NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc]:  # True represents a walkable cell
                neighbors.append((nr, nc))
                
    return neighbors


def reconstruct_path(predecessors: Dict[Coordinate, Optional[Coordinate]], goal: Coordinate) -> List[Coordinate]:
    """
    Backtracks from the goal node to the start node using the predecessor mapping.
    
    Returns:
        Ordered list of coordinates from Start to Goal inclusive.
    """
    path: List[Coordinate] = []
    curr: Optional[Coordinate] = goal
    while curr is not None:
        path.append(curr)
        curr = predecessors.get(curr)
    path.reverse()
    return path


def dijkstra(grid: List[List[bool]], start: Coordinate, goal: Coordinate) -> PathfindingResult:
    """
    Executes Dijkstra's Algorithm to find the shortest path from start to goal.
    
    Data Structures:
      - Priority Queue (min-heap): stores tuples of (cost, tie_breaker, coordinate)
      - distances: map of Coordinate -> shortest known distance from start
      - predecessors: map of Coordinate -> parent coordinate for path reconstruction
      - visited_cells: set of coordinates expanded (popped from PQ)
      
    Time Complexity:
      O(V * log V + E) where V is walkable grid cells, E <= 4*V edges.
      
    Space Complexity:
      O(V) to store distances, predecessors, and heap elements.
    """
    tracemalloc.start()
    t_start = time.perf_counter()

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Boundary & Walkability validation
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return PathfindingResult(
            algorithm="Dijkstra",
            success=False,
            execution_time_ms=(t_end - t_start) * 1000.0,
            peak_memory_kb=peak_mem / 1024.0,
            message="Start coordinate is outside the grid bounds."
        )

    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return PathfindingResult(
            algorithm="Dijkstra",
            success=False,
            execution_time_ms=(t_end - t_start) * 1000.0,
            peak_memory_kb=peak_mem / 1024.0,
            message="Goal coordinate is outside the grid bounds."
        )

    if not grid[start[0]][start[1]]:
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return PathfindingResult(
            algorithm="Dijkstra",
            success=False,
            execution_time_ms=(t_end - t_start) * 1000.0,
            peak_memory_kb=peak_mem / 1024.0,
            message="Start position is inside a blocked wall/obstacle."
        )

    if not grid[goal[0]][goal[1]]:
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return PathfindingResult(
            algorithm="Dijkstra",
            success=False,
            execution_time_ms=(t_end - t_start) * 1000.0,
            peak_memory_kb=peak_mem / 1024.0,
            message="Goal position is inside a blocked wall/obstacle."
        )

    # Base case: Start equals Goal
    if start == goal:
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return PathfindingResult(
            algorithm="Dijkstra",
            success=True,
            path=[start],
            cost=0,
            nodes_expanded=1,
            execution_time_ms=(t_end - t_start) * 1000.0,
            peak_memory_kb=peak_mem / 1024.0,
            visited_cells={start},
            message="Start and Goal are the same location."
        )

    # Initialize Dijkstra data structures
    distances: Dict[Coordinate, int] = {start: 0}
    predecessors: Dict[Coordinate, Optional[Coordinate]] = {start: None}
    visited_cells: Set[Coordinate] = set()

    # Priority queue: (cost, entry_count, coordinate)
    # entry_count serves as a strictly increasing tie-breaker to avoid coordinate comparison
    entry_count = 0
    pq: List[Tuple[int, int, Coordinate]] = [(0, entry_count, start)]
    nodes_expanded = 0
    found = False

    while pq:
        curr_dist, _, curr_node = heapq.heappop(pq)

        if curr_node in visited_cells:
            continue

        visited_cells.add(curr_node)
        nodes_expanded += 1

        if curr_node == goal:
            found = True
            break

        # Explore 4-connected neighbors
        for neighbor in get_valid_neighbors(curr_node, grid):
            if neighbor in visited_cells:
                continue

            tentative_dist = curr_dist + 1  # Unit edge weight
            if neighbor not in distances or tentative_dist < distances[neighbor]:
                distances[neighbor] = tentative_dist
                predecessors[neighbor] = curr_node
                entry_count += 1
                heapq.heappush(pq, (tentative_dist, entry_count, neighbor))

    t_end = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    execution_time_ms = (t_end - t_start) * 1000.0
    peak_memory_kb = peak_mem / 1024.0

    if not found:
        return PathfindingResult(
            algorithm="Dijkstra",
            success=False,
            path=[],
            cost=0,
            nodes_expanded=nodes_expanded,
            execution_time_ms=execution_time_ms,
            peak_memory_kb=peak_memory_kb,
            visited_cells=visited_cells,
            message="No route exists: Goal is disconnected from Start by obstacles."
        )

    path = reconstruct_path(predecessors, goal)
    return PathfindingResult(
        algorithm="Dijkstra",
        success=True,
        path=path,
        cost=len(path) - 1,
        nodes_expanded=nodes_expanded,
        execution_time_ms=execution_time_ms,
        peak_memory_kb=peak_memory_kb,
        visited_cells=visited_cells,
        message=f"Optimal shortest path found in {len(path) - 1} steps."
    )


def astar(grid: List[List[bool]], start: Coordinate, goal: Coordinate) -> PathfindingResult:
    """
    Executes A* Search using the Manhattan-distance heuristic.
    
    Data Structures:
      - Priority Queue (min-heap): stores tuples of (f_score, h_score, tie_breaker, coordinate)
        (Using h_score as secondary priority guides the search faster toward goal when f_scores tie)
      - g_score: exact distance from start to current node
      - f_score: g_score + h(node, goal)
      - predecessors: map of Coordinate -> parent coordinate for path reconstruction
      - closed_set: set of expanded coordinates
      
    Heuristic:
      h(r, c) = |r - goal_r| + |c - goal_c| (Manhattan Distance)
      Admissible and Consistent for 4-directional unit grid movements.
    """
    tracemalloc.start()
    t_start = time.perf_counter()

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Boundary & Walkability validation
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return PathfindingResult(
            algorithm="A* Search",
            success=False,
            execution_time_ms=(t_end - t_start) * 1000.0,
            peak_memory_kb=peak_mem / 1024.0,
            message="Start coordinate is outside the grid bounds."
        )

    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return PathfindingResult(
            algorithm="A* Search",
            success=False,
            execution_time_ms=(t_end - t_start) * 1000.0,
            peak_memory_kb=peak_mem / 1024.0,
            message="Goal coordinate is outside the grid bounds."
        )

    if not grid[start[0]][start[1]]:
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return PathfindingResult(
            algorithm="A* Search",
            success=False,
            execution_time_ms=(t_end - t_start) * 1000.0,
            peak_memory_kb=peak_mem / 1024.0,
            message="Start position is inside a blocked wall/obstacle."
        )

    if not grid[goal[0]][goal[1]]:
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return PathfindingResult(
            algorithm="A* Search",
            success=False,
            execution_time_ms=(t_end - t_start) * 1000.0,
            peak_memory_kb=peak_mem / 1024.0,
            message="Goal position is inside a blocked wall/obstacle."
        )

    # Base case: Start equals Goal
    if start == goal:
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return PathfindingResult(
            algorithm="A* Search",
            success=True,
            path=[start],
            cost=0,
            nodes_expanded=1,
            execution_time_ms=(t_end - t_start) * 1000.0,
            peak_memory_kb=peak_mem / 1024.0,
            visited_cells={start},
            message="Start and Goal are the same location."
        )

    # Initialize A* structures
    g_score: Dict[Coordinate, int] = {start: 0}
    predecessors: Dict[Coordinate, Optional[Coordinate]] = {start: None}
    closed_set: Set[Coordinate] = set()

    h_start = manhattan_distance(start, goal)
    entry_count = 0
    # PQ stores (f_score, h_score, tie_breaker, coordinate)
    open_pq: List[Tuple[int, int, int, Coordinate]] = [(h_start, h_start, entry_count, start)]
    nodes_expanded = 0
    found = False

    while open_pq:
        curr_f, curr_h, _, curr_node = heapq.heappop(open_pq)

        if curr_node in closed_set:
            continue

        closed_set.add(curr_node)
        nodes_expanded += 1

        if curr_node == goal:
            found = True
            break

        curr_g = g_score[curr_node]

        for neighbor in get_valid_neighbors(curr_node, grid):
            if neighbor in closed_set:
                continue

            tentative_g = curr_g + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                predecessors[neighbor] = curr_node
                h_val = manhattan_distance(neighbor, goal)
                f_val = tentative_g + h_val
                entry_count += 1
                heapq.heappush(open_pq, (f_val, h_val, entry_count, neighbor))

    t_end = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    execution_time_ms = (t_end - t_start) * 1000.0
    peak_memory_kb = peak_mem / 1024.0

    if not found:
        return PathfindingResult(
            algorithm="A* Search",
            success=False,
            path=[],
            cost=0,
            nodes_expanded=nodes_expanded,
            execution_time_ms=execution_time_ms,
            peak_memory_kb=peak_memory_kb,
            visited_cells=closed_set,
            message="No route exists: Goal is disconnected from Start by obstacles."
        )

    path = reconstruct_path(predecessors, goal)
    return PathfindingResult(
        algorithm="A* Search",
        success=True,
        path=path,
        cost=len(path) - 1,
        nodes_expanded=nodes_expanded,
        execution_time_ms=execution_time_ms,
        peak_memory_kb=peak_memory_kb,
        visited_cells=closed_set,
        message=f"Optimal shortest path found in {len(path) - 1} steps."
    )


def compare_algorithms(grid: List[List[bool]], start: Coordinate, goal: Coordinate) -> Tuple[PathfindingResult, PathfindingResult]:
    """
    Runs both Dijkstra's Algorithm and A* Search on the identical grid and points.
    
    Returns:
        (dijkstra_result, astar_result)
    """
    dijkstra_res = dijkstra(grid, start, goal)
    astar_res = astar(grid, start, goal)
    return dijkstra_res, astar_res
