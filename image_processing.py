"""
image_processing.py
===================
Image-to-grid conversion and route overlay rendering for floor plans and mazes.

Features:
  - Loads PNG, JPG, JPEG images.
  - Converts images to grayscale and thresholds them into binary walkable/blocked grids.
  - Adjustable threshold and selectable polarity (bright vs dark walkable).
  - Configurable cell sizing/downsampling to handle high-resolution architectural plans
    without memory explosion.
  - Precise bidirectional coordinate mapping:
      Canvas Display (pixels) <--> Original Image (pixels) <--> Grid Graph (row, col)
  - Visual route overlay with distinct Start, Goal, and Path markers.
  - Explanations of image-based classification limitations.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Coordinate = Tuple[int, int]  # (row, col)


class ImageGridProcessor:
    """
    Handles image loading, preprocessing, thresholding into a 2D discrete grid,
    and rendering solution routes back onto the original visual coordinate space.
    """

    LIMITATIONS_EXPLANATION = (
        "Limitations of Image-Based Grid Classification:\n"
        "1. Lighting & Vignetting: Real photographs often have shadows or uneven gradients "
        "causing the same physical floor surface to appear bright in the center and dark at the edges.\n"
        "2. Resolution & Thin Walls: Narrow walls or tight corridors can be blurred or broken "
        "if downsampled aggressively, leading to false disconnects or illusory passages.\n"
        "3. Blueprint Annotations: Text labels, room dimensions, hatchings, and door-swing arcs "
        "in architectural blueprints are treated as obstacles unless filtered.\n"
        "4. Compression Artifacts: JPEG compression creates 8x8 block artifacts and ringing noise "
        "along high-contrast edges, which can create false 1-pixel holes or blocks.\n"
        "5. Color Variations: Multi-colored floor finishes or textured rugs may cross the threshold, "
        "producing patchy walkable regions unless pre-filtered."
    )

    def __init__(self, image_path: Optional[str] = None):
        self.original_image: Optional[Image.Image] = None
        self.image_path: Optional[str] = None
        self.width: int = 0
        self.height: int = 0
        
        # Grid parameters
        self.cell_size: int = 1  # 1 means 1 pixel = 1 grid cell
        self.threshold: int = 100  # Default brightness threshold
        self.walkable_is_bright: bool = True
        self.grid: List[List[bool]] = []
        self.grid_rows: int = 0
        self.grid_cols: int = 0

        if image_path:
            self.load_image(image_path)

    def load_image(self, file_path: str) -> Image.Image:
        """
        Loads an image file (PNG, JPG, JPEG) and preserves the RGB color mode.
        """
        img = Image.open(file_path).convert("RGB")
        self.original_image = img
        self.image_path = file_path
        self.width, self.height = img.size
        return self.original_image

    def load_pil_image(self, pil_image: Image.Image) -> Image.Image:
        """
        Loads directly from an in-memory PIL Image object (e.g. for generated samples).
        """
        self.original_image = pil_image.convert("RGB")
        self.image_path = None
        self.width, self.height = self.original_image.size
        return self.original_image

    def compute_grid(
        self,
        threshold: int = 100,
        walkable_is_bright: bool = True,
        cell_size: int = 1
    ) -> List[List[bool]]:
        """
        Converts the loaded image into a 2D boolean grid:
          - True  => Walkable
          - False => Blocked (Wall/Obstacle)
          
        Args:
            threshold: Brightness cutoff (0-255). Default is 100.
                       - In bright mode (default): Pixels with brightness >= threshold are walkable.
                         Higher thresholds classify fewer pixels as walkable (stricter / narrower paths).
                         Lower thresholds classify more pixels as walkable (more permissive).
                       - In dark mode: Pixels with brightness < threshold are walkable.
            walkable_is_bright: If True, pixels with brightness >= threshold are walkable.
                                If False, pixels with brightness < threshold are walkable.
            cell_size: Downsampling factor. If > 1, downsamples by averaging or resizing.
                       e.g. cell_size=2 turns a 1000x800 image into a 500x400 grid.
        """
        if self.original_image is None:
            raise ValueError("No image loaded to compute grid.")

        self.threshold = int(threshold)
        self.walkable_is_bright = bool(walkable_is_bright)
        self.cell_size = max(1, int(cell_size))

        # Convert to grayscale
        gray_img = self.original_image.convert("L")

        # Downsample if cell_size > 1 using high-quality Box/Area resampling
        if self.cell_size > 1:
            target_cols = max(1, self.width // self.cell_size)
            target_rows = max(1, self.height // self.cell_size)
            gray_img = gray_img.resize((target_cols, target_rows), resample=Image.Resampling.BOX)

        # Convert to numpy array for fast thresholding
        arr = np.array(gray_img, dtype=np.uint8)
        self.grid_rows, self.grid_cols = arr.shape

        if self.walkable_is_bright:
            # Walkable when brightness >= threshold
            mask = arr >= self.threshold
        else:
            # Walkable when brightness < threshold (dark corridors)
            mask = arr < self.threshold

        # Convert boolean numpy array to Python list of lists for fast indexing in search
        self.grid = mask.tolist()
        return self.grid

    def image_to_grid_coord(self, img_x: float, img_y: float) -> Tuple[int, int]:
        """
        Maps an (x, y) coordinate from the original full-size image to (row, col) in the grid.
        """
        if self.grid_rows == 0 or self.grid_cols == 0:
            raise ValueError("Grid has not been computed yet.")

        # Account for cell_size scaling
        col = int(img_x // self.cell_size)
        row = int(img_y // self.cell_size)

        # Clamp within grid bounds
        col = max(0, min(self.grid_cols - 1, col))
        row = max(0, min(self.grid_rows - 1, row))
        return (row, col)

    def grid_to_image_coord(self, row: int, col: int) -> Tuple[int, int]:
        """
        Maps a grid (row, col) back to the center pixel (x, y) of that cell on the original image.
        """
        x = int(col * self.cell_size + self.cell_size // 2)
        y = int(row * self.cell_size + self.cell_size // 2)
        return (x, y)

    def is_cell_walkable(self, row: int, col: int) -> bool:
        """
        Checks if the cell at (row, col) is walkable.
        """
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            return bool(self.grid[row][col])
        return False

    def get_cell_status_description(self, row: int, col: int) -> Tuple[bool, str]:
        """
        Returns (is_walkable, detailed_message) for user feedback upon clicking.
        """
        if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
            return False, f"Selected point ({col}, {row}) is outside grid boundaries."
        
        walkable = bool(self.grid[row][col])
        if walkable:
            return True, f"Valid walkable cell at grid row={row}, col={col}."
        else:
            condition = (
                f"brightness < {self.threshold}" if self.walkable_is_bright 
                else f"brightness >= {self.threshold}"
            )
            return False, (
                f"Blocked cell at grid row={row}, col={col} (wall/obstacle detected; {condition}). "
                "Please click on an open corridor or adjust threshold settings."
            )

    def render_route_overlay(
        self,
        path: Optional[List[Coordinate]] = None,
        start: Optional[Coordinate] = None,
        goal: Optional[Coordinate] = None,
        visited_cells: Optional[set[Coordinate]] = None,
        path_color: Tuple[int, int, int] = (0, 220, 255),  # Electric Cyan
        visited_color: Optional[Tuple[int, int, int]] = (100, 180, 255), # Faint blue
        line_width: int = 4
    ) -> Image.Image:
        """
        Draws the computed route, Start, and Goal markers onto a copy of the original image.
        
        Returns:
            A new PIL Image with the overlaid route and markers.
        """
        if self.original_image is None:
            raise ValueError("No base image to draw overlay on.")

        overlay_img = self.original_image.copy()
        draw = ImageDraw.Draw(overlay_img, "RGBA")

        # 1. Optionally draw visited/expanded cells as translucent dots/squares
        if visited_cells:
            vis_rgba = (visited_color[0], visited_color[1], visited_color[2], 45)
            # Create a separate transparent layer for smooth blending
            tint_layer = Image.new("RGBA", overlay_img.size, (0, 0, 0, 0))
            tint_draw = ImageDraw.Draw(tint_layer)
            
            cs = self.cell_size
            for r, c in visited_cells:
                x0 = c * cs
                y0 = r * cs
                x1 = x0 + cs
                y1 = y0 + cs
                tint_draw.rectangle([x0, y0, x1, y1], fill=vis_rgba)
                
            overlay_img = Image.alpha_composite(overlay_img.convert("RGBA"), tint_layer).convert("RGB")
            draw = ImageDraw.Draw(overlay_img)

        # 2. Draw the path line
        if path and len(path) > 1:
            pixel_coords = [self.grid_to_image_coord(r, c) for r, c in path]
            # Draw route line
            eff_width = max(2, min(line_width, self.cell_size * 2))
            draw.line(pixel_coords, fill=path_color, width=eff_width, joint="curve")
            
            # Draw subtle dots on path nodes
            radius = max(1, eff_width // 2)
            for x, y in pixel_coords:
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=path_color)

        # Helper to draw a prominent marker with concentric ring
        def draw_marker(coord: Coordinate, fill_color: Tuple[int, int, int], label: str):
            x, y = self.grid_to_image_coord(coord[0], coord[1])
            # Scale marker size reasonably relative to cell_size or minimum 8px
            r = max(7, int(self.cell_size * 1.5))
            
            # Outer dark shadow ring for high contrast
            draw.ellipse([x - (r + 3), y - (r + 3), x + (r + 3), y + (r + 3)], fill=(20, 20, 20))
            # Primary colored circle
            draw.ellipse([x - r, y - r, x + r, y + r], fill=fill_color)
            # Inner white accent ring
            inner_r = max(3, r - 3)
            draw.ellipse([x - inner_r, y - inner_r, x + inner_r, y + inner_r], outline=(255, 255, 255), width=2)

        # 3. Draw Start Marker (Bright Lime Green)
        if start is not None:
            draw_marker(start, fill_color=(0, 230, 70), label="S")

        # 4. Draw Goal Marker (Vivid Crimson / Orange-Red)
        if goal is not None:
            draw_marker(goal, fill_color=(255, 45, 45), label="G")

        return overlay_img


def generate_sample_maze(
    width: int = 400,
    height: int = 400,
    cell_dimension: int = 20,
    wall_thickness: int = 4
) -> Image.Image:
    """
    Generates a crisp, aesthetically pleasing solvable 2D maze using Randomized Depth-First Search.
    
    Walls are colored solid black (0, 0, 0) and walkable corridors are solid white (255, 255, 255).
    Guarantees a clear open corridor from top-left entrance to bottom-right exit.
    """
    import random

    maze_cols = width // cell_dimension
    maze_rows = height // cell_dimension

    # Grid of cells: each cell has walls [Top, Right, Bottom, Left]
    # True means wall exists, False means wall has been carved open
    visited = [[False for _ in range(maze_cols)] for _ in range(maze_rows)]
    walls = [[[True, True, True, True] for _ in range(maze_cols)] for _ in range(maze_rows)]

    # Depth-first search maze generator with backtracking stack
    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        cr, cc = stack[-1]
        neighbors = []
        # Directions: (dr, dc, wall_index_curr, wall_index_neighbor)
        dirs = [
            (-1, 0, 0, 2),  # Up: current's Top, neighbor's Bottom
            (0, 1, 1, 3),   # Right: current's Right, neighbor's Left
            (1, 0, 2, 0),   # Down: current's Bottom, neighbor's Top
            (0, -1, 3, 1),  # Left: current's Left, neighbor's Right
        ]
        
        for dr, dc, w_curr, w_next in dirs:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < maze_rows and 0 <= nc < maze_cols and not visited[nr][nc]:
                neighbors.append((nr, nc, w_curr, w_next))

        if neighbors:
            # Pick a random unvisited neighbor
            nr, nc, w_curr, w_next = random.choice(neighbors)
            walls[cr][cc][w_curr] = False
            walls[nr][nc][w_next] = False
            visited[nr][nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    # Create PIL Image: corridors white, walls black
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    half_w = wall_thickness // 2

    # Draw outer border
    draw.rectangle([0, 0, width - 1, height - 1], outline=(0, 0, 0), width=wall_thickness)

    # Draw cell walls
    for r in range(maze_rows):
        for c in range(maze_cols):
            x1 = c * cell_dimension
            y1 = r * cell_dimension
            x2 = x1 + cell_dimension
            y2 = y1 + cell_dimension

            # Top wall
            if walls[r][c][0] and r > 0:
                draw.line([(x1, y1), (x2, y1)], fill=(0, 0, 0), width=wall_thickness)
            # Right wall
            if walls[r][c][1] and c < maze_cols - 1:
                draw.line([(x2, y1), (x2, y2)], fill=(0, 0, 0), width=wall_thickness)
            # Bottom wall
            if walls[r][c][2] and r < maze_rows - 1:
                draw.line([(x1, y2), (x2, y2)], fill=(0, 0, 0), width=wall_thickness)
            # Left wall
            if walls[r][c][3] and c > 0:
                draw.line([(x1, y1), (x1, y2)], fill=(0, 0, 0), width=wall_thickness)

    # Open entrance at top-left and exit at bottom-right
    # Entrance
    draw.line([(0, 0), (0, cell_dimension)], fill=(255, 255, 255), width=wall_thickness + 2)
    # Exit
    draw.line([(width - 1, height - cell_dimension), (width - 1, height - 1)], fill=(255, 255, 255), width=wall_thickness + 2)

    return img
