"""Procedural Tilt Maze builder for Blender.

This script constructs multiple tilt maze levels that respect the
physical measurements requested for the game prototype.  Each level is a
perfect maze (there is exactly one route between any two cells) with
carefully dimensioned walls, corridors, start/end markers and a metal
ball positioned at the entrance.

Usage
-----
Open the Blender scripting workspace, create a new text block and paste
the contents of this module.  Run the script to generate the collections
`Level 1`, `Level 2` and `Level 3` inside the current Blender scene.

All dimensions are expressed in meters.
"""

from __future__ import annotations

import bpy
import bmesh
import random
from dataclasses import dataclass
from mathutils import Vector
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Dimensional constants derived from the specification
# ---------------------------------------------------------------------------

BASE_SIZE = 0.4  # square base: 0.4 m x 0.4 m
BASE_THICKNESS = 0.02

OUTER_WALL_THICKNESS = 0.014
OUTER_WALL_HEIGHT = 0.02

BALL_DIAMETER = 0.025

# Default corridor and wall measurements (slightly adjustable per layout)
TARGET_CORRIDOR_WIDTH = 0.03
INNER_WALL_THICKNESS = 0.012
INNER_WALL_HEIGHT = 0.018


# ---------------------------------------------------------------------------
# Data containers and helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LevelSpec:
    name: str
    cell_count_x: int
    cell_count_y: int
    random_seed: int
    offset: Vector


Direction = Tuple[int, int]
CELL_DIRECTIONS: Dict[str, Direction] = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}


def ensure_collection(name: str) -> bpy.types.Collection:
    """Return a clean collection ready to host generated objects."""

    if collection := bpy.data.collections.get(name):
        # Remove existing objects so the scene stays consistent when the
        # script is executed multiple times.
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def get_or_create_material(
    name: str,
    *,
    base_color: Tuple[float, float, float, float],
    metallic: float,
    roughness: float,
    specular: float = 0.5,
) -> bpy.types.Material:
    """Create or update a Principled BSDF material with the given style."""

    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is None:
        principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs[0].default_value = base_color
    principled.inputs[4].default_value = metallic
    principled.inputs[7].default_value = roughness
    principled.inputs[2].default_value = specular
    return material


def link_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def create_cube(
    name: str,
    size: Vector,
    location: Vector,
    collection: bpy.types.Collection,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Create a cube primitive scaled to the requested size."""

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()

    obj.scale = Vector((size.x / 2.0, size.y / 2.0, size.z / 2.0))
    obj.location = location
    link_material(obj, material)
    return obj


def create_cylinder(
    name: str,
    radius: float,
    height: float,
    location: Vector,
    collection: bpy.types.Collection,
    material: bpy.types.Material,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=48,
        radius1=radius,
        radius2=radius,
        depth=height,
    )
    bm.to_mesh(mesh)
    bm.free()

    obj.location = location
    link_material(obj, material)
    return obj


def create_uv_sphere(
    name: str,
    radius: float,
    location: Vector,
    collection: bpy.types.Collection,
    material: bpy.types.Material,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, diameter=2.0)
    bm.to_mesh(mesh)
    bm.free()

    obj.scale = Vector((radius, radius, radius))
    obj.location = location
    link_material(obj, material)
    return obj


# ---------------------------------------------------------------------------
# Maze generation logic
# ---------------------------------------------------------------------------

def generate_perfect_maze(width: int, height: int, seed: int) -> List[List[Dict[str, bool]]]:
    """Return a perfect maze grid using recursive backtracking.

    Each cell contains four boolean entries (N, S, E, W) indicating if the
    corresponding wall is present (True) or carved (False).
    """

    grid: List[List[Dict[str, bool]]] = [
        [dict.fromkeys(CELL_DIRECTIONS.keys(), True) for _ in range(height)]
        for _ in range(width)
    ]
    visited = [[False for _ in range(height)] for _ in range(width)]

    random.seed(seed)

    def carve(x: int, y: int) -> None:
        visited[x][y] = True
        directions = list(CELL_DIRECTIONS.items())
        random.shuffle(directions)
        for label, (dx, dy) in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny]:
                # Carve passage in both directions.
                grid[x][y][label] = False
                opposite = {
                    "N": "S",
                    "S": "N",
                    "E": "W",
                    "W": "E",
                }[label]
                grid[nx][ny][opposite] = False
                carve(nx, ny)

    carve(0, 0)
    return grid


# ---------------------------------------------------------------------------
# Level construction helpers
# ---------------------------------------------------------------------------

def compute_corridor_widths(
    cells_x: int, cells_y: int
) -> Tuple[float, float, float]:
    """Calculate the corridor width and useful offsets within the base."""

    inner_span = BASE_SIZE - 2 * OUTER_WALL_THICKNESS
    corridor_x = (
        inner_span - INNER_WALL_THICKNESS * max(cells_x - 1, 0)
    ) / cells_x
    corridor_y = (
        inner_span - INNER_WALL_THICKNESS * max(cells_y - 1, 0)
    ) / cells_y

    # Keep a uniform corridor width while honouring the requested ~0.03 m.
    corridor = min(corridor_x, corridor_y)
    corridor = max(min(corridor, TARGET_CORRIDOR_WIDTH + 0.004), TARGET_CORRIDOR_WIDTH - 0.004)

    if corridor <= 0:
        raise ValueError("Corridor width calculation resulted in a non-positive value.")

    usable_x = corridor * cells_x + INNER_WALL_THICKNESS * max(cells_x - 1, 0)
    usable_y = corridor * cells_y + INNER_WALL_THICKNESS * max(cells_y - 1, 0)

    margin_x = (inner_span - usable_x) / 2.0
    margin_y = (inner_span - usable_y) / 2.0
    return corridor, margin_x, margin_y


def build_outer_structure(
    collection: bpy.types.Collection,
    base_material: bpy.types.Material,
    wall_material: bpy.types.Material,
    level_origin: Vector,
) -> None:
    """Create the base plate and surrounding outer walls."""

    base_location = level_origin + Vector((0.0, 0.0, BASE_THICKNESS / 2.0))
    create_cube(
        "Base",
        Vector((BASE_SIZE, BASE_SIZE, BASE_THICKNESS)),
        base_location,
        collection,
        base_material,
    )

    wall_z = BASE_THICKNESS + OUTER_WALL_HEIGHT / 2.0
    # North & South walls (parallel to X axis)
    for sign, suffix in ((1.0, "North"), (-1.0, "South")):
        location = level_origin + Vector((0.0, sign * (BASE_SIZE - OUTER_WALL_THICKNESS) / 2.0, wall_z))
        create_cube(
            f"OuterWall_{suffix}",
            Vector((BASE_SIZE, OUTER_WALL_THICKNESS, OUTER_WALL_HEIGHT)),
            location,
            collection,
            wall_material,
        )

    # East & West walls (parallel to Y axis)
    for sign, suffix in ((1.0, "East"), (-1.0, "West")):
        location = level_origin + Vector((sign * (BASE_SIZE - OUTER_WALL_THICKNESS) / 2.0, 0.0, wall_z))
        create_cube(
            f"OuterWall_{suffix}",
            Vector((OUTER_WALL_THICKNESS, BASE_SIZE - 2 * OUTER_WALL_THICKNESS, OUTER_WALL_HEIGHT)),
            location,
            collection,
            wall_material,
        )


def create_inner_walls(
    collection: bpy.types.Collection,
    maze_grid: Sequence[Sequence[Dict[str, bool]]],
    corridor_width: float,
    margin_x: float,
    margin_y: float,
    level_origin: Vector,
    wall_material: bpy.types.Material,
    name_prefix: str,
) -> None:
    """Create the inner wall network derived from the maze grid."""

    cells_x = len(maze_grid)
    cells_y = len(maze_grid[0])

    x_min = level_origin.x - BASE_SIZE / 2.0 + OUTER_WALL_THICKNESS + margin_x
    y_min = level_origin.y - BASE_SIZE / 2.0 + OUTER_WALL_THICKNESS + margin_y

    wall_z = BASE_THICKNESS + INNER_WALL_HEIGHT / 2.0
    pitch = corridor_width + INNER_WALL_THICKNESS

    # Vertical walls (between cells horizontally)
    for x in range(cells_x - 1):
        for y in range(cells_y):
            if maze_grid[x][y]["E"]:
                wall_center = Vector(
                    (
                        x_min + (x + 1) * corridor_width + x * INNER_WALL_THICKNESS + INNER_WALL_THICKNESS / 2.0,
                        y_min + y * pitch + corridor_width / 2.0,
                        wall_z,
                    )
                )
                create_cube(
                    f"{name_prefix}_Wall_V_{x}_{y}",
                    Vector((INNER_WALL_THICKNESS, corridor_width, INNER_WALL_HEIGHT)),
                    wall_center,
                    collection,
                    wall_material,
                )

    # Horizontal walls (between cells vertically)
    for x in range(cells_x):
        for y in range(cells_y - 1):
            if maze_grid[x][y]["N"]:
                wall_center = Vector(
                    (
                        x_min + x * pitch + corridor_width / 2.0,
                        y_min + (y + 1) * corridor_width + y * INNER_WALL_THICKNESS + INNER_WALL_THICKNESS / 2.0,
                        wall_z,
                    )
                )
                create_cube(
                    f"{name_prefix}_Wall_H_{x}_{y}",
                    Vector((corridor_width, INNER_WALL_THICKNESS, INNER_WALL_HEIGHT)),
                    wall_center,
                    collection,
                    wall_material,
                )


def place_markers_and_ball(
    collection: bpy.types.Collection,
    start_cell: Tuple[int, int],
    end_cell: Tuple[int, int],
    corridor_width: float,
    margin_x: float,
    margin_y: float,
    level_origin: Vector,
    start_material: bpy.types.Material,
    finish_material: bpy.types.Material,
    ball_material: bpy.types.Material,
    name_prefix: str,
) -> None:
    """Add the spherical ball and coloured start/end markers."""

    pitch = corridor_width + INNER_WALL_THICKNESS
    x_min = level_origin.x - BASE_SIZE / 2.0 + OUTER_WALL_THICKNESS + margin_x
    y_min = level_origin.y - BASE_SIZE / 2.0 + OUTER_WALL_THICKNESS + margin_y

    def cell_center(index: Tuple[int, int]) -> Vector:
        cx, cy = index
        return Vector(
            (
                x_min + cx * pitch + corridor_width / 2.0,
                y_min + cy * pitch + corridor_width / 2.0,
                BASE_THICKNESS,
            )
        )

    start_center = cell_center(start_cell)
    finish_center = cell_center(end_cell)

    marker_height = 0.004
    marker_radius = corridor_width * 0.35

    create_cylinder(
        f"{name_prefix}_StartMarker",
        radius=marker_radius,
        height=marker_height,
        location=start_center + Vector((0.0, 0.0, marker_height / 2.0 + 1e-4)),
        collection=collection,
        material=start_material,
    )

    create_cylinder(
        f"{name_prefix}_FinishMarker",
        radius=marker_radius,
        height=marker_height,
        location=finish_center + Vector((0.0, 0.0, marker_height / 2.0 + 1e-4)),
        collection=collection,
        material=finish_material,
    )

    ball_radius = BALL_DIAMETER / 2.0
    create_uv_sphere(
        f"{name_prefix}_PlayerBall",
        radius=ball_radius,
        location=start_center + Vector((0.0, 0.0, ball_radius + marker_height + 2e-4)),
        collection=collection,
        material=ball_material,
    )


def construct_level(level: LevelSpec) -> None:
    collection = ensure_collection(level.name)

    wood_material = get_or_create_material(
        "Maze_Wood",
        base_color=(0.550, 0.320, 0.130, 1.0),
        metallic=0.0,
        roughness=0.45,
    )
    darker_wood = get_or_create_material(
        "Maze_Base",
        base_color=(0.360, 0.210, 0.090, 1.0),
        metallic=0.0,
        roughness=0.55,
    )
    metal_material = get_or_create_material(
        "Maze_BallMetal",
        base_color=(0.660, 0.660, 0.680, 1.0),
        metallic=0.92,
        roughness=0.18,
        specular=0.65,
    )
    start_material = get_or_create_material(
        "Maze_Start",
        base_color=(0.090, 0.330, 0.880, 1.0),
        metallic=0.1,
        roughness=0.25,
    )
    finish_material = get_or_create_material(
        "Maze_Finish",
        base_color=(0.930, 0.270, 0.520, 1.0),
        metallic=0.05,
        roughness=0.35,
    )

    corridor_width, margin_x, margin_y = compute_corridor_widths(
        level.cell_count_x, level.cell_count_y
    )

    build_outer_structure(collection, darker_wood, wood_material, level.offset)

    maze = generate_perfect_maze(level.cell_count_x, level.cell_count_y, level.random_seed)

    create_inner_walls(
        collection,
        maze,
        corridor_width,
        margin_x,
        margin_y,
        level.offset,
        wood_material,
        level.name.replace(" ", "_"),
    )

    start_cell = (0, 0)
    end_cell = (level.cell_count_x - 1, level.cell_count_y - 1)

    place_markers_and_ball(
        collection,
        start_cell,
        end_cell,
        corridor_width,
        margin_x,
        margin_y,
        level.offset,
        start_material,
        finish_material,
        metal_material,
        level.name.replace(" ", "_"),
    )


def main() -> None:
    # Prepare three difficulty tiers by varying cell counts and seeds.  The
    # offsets keep the mazes spatially separated inside the scene.
    level_specs = [
        LevelSpec("Level 1", cell_count_x=5, cell_count_y=5, random_seed=21, offset=Vector((-0.5, 0.0, 0.0))),
        LevelSpec("Level 2", cell_count_x=7, cell_count_y=6, random_seed=87, offset=Vector((0.0, 0.0, 0.0))),
        LevelSpec("Level 3", cell_count_x=9, cell_count_y=8, random_seed=133, offset=Vector((0.5, 0.0, 0.0))),
    ]

    for level in level_specs:
        construct_level(level)


if __name__ == "__main__":
    main()

