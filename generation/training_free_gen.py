"""
Unified batch inpainting script. Select one or several canvas layouts via --canvas.

Usage:
    python generation/training_free_gen.py --list
    python generation/training_free_gen.py --config configs/common_diffusion_base.yaml    --codes-json models/semantic_ae/common_ae_base/vqvae_codes_99_coverage.json --canvas roundabout
    python generation/training_free_gen.py --config configs/common_diffusion_base.yaml \\
        --codes-json PATH/TO/vqvae_codes_99_coverage.json --canvas s_road u_road cross_road
    python generation/training_free_gen.py --config configs/common_diffusion_base.yaml \\
        --codes-json PATH/TO/vqvae_codes_99_coverage.json --canvas all

Required:
    --config PATH           Path to the diffusion YAML config (used to build the sampling model)
    --codes-json PATH       Path to vqvae_codes_99_coverage.json (from analyze_vqvae_codes.py)

Optional overrides:
    --n_samples 20          Number of inpainting iterations per layout (default: 20)
    --repaint_steps 60      Override repaint steps for sample_fn (default: per-layout registry)
    --signature-code-pick recommended|volume  How to pick one code per class from JSON

VQ-VAE conditioning codes are read **only** from the JSON path you pass; there is no auto-discovery.
Use ``--list`` without ``--codes-json`` to print layout names only.
"""

import os
# Configure for headless rendering - MUST be set before importing VTK
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.pop("DISPLAY", None)

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import sys
import vtk
from vtk.util import numpy_support
import trimesh
import gc
import torch.nn.functional as F

sys.path.append(os.getcwd())

from diffusion.triplane_util import build_sampling_model, compose_featmaps, decompose_featmaps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default train_id -> label_id mapping for bev_to_rgb visualisation
TRAIN_TO_LABEL_DEFAULT = {
    1: 10, 9: 40, 6: 30, 18: 80, 15: 70, 16: 71, 17: 72,
}

# Mapping for layouts that use bicycle (2) and building (13) instead
TRAIN_TO_LABEL_BUILDING_BICYCLE = {
    2: 11, 9: 40, 13: 50, 18: 80,
}

OFFICIAL_RGB = {
    0: [255, 255, 255], 10: [124, 152, 187], 11: [69, 219, 252], 13: [250, 80, 100],
    15: [0, 0, 128],    16: [255, 0, 0],     18: [180, 30, 80],  20: [32, 71, 193],
    30: [255, 0, 0],    31: [240, 5, 208],   32: [90, 30, 150],  40: [135, 0, 135],
    44: [255, 182, 193],48: [75, 0, 75],     49: [103, 16, 55],  50: [181, 171, 6],
    51: [255, 131, 0],  52: [0, 150, 255],   60: [170, 255, 150],70: [0, 157, 0],
    71: [144, 70, 38],  72: [216, 254, 135], 80: [250, 250, 172],81: [250, 0, 0],
    99: [255, 255, 50],
}

# VQ-VAE train_id → codebook index. Filled from JSON only; all layouts share this dict.
SIGNATURE_CODES = {}

# Train ids that appear on at least one registered canvas (JSON must cover each).
_SIGNATURE_TRAIN_IDS = (0, 1, 2, 6, 9, 13, 15, 16, 18)

# SemanticKITTI learning labels (ids 0–19); JSON keys = class name UPPER + EMPTY_SPACE for 0.
_SEMKITTI_LEARNING_CLASS_NAMES = (
    "unlabeled",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
)


def _primary_vq_code_from_coverage_json(
    data: dict, train_id: int, pick: str = "recommended",
) -> int:
    """
    pick='recommended': use ``recommended_code`` from analyze_vqvae_codes.py when present
    (highest purity among codes in the ~99% coverage set); else ``codes[0]``.
    pick='volume': always ``codes[0]`` (largest class volume).
    """
    key = "EMPTY_SPACE" if train_id == 0 else _SEMKITTI_LEARNING_CLASS_NAMES[train_id].upper()
    if key not in data:
        raise KeyError(f"VQ-VAE codes JSON missing key '{key}' (train_id={train_id})")
    entry = data[key]
    codes = entry.get("codes") or []
    if not codes:
        raise ValueError(f"VQ-VAE codes JSON has empty 'codes' for '{key}'")
    if pick == "volume":
        return int(codes[0])
    rec = entry.get("recommended_code")
    if rec is not None:
        return int(rec)
    return int(codes[0])


def load_signature_codes_from_json(json_path: str, pick: str = "recommended") -> dict:
    """train_id → code for every id in ``_SIGNATURE_TRAIN_IDS``."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return {
        tid: _primary_vq_code_from_coverage_json(data, tid, pick=pick)
        for tid in _SIGNATURE_TRAIN_IDS
    }


def apply_signature_codes_from_json(json_path: str, pick: str = "recommended") -> None:
    """Fill ``SIGNATURE_CODES`` from JSON (CANVAS_REGISTRY entries reference this dict)."""
    loaded = load_signature_codes_from_json(json_path, pick=pick)
    SIGNATURE_CODES.clear()
    SIGNATURE_CODES.update(loaded)
    print(
        f"VQ-VAE signature codes (pick={pick!r}, train_id → code):",
        dict(SIGNATURE_CODES),
    )


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def bev_to_rgb(bev_indices, official_rgb, train_to_label=None):
    if train_to_label is None:
        train_to_label = TRAIN_TO_LABEL_DEFAULT
    h, w = bev_indices.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    for train_id, label_id in train_to_label.items():
        if label_id in official_rgb:
            rgb[bev_indices == train_id] = official_rgb[label_id]
    return rgb


def build_voxel_mesh(occupied_voxels, colors, voxel_size=1.0):
    unit_voxel = trimesh.creation.box(extents=(voxel_size, voxel_size, voxel_size))
    base_vertices = unit_voxel.vertices
    base_faces = unit_voxel.faces
    num_voxels = occupied_voxels.shape[0]
    all_vertices = np.concatenate([base_vertices + offset for offset in occupied_voxels], axis=0)
    face_offsets = np.arange(num_voxels) * base_vertices.shape[0]
    all_faces = np.concatenate([base_faces + offset for offset in face_offsets], axis=0)
    all_colors = np.concatenate([
        np.tile(np.append(color, 255), (base_vertices.shape[0], 1)) for color in colors
    ], axis=0)
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    mesh.visual.vertex_colors = all_colors
    return mesh


def render_with_vtk(combined_mesh, type='semcity', view='up',
                    original_shape=(256, 256, 32), voxel_map=None, occupied_voxels=None):
    if type == 'semkitti':
        width, height = 1241, 376
    else:
        width, height = 2048, 1440

    vertices = combined_mesh.vertices.astype(np.float32)
    faces = combined_mesh.faces.astype(np.int32)
    colors = combined_mesh.visual.vertex_colors[:, :3]

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(vertices))
    vtk_cells = vtk.vtkCellArray()
    for face in faces:
        vtk_cells.InsertNextCell(3)
        for vertex_id in face:
            vtk_cells.InsertCellPoint(vertex_id)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)

    colors_normalized = (colors.astype(np.float32) / 255.0
                         if colors.max() > 1.0 else colors.astype(np.float32))
    vtk_colors = numpy_support.numpy_to_vtk(colors_normalized)
    vtk_colors.SetName("Colors")
    polydata.GetPointData().SetScalars(vtk_colors)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarModeToUsePointData()
    mapper.SetColorModeToDirectScalars()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetInterpolationToFlat()
    prop.SetSpecular(0.0)
    prop.SetSpecularPower(1.0)
    prop.SetDiffuse(0.3)
    prop.SetAmbient(0.9)
    prop.SetRepresentationToSurface()

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.5, 0.5, 0.5)
    renderer.AddActor(actor)
    renderer.SetUseShadows(0)

    if type == 'semcity':
        camera_location = np.array([20.0, 128.0, 10.0])
    else:
        camera_location = np.array([45.0, 128.0, 8.65])

    if view == 'up':
        camera_location = np.array([128.0, 128.0, 140.0])
        look_at_point = np.array([128.0, 128.0, 16.0])
        up_vector = np.array([1.0, 0.0, 0.0])
    else:
        look_at_point = camera_location + np.array([1.0, 0.0, 0.0])
        up_vector = np.array([0.0, 0.0, 1.0])

    camera = renderer.GetActiveCamera()
    camera.SetPosition(camera_location)
    camera.SetFocalPoint(look_at_point)
    camera.SetViewUp(up_vector)
    if type == 'semkitti':
        camera.SetViewAngle(np.degrees(np.pi / 2) * 2.571)
    else:
        camera.SetViewAngle(np.degrees(np.pi / 2))

    renderer.SetAmbient(0.5, 0.5, 0.5)
    light_direction = np.array([np.cos(np.pi / 2), 0, np.sin(np.pi / 2)])
    light_position = camera_location + light_direction * 1000
    light = vtk.vtkLight()
    light.SetPosition(*light_position)
    light.SetFocalPoint(*look_at_point)
    light.SetIntensity(1.0)
    light.SetColor(1.0, 1.0, 1.0)
    renderer.AddLight(light)
    renderer.SetAutomaticLightCreation(0)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(width, height)
    render_window.AddRenderer(renderer)
    render_window.Render()

    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetInputBufferTypeToRGBA()
    window_to_image.Update()
    vtk_image = window_to_image.GetOutput()
    vtk_array = vtk_image.GetPointData().GetScalars()
    arr = numpy_support.vtk_to_numpy(vtk_array)
    arr = arr.reshape((height, width, vtk_array.GetNumberOfComponents()))
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)

    renderer.RemoveAllLights()
    renderer.RemoveAllViewProps()
    render_window.RemoveRenderer(renderer)
    try:
        render_window.Finalize()
    except Exception:
        pass
    del window_to_image, vtk_image, vtk_array, render_window, renderer, light
    del actor, mapper, polydata, vtk_colors, vtk_cells, vtk_points
    gc.collect()
    return arr


# ---------------------------------------------------------------------------
# Canvas builders – original layouts
# ---------------------------------------------------------------------------

def fill_roundabout_roads(canvas, cx=64, cy=64, R_inner=22, R_outer=32, road_w=20):
    """Fill road (train_id 9): ring + four arms to canvas edges (clean_Semcity helper)."""
    H, W = canvas.shape
    yy, xx = np.ogrid[0:H, 0:W]
    dist = np.sqrt((yy.astype(np.float64) - cy) ** 2 + (xx.astype(np.float64) - cx) ** 2)
    ring = (dist >= R_inner) & (dist <= R_outer)
    half_w = road_w // 2
    rs, re = cy - R_outer, cy + R_outer
    cs, ce = cx - R_outer, cx + R_outer
    north = (xx >= cx - half_w) & (xx < cx + half_w) & (yy < rs)
    south = (xx >= cx - half_w) & (xx < cx + half_w) & (yy >= re)
    west = (yy >= cy - half_w) & (yy < cy + half_w) & (xx < cs)
    east = (yy >= cy - half_w) & (yy < cy + half_w) & (xx >= ce)
    road = ring | north | south | west | east
    canvas[road] = 9
    return dist, R_inner, R_outer


def build_canvas_roundabout(size=128):
    """Petit rond-point (anneau + 4 bras), îlot végétation, bloc SE, voiture sur le bras nord.

    Same geometry as ``clean_Semcity/codes_analysis/batch_inpainting_roundabout_trunks_vegetation_person.py``.
    Intended for ``size == 128`` (fixed pixel coords for the peripheral vegetation patch).
    """
    canvas = np.zeros((size, size), dtype=np.int32)
    cx = cy = size // 2
    R_inner, R_outer = 12, 20
    road_w = 16
    dist, _, _ = fill_roundabout_roads(
        canvas, cx=cx, cy=cy, R_inner=R_inner, R_outer=R_outer, road_w=road_w
    )

    canvas[dist < R_inner] = 15

    yy, xx = np.ogrid[0:size, 0:size]
    dist2 = np.sqrt((yy.astype(np.float64) - cy) ** 2 + (xx.astype(np.float64) - cx) ** 2)
    ring_only = (dist2 >= R_inner) & (dist2 <= R_outer)
    canvas[ring_only] = 9

    vegetation_h, vegetation_w = 22, 18
    veg_x_start = 106
    veg_y_start = 100
    canvas[
        veg_y_start : veg_y_start + vegetation_h,
        veg_x_start : veg_x_start + vegetation_w,
    ] = 15

    car_h, car_w = 12, 6
    rs = cy - R_outer
    row_car = (rs - car_h) // 2
    col_car = cx - car_w // 2
    if row_car < 0:
        row_car = 0
    canvas[row_car : row_car + car_h, col_car : col_car + car_w] = 1
    return canvas


def build_canvas_roundabout_vegetation_car(size=128):
    """Alias of ``build_canvas_roundabout`` (same layout)."""
    return build_canvas_roundabout(size)


def build_canvas_road_trunks_vegetation_person(size=128):
    """Route verticale, troncs gauche, végétation droite, personne, voiture."""
    road_w = 20
    road_start = 64 - road_w // 2

    canvas = np.zeros((size, size), dtype=np.int32)
    canvas[:, road_start:road_start + road_w] = 9

    trunk_size = 4
    trunk_x = 40
    canvas[20:20 + trunk_size, trunk_x:trunk_x + trunk_size] = 16
    canvas[55:55 + trunk_size, trunk_x:trunk_x + trunk_size] = 16
    canvas[90:90 + trunk_size, trunk_x:trunk_x + trunk_size] = 16
    canvas[44:44 + 40, 78:78 + 20] = 15
    canvas[58:58 + 4, 54:54 + 4] = 6
    canvas[75:75 + 12, 60:60 + 6] = 1
    return canvas


def build_canvas_road_car(size=128):
    """Route verticale, voiture sur la route, bloc végétation à gauche."""
    road_w = 20
    road_start = 64 - road_w // 2

    canvas = np.zeros((size, size), dtype=np.int32)
    canvas[:, road_start:road_start + road_w] = 9
    canvas[58:58 + 12, 58:58 + 6] = 1
    canvas[58:58 + 40, 40:40 + 10] = 15
    return canvas


def build_canvas_road_building_bicycle_poles(size=128):
    """Route verticale, bâtiment à gauche, vélo sur la route, poteaux à droite."""
    road_w = 20
    road_start = 64 - road_w // 2

    canvas = np.zeros((size, size), dtype=np.int32)
    canvas[:, road_start:road_start + road_w] = 9
    canvas[10:118, 12:52] = 13  # bâtiment

    pole_w, pole_h = 2, 5
    pole_x = road_start + road_w + 2
    for y0 in (24, 56, 88, 118):
        y0c = min(y0, size - pole_h)
        canvas[y0c:y0c + pole_h, pole_x:pole_x + pole_w] = 18

    canvas[62:62 + 5, 60:60 + 8] = 2  # vélo
    return canvas


# ---------------------------------------------------------------------------
# Canvas builders – various layouts
# ---------------------------------------------------------------------------

def make_s_road(road_w=24, veg_w=12):
    """Route en S avec végétation de chaque côté (amplitude 20 px)."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    half_w = road_w // 2
    for row in range(128):
        x_center = int(round(64 + 20 * np.sin(2 * np.pi * row / 128)))
        veg_l_start = max(0, x_center - half_w - veg_w)
        veg_l_end   = max(0, x_center - half_w)
        canvas[row, veg_l_start:veg_l_end] = 15
        veg_r_start = min(128, x_center + half_w)
        veg_r_end   = min(128, x_center + half_w + veg_w)
        canvas[row, veg_r_start:veg_r_end] = 15
        canvas[row, max(0, x_center - half_w):min(128, x_center + half_w)] = 9
    return canvas


def make_s_road_sharp(road_w=34, veg_w=12):
    """Route en S avec courbure très prononcée (amplitude 35 px)."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    half_w = road_w // 2
    for row in range(128):
        x_center = int(round(64 + 35 * np.sin(2 * np.pi * row / 128)))
        veg_l_start = max(0, x_center - half_w - veg_w)
        veg_l_end   = max(0, x_center - half_w)
        canvas[row, veg_l_start:veg_l_end] = 15
        veg_r_start = min(128, x_center + half_w)
        veg_r_end   = min(128, x_center + half_w + veg_w)
        canvas[row, veg_r_start:veg_r_end] = 15
        canvas[row, max(0, x_center - half_w):min(128, x_center + half_w)] = 9
    return canvas


def make_v_road(road_w=22, veg_w=10):
    """Route en U incliné : deux bras obliques reliés par un arc, végétation intérieure."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    half_w    = road_w // 2
    car_along = 6
    car_perp  = 3

    arm_dx  = 20.0
    arm_dy  = 82.0
    arm_len = np.sqrt(arm_dx ** 2 + arm_dy ** 2)

    left_top  = np.array([0.,          0.])
    left_bot  = np.array([arm_dx,      arm_dy])
    right_top = np.array([127.,        0.])
    right_bot = np.array([127. - arm_dx, arm_dy])

    R      = (right_bot[0] - left_bot[0]) * arm_len / (2.0 * arm_dy)
    arc_cx = left_bot[0] + R * arm_dy / arm_len
    arc_cy = left_bot[1] - R * arm_dx / arm_len

    theta_start = np.arctan2(left_bot[1] - arc_cy,  arc_cx - left_bot[0])
    theta_end   = np.pi - theta_start

    def paint_strip(cx, cy, nx_, ny_, inner_sign):
        veg_d = range(-(half_w + veg_w), -half_w) if inner_sign < 0 \
                else range(half_w, half_w + veg_w)
        for d in veg_d:
            px_ = int(round(cx + d * nx_))
            py_ = int(round(cy + d * ny_))
            if 0 <= px_ < 128 and 0 <= py_ < 128 and canvas[py_, px_] != 9:
                canvas[py_, px_] = 15
        for d in range(-half_w, half_w):
            px_ = int(round(cx + d * nx_))
            py_ = int(round(cy + d * ny_))
            if 0 <= px_ < 128 and 0 <= py_ < 128:
                canvas[py_, px_] = 9

    def paint_car(cx, cy, tx_, ty_, nx_, ny_):
        for a in np.arange(-car_along, car_along + 0.1, 0.4):
            for p in np.arange(-car_perp, car_perp + 0.1, 0.4):
                px_ = int(round(cx + a * tx_ + p * nx_))
                py_ = int(round(cy + a * ty_ + p * ny_))
                if 0 <= px_ < 128 and 0 <= py_ < 128:
                    canvas[py_, px_] = 1

    dx_L, dy_L = left_bot - left_top
    len_L = np.sqrt(dx_L ** 2 + dy_L ** 2)
    tx_L, ty_L = dx_L / len_L, dy_L / len_L
    nx_L, ny_L = -ty_L, tx_L
    for t in np.linspace(0, 1, 400):
        paint_strip(left_top[0] + t * dx_L, left_top[1] + t * dy_L, nx_L, ny_L, -1)
    t_car = 0.38
    paint_car(left_top[0] + t_car * dx_L, left_top[1] + t_car * dy_L, tx_L, ty_L, nx_L, ny_L)

    for theta in np.linspace(theta_start, theta_end, 350):
        cx_a = arc_cx - R * np.cos(theta)
        cy_a = arc_cy + R * np.sin(theta)
        tx_a, ty_a =  np.sin(theta),  np.cos(theta)
        nx_a, ny_a = -ty_a,           tx_a
        paint_strip(cx_a, cy_a, nx_a, ny_a, -1)

    dx_R, dy_R = right_bot - right_top
    len_R = np.sqrt(dx_R ** 2 + dy_R ** 2)
    tx_R, ty_R = dx_R / len_R, dy_R / len_R
    nx_R, ny_R = -ty_R, tx_R
    for t in np.linspace(0, 1, 400):
        paint_strip(right_top[0] + t * dx_R, right_top[1] + t * dy_R, nx_R, ny_R, +1)
    paint_car(right_top[0] + t_car * dx_R, right_top[1] + t_car * dy_R, tx_R, ty_R, nx_R, ny_R)

    return canvas


def make_two_parallel_roads_cars(road_w=16):
    """Deux routes parallèles horizontales, une voiture par route."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    y1_center, y2_center = 38, 90
    car_h, car_w = 6, 12
    for yc in (y1_center, y2_center):
        canvas[yc - road_w // 2: yc + road_w // 2, :] = 9
    cx1 = 32
    canvas[y1_center - car_h // 2: y1_center + car_h // 2,
           cx1 - car_w // 2: cx1 + car_w // 2] = 1
    cx2 = 90
    canvas[y2_center - car_h // 2: y2_center + car_h // 2,
           cx2 - car_w // 2: cx2 + car_w // 2] = 1
    return canvas


def make_two_vertical_parallel_roads_cars(road_w=16):
    """Deux routes parallèles verticales, une voiture par route."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    x1_center, x2_center = 38, 90
    car_h, car_w = 12, 6
    for xc in (x1_center, x2_center):
        canvas[:, xc - road_w // 2: xc + road_w // 2] = 9
    cy1 = 35
    canvas[cy1 - car_h // 2: cy1 + car_h // 2,
           x1_center - car_w // 2: x1_center + car_w // 2] = 1
    cy2 = 90
    canvas[cy2 - car_h // 2: cy2 + car_h // 2,
           x2_center - car_w // 2: x2_center + car_w // 2] = 1
    return canvas


def make_u_road(road_w=22, veg_w=10):
    """Route en U pur : deux bras verticaux reliés par un arc, végétation intérieure."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    half_w    = road_w // 2
    car_along = 6
    car_perp  = 3

    left_x     = 30.0
    right_x    = 97.0
    junction_y = 82.0
    arc_cx     = (left_x + right_x) / 2.0
    arc_R      = (right_x - left_x) / 2.0
    arc_cy     = junction_y

    def paint_strip(cx, cy, nx_, ny_, inner_sign):
        veg_d = range(-(half_w + veg_w), -half_w) if inner_sign < 0 \
                else range(half_w, half_w + veg_w)
        for d in veg_d:
            px_ = int(round(cx + d * nx_))
            py_ = int(round(cy + d * ny_))
            if 0 <= px_ < 128 and 0 <= py_ < 128 and canvas[py_, px_] != 9:
                canvas[py_, px_] = 15
        for d in range(-half_w, half_w):
            px_ = int(round(cx + d * nx_))
            py_ = int(round(cy + d * ny_))
            if 0 <= px_ < 128 and 0 <= py_ < 128:
                canvas[py_, px_] = 9

    def paint_car(cx, cy, tx_, ty_, nx_, ny_):
        for a in np.arange(-car_along, car_along + 0.1, 0.4):
            for p in np.arange(-car_perp, car_perp + 0.1, 0.4):
                px_ = int(round(cx + a * tx_ + p * nx_))
                py_ = int(round(cy + a * ty_ + p * ny_))
                if 0 <= px_ < 128 and 0 <= py_ < 128:
                    canvas[py_, px_] = 1

    tx_arm, ty_arm = 0.0, 1.0
    nx_arm, ny_arm = -1.0, 0.0

    for t in np.linspace(0, 1, 300):
        paint_strip(left_x, t * junction_y, nx_arm, ny_arm, -1)
    paint_car(left_x, 0.35 * junction_y, tx_arm, ty_arm, nx_arm, ny_arm)

    for theta in np.linspace(0, np.pi, 300):
        cx_a = arc_cx - arc_R * np.cos(theta)
        cy_a = arc_cy + arc_R * np.sin(theta)
        tx_a, ty_a =  np.sin(theta),  np.cos(theta)
        nx_a, ny_a = -ty_a,           tx_a
        paint_strip(cx_a, cy_a, nx_a, ny_a, -1)

    for t in np.linspace(0, 1, 300):
        paint_strip(right_x, t * junction_y, nx_arm, ny_arm, +1)
    paint_car(right_x, 0.35 * junction_y, tx_arm, ty_arm, nx_arm, ny_arm)

    return canvas


def make_three_parallel_roads_vegetation(road_w=14, veg_w=18):
    """3 routes parallèles horizontales séparées par des bandes de végétation."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    total_h = 3 * road_w + 2 * veg_w
    y0 = (128 - total_h) // 2
    r1_s = y0;       r1_e = r1_s + road_w
    v1_s = r1_e;     v1_e = v1_s + veg_w
    r2_s = v1_e;     r2_e = r2_s + road_w
    v2_s = r2_e;     v2_e = v2_s + veg_w
    r3_s = v2_e;     r3_e = r3_s + road_w
    canvas[r1_s:r1_e, :] = 9
    canvas[v1_s:v1_e, :] = 15
    canvas[r2_s:r2_e, :] = 9
    canvas[v2_s:v2_e, :] = 15
    canvas[r3_s:r3_e, :] = 9
    return canvas


def make_cross_road(road_w=16, veg_w=10):
    """Croisement en + : route horizontale + verticale, végétation dans les 4 coins."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    half_w = road_w // 2
    yy, xx = np.mgrid[0:128, 0:128]
    h_mask = (yy >= 64 - half_w) & (yy < 64 + half_w)
    v_mask = (xx >= 64 - half_w) & (xx < 64 + half_w)
    near   = (np.abs(yy - 64) < half_w + veg_w) & (np.abs(xx - 64) < half_w + veg_w)
    canvas[near & ~h_mask & ~v_mask] = 15
    canvas[h_mask | v_mask] = 9
    return canvas


def make_y_junction(road_w=18, veg_w=10):
    """Jonction en Y : bras vertical vers le haut + deux bras à 120° vers le bas."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    half_w = road_w // 2
    cx, cy = 64., 72.
    yy, xx = np.mgrid[0:128, 0:128].astype(float)

    def arm(adx, ady):
        lng = np.sqrt(adx ** 2 + ady ** 2)
        dx, dy = adx / lng, ady / lng
        nx, ny = -dy, dx
        perp  = np.abs((xx - cx) * nx + (yy - cy) * ny)
        proj  = (xx - cx) * dx + (yy - cy) * dy
        going = proj >= 0
        return (perp < half_w) & going, (perp >= half_w) & (perp < half_w + veg_w) & going

    r1, v1 = arm(0., -1.)
    r2, v2 = arm(-np.sin(np.radians(60)), np.cos(np.radians(60)))
    r3, v3 = arm( np.sin(np.radians(60)), np.cos(np.radians(60)))
    road = r1 | r2 | r3
    canvas[(v1 | v2 | v3) & ~road] = 15
    canvas[road] = 9
    return canvas


def make_diagonal_road(road_w=18, veg_w=10):
    """Route diagonale NW → SE, végétation des deux côtés."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    half_w = road_w // 2
    yy, xx = np.mgrid[0:128, 0:128].astype(float)
    dist   = np.abs(yy - xx) / np.sqrt(2)
    canvas[dist < half_w + veg_w] = 15
    canvas[dist < half_w] = 9
    return canvas


def make_zigzag_road(road_w=16, veg_w=10):
    """Route en zigzag (3 segments obliques alternés)."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    half_w = road_w // 2
    yy, xx = np.mgrid[0:128, 0:128].astype(float)
    pts    = [(0., 112.), (42., 16.), (85., 112.), (128., 16.)]
    road   = np.zeros((128, 128), dtype=bool)
    veg    = np.zeros((128, 128), dtype=bool)
    for i in range(len(pts) - 1):
        x0, y0 = pts[i];  x1, y1 = pts[i + 1]
        dx, dy  = x1 - x0, y1 - y0
        lng     = np.sqrt(dx ** 2 + dy ** 2)
        dx, dy  = dx / lng, dy / lng
        nx, ny  = -dy, dx
        perp    = np.abs((xx - x0) * nx + (yy - y0) * ny)
        proj    = (xx - x0) * dx + (yy - y0) * dy
        on_seg  = (proj >= 0) & (proj <= lng)
        road   |= (perp < half_w) & on_seg
        veg    |= (perp >= half_w) & (perp < half_w + veg_w) & on_seg
    canvas[veg & ~road] = 15
    canvas[road]        = 9
    return canvas


def make_roundabout_plus(road_w=16, veg_w=8):
    """Rond-point avec 4 routes d'accès (N, S, E, O) et île centrale végétalisée."""
    canvas = np.zeros((128, 128), dtype=np.int32)
    half_w = road_w // 2
    R_rb   = 26
    R_isle = R_rb - half_w - 2
    yy, xx = np.mgrid[0:128, 0:128].astype(float)
    dist   = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2)
    h_arm  = np.abs(yy - 64) < half_w
    v_arm  = np.abs(xx - 64) < half_w
    canvas[(np.abs(dist - R_rb) > half_w) & (np.abs(dist - R_rb) <= half_w + veg_w)] = 15
    canvas[dist <= R_isle] = 15
    canvas[(np.abs(yy - 64) >= half_w) & (np.abs(yy - 64) < half_w + veg_w) & h_arm] = 15
    canvas[(np.abs(xx - 64) >= half_w) & (np.abs(xx - 64) < half_w + veg_w) & v_arm] = 15
    canvas[np.abs(dist - R_rb) <= half_w] = 9
    canvas[h_arm | v_arm] = 9
    canvas[dist < R_rb - half_w] = 15
    return canvas


def _strip_veg(canvas):
    """Retourne une copie du canvas sans végétation (label 15 → 0)."""
    out = canvas.copy()
    out[out == 15] = 0
    return out


# ---------------------------------------------------------------------------
# Canvas registry
# ---------------------------------------------------------------------------

def _various(build_fn, name):
    return {
        "build_fn":        build_fn,
        "signature_codes": SIGNATURE_CODES,
        "save_dir_suffix": "custom_batch_inpainting_various_layouts",
        "repaint_steps":   80,
        "description":     build_fn.__doc__.split("\n")[0].strip() if build_fn.__doc__ else name,
    }


CANVAS_REGISTRY = {
    # --- original canvases ---------------------------------------------------
    "roundabout": {
        "build_fn":        build_canvas_roundabout,
        "signature_codes": SIGNATURE_CODES,
        "save_dir_suffix": "custom_batch_inpainting_roundabout",
        "repaint_steps":   90,
        "description":     "Petit rond-point, îlot végétation, bloc SE, voiture bras nord (clean_Semcity trunks_vegetation_person script)",
    },
    "roundabout_vegetation_car": {
        "build_fn":        build_canvas_roundabout_vegetation_car,
        "signature_codes": SIGNATURE_CODES,
        "save_dir_suffix": "custom_batch_inpainting_roundabout_vegetation_car",
        "repaint_steps":   60,
        "description":     "Petit rond-point, îlot végétation, bloc SE, voiture bras nord",
    },
    "road_trunks_vegetation_person": {
        "build_fn":        build_canvas_road_trunks_vegetation_person,
        "signature_codes": SIGNATURE_CODES,
        "save_dir_suffix": "custom_batch_inpainting_road_trunks_vegetation_person",
        "repaint_steps":   60,
        "description":     "Route verticale, troncs, végétation, personne, voiture",
    },
    "road_car": {
        "build_fn":        build_canvas_road_car,
        "signature_codes": SIGNATURE_CODES,
        "save_dir_suffix": "custom_batch_inpainting_road_car",
        "repaint_steps":   None,
        "description":     "Route verticale, voiture, bloc végétation",
    },
    "road_building_bicycle_poles": {
        "build_fn":        build_canvas_road_building_bicycle_poles,
        "signature_codes": SIGNATURE_CODES,
        "train_to_label":  TRAIN_TO_LABEL_BUILDING_BICYCLE,
        "save_dir_suffix": "custom_batch_inpainting_road_building_bicycle_poles",
        "repaint_steps":   60,
        "description":     "Route verticale, bâtiment gauche, vélo sur route, poteaux droite",
    },
    # --- various layouts (avec végétation) -----------------------------------
    "s_road":                           _various(make_s_road,                           "s_road"),
    "s_road_sharp":                     _various(make_s_road_sharp,                     "s_road_sharp"),
    "v_road":                           _various(make_v_road,                           "v_road"),
    "u_road":                           _various(make_u_road,                           "u_road"),
    "two_parallel_roads_cars":          _various(make_two_parallel_roads_cars,          "two_parallel_roads_cars"),
    "two_vertical_parallel_roads_cars": _various(make_two_vertical_parallel_roads_cars, "two_vertical_parallel_roads_cars"),
    "three_parallel_roads_veg":         _various(make_three_parallel_roads_vegetation,  "three_parallel_roads_veg"),
    "cross_road":                       _various(make_cross_road,                       "cross_road"),
    "y_junction":                       _various(make_y_junction,                       "y_junction"),
    "diagonal_road":                    _various(make_diagonal_road,                    "diagonal_road"),
    "zigzag_road":                      _various(make_zigzag_road,                      "zigzag_road"),
    "roundabout_plus":                  _various(make_roundabout_plus,                  "roundabout_plus"),
    # --- mêmes layouts sans végétation (_nv) ---------------------------------
    "cross_road_nv":     _various(lambda: _strip_veg(make_cross_road()),     "cross_road_nv"),
    "y_junction_nv":     _various(lambda: _strip_veg(make_y_junction()),     "y_junction_nv"),
    "diagonal_road_nv":  _various(lambda: _strip_veg(make_diagonal_road()),  "diagonal_road_nv"),
    "zigzag_road_nv":    _various(lambda: _strip_veg(make_zigzag_road()),    "zigzag_road_nv"),
    "roundabout_plus_nv":_various(lambda: _strip_veg(make_roundabout_plus()),"roundabout_plus_nv"),
}

# Convenience: patch descriptions for _nv entries
for _nv_name in ["cross_road_nv", "y_junction_nv", "diagonal_road_nv",
                  "zigzag_road_nv", "roundabout_plus_nv"]:
    _base = CANVAS_REGISTRY[_nv_name.replace("_nv", "")]["description"]
    CANVAS_REGISTRY[_nv_name]["description"] = _base + " (sans végétation)"


# ---------------------------------------------------------------------------
# Inpainting pipeline
# ---------------------------------------------------------------------------

def run_inpainting(canvas_name, training_free_dir, model, ae, sample_fn, out_shape,
                   learning_map_inv, n_samples=20, repaint_steps_override=None):
    entry           = CANVAS_REGISTRY[canvas_name]
    build_fn        = entry["build_fn"]
    signature_codes = entry["signature_codes"]
    train_to_label  = entry.get("train_to_label", None)
    save_dir_suffix = entry["save_dir_suffix"]
    repaint_steps   = (repaint_steps_override
                       if repaint_steps_override is not None
                       else entry["repaint_steps"])
    print("repaint_steps", repaint_steps)

    base_save_dir = os.path.join(training_free_dir, save_dir_suffix)
    os.makedirs(base_save_dir, exist_ok=True)
    output_dir = os.path.join(base_save_dir, canvas_name)
    os.makedirs(output_dir, exist_ok=True)

    canvas = build_fn()

    # Encode canvas into VQ-VAE condition
    indices = np.full((128, 128), signature_codes[0], dtype=np.int64)
    for train_id, code in signature_codes.items():
        indices[canvas == train_id] = code
    indices_th = torch.from_numpy(indices).cuda().unsqueeze(0)

    with torch.no_grad():
        quant = ae.vqvae.quantize.embedding(indices_th.view(1, -1))
        quant = quant.view(1, 128, 128, -1).permute(0, 3, 1, 2).contiguous()
        cond, _ = compose_featmaps(
            quant.squeeze(0),
            torch.zeros(1, 8, 128, 32).cuda().squeeze(0),
            torch.zeros(1, 8, 128, 32).cuda().squeeze(0),
            (128, 128, 32),
        )
        cond = cond.unsqueeze(0)

    mask_bev  = (canvas == 0).astype(np.float32)
    full_mask = torch.zeros(out_shape).cuda()
    full_mask[:, :, :128, :128] = (
        torch.from_numpy(mask_bev).unsqueeze(0).unsqueeze(0).cuda()
    )
    model_kwargs = {
        'H': [128], 'W': [128], 'D': [32],
        'y': torch.zeros(1, 1).cuda(), 'data': None,
    }

    # Save input BEV visualisation
    rgb_input = bev_to_rgb(canvas, OFFICIAL_RGB, train_to_label)
    rgb_tensor = torch.from_numpy(rgb_input).permute(2, 0, 1).unsqueeze(0).float()
    rgb_highres = (F.interpolate(rgb_tensor, size=(1024, 1024), mode='nearest')
                   .squeeze().permute(1, 2, 0).byte().numpy())
    plt.imsave(os.path.join(output_dir, "input_layout.png"), rgb_highres)

    sample_kwargs = dict(cond=cond, mode=full_mask, overlap=0, model_kwargs=model_kwargs)
    if repaint_steps is not None:
        sample_kwargs["repaint_steps"] = repaint_steps

    for n in range(n_samples):
        print(f"  [{canvas_name}] Iteration {n + 1}/{n_samples}...")
        with torch.no_grad():
            samples = sample_fn(model, out_shape, **sample_kwargs)
            xy_feat, _, _ = decompose_featmaps(samples, (128, 128, 32))
            logits = ae.decode(xy_feat)
            res_voxels = (torch.softmax(logits, dim=1)
                          .argmax(dim=1)[0].cpu().numpy().astype(np.uint8))

            occupied_voxels = np.argwhere(res_voxels > 0)
            colors = [
                OFFICIAL_RGB.get(learning_map_inv.get(int(v), 0), [255, 255, 255])
                for v in res_voxels[res_voxels > 0]
            ]
            mesh    = build_voxel_mesh(occupied_voxels, np.array(colors))
            img_vtk = render_with_vtk(mesh)

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(img_vtk)
            ax.set_title(f"{canvas_name} – Inpainted Result #{n + 1}")
            ax.axis("off")
            plt.savefig(
                os.path.join(output_dir, f"output_{n + 1}.png"),
                dpi=300, bbox_inches='tight', pad_inches=0.1,
            )
            plt.close(fig)
            np.save(os.path.join(output_dir, f"output_{n + 1}_3d.npy"), res_voxels)

        torch.cuda.empty_cache()

    print(f"  → Saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch inpainting with configurable canvas layout(s).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        metavar="PATH",
        help=(
            "Path to the diffusion YAML config used to build the sampling model.\n"
            "Required unless --list is passed."
        ),
    )
    parser.add_argument(
        "--canvas", nargs="+", default=None,
        metavar="NAME",
        help=(
            "Canvas layout(s) to run. Pass 'all' to run every canvas.\n"
            "Examples:\n"
            "  --canvas roundabout\n"
            "  --canvas s_road u_road cross_road\n"
            "  --canvas all"
        ),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available canvas layouts and exit.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=20,
        help="Number of inpainting samples per layout (default: 20).",
    )
    parser.add_argument(
        "--repaint_steps",
        type=int,
        default=70,
        help=(
            "Override repaint steps for every selected layout. "
            "Default: use each layout's value in CANVAS_REGISTRY (e.g. 80 for roundabout)."
        ),
    )
    parser.add_argument(
        "--codes-json",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to vqvae_codes_99_coverage.json (from analyze_vqvae_codes.py). "
            "Required for every run except --list."
        ),
    )
    parser.add_argument(
        "--signature-code-pick",
        type=str,
        choices=("recommended", "volume"),
        default="volume",
        help=(
            "How to choose one code per class from JSON: "
            "'recommended' uses recommended_code (max purity in the coverage set; default); "
            "'volume' uses codes[0] (largest class volume)."
        ),
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'Canvas name':<40} {'repaint_steps':>13}  Description")
        print("-" * 90)
        for name, entry in CANVAS_REGISTRY.items():
            rs = str(entry["repaint_steps"]) if entry["repaint_steps"] is not None else "none"
            print(f"  {name:<38} {rs:>13}  {entry['description']}")
        return

    if args.canvas is None:
        parser.print_help()
        return

    if args.config is None:
        parser.error("--config is required (path to the diffusion YAML config).")

    selected = list(CANVAS_REGISTRY.keys()) if "all" in args.canvas else args.canvas

    unknown = [n for n in selected if n not in CANVAS_REGISTRY]
    if unknown:
        parser.error(
            f"Unknown canvas(es): {unknown}\n"
            f"Run with --list to see available canvases."
        )

    pick = args.signature_code_pick
    if not args.codes_json:
        parser.error(
            "--codes-json PATH is required. Example:\n"
            "  --codes-json models/semantic_ae/common_ae_base/vqvae_codes_99_coverage.json"
        )
    if not os.path.isfile(args.codes_json):
        parser.error(f"--codes-json path not found: {args.codes_json}")

    cfg = OmegaConf.load(args.config)
    cfg.repaint_descent_only = True
    cfg.batch_size = 1

    print(f"Loading VQ-VAE signature codes (pick={pick}):\n  {args.codes_json}")
    apply_signature_codes_from_json(args.codes_json, pick=pick)

    print("Building models...")
    (model, ae, sample_fn, coords, query, out_shape,
     learning_map, learning_map_inv, H, W, D, grid_size,
     class_name, cfg, diffusion) = build_sampling_model(cfg)

    training_free_dir = os.path.dirname(cfg.training_free_dir)

    print(f"\nLayouts to process ({len(selected)}): {selected}\n")
    for canvas_name in selected:
        print(f"\n{'=' * 60}")
        print(f"Processing: {canvas_name}")
        run_inpainting(
            canvas_name=canvas_name,
            training_free_dir=training_free_dir,
            model=model, ae=ae, sample_fn=sample_fn,
            out_shape=out_shape, learning_map_inv=learning_map_inv,
            n_samples=args.n_samples,
            repaint_steps_override=args.repaint_steps,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
