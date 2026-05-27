"""
Voxel-scene rendering utilities (VTK only, semcity / top-down view).

Provides offscreen rendering of a 3D semantic voxel scene to an RGB image
using VTK. The camera setup is fixed to the SemanticCity ``view='up'``
configuration: 2048x1440 viewport, top-down camera at [128, 128, 140]
looking at [128, 128, 16].

Main entry points used by ``generation/generate_samples.py``:

- ``render_voxel_scene``: render a voxel grid (class ids) to RGB.
- ``save_generated_scene``: render a generated voxel scene and save it as a
  single PNG (no GT, no conditioning, no metrics).
"""

import os

# Configure for headless rendering BEFORE importing VTK.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("VTK_DEFAULT_EGL_DEVICE_INDEX", "0")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("DISPLAY", None)

import gc

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import trimesh


# Fixed semcity / top-down rendering parameters.
_VIEWPORT_WIDTH = 2048
_VIEWPORT_HEIGHT = 1440
_CAMERA_LOCATION = np.array([128.0, 128.0, 140.0])
_LOOK_AT_POINT = np.array([128.0, 128.0, 16.0])
_UP_VECTOR = np.array([1.0, 0.0, 0.0])
_VIEW_ANGLE_DEG = float(np.degrees(np.pi / 2))


def _import_vtk():
    import vtk
    from vtk.util import numpy_support
    vtk.vtkObject.GlobalWarningDisplayOff()
    return vtk, numpy_support


def build_voxel_mesh(occupied_voxels, colors, voxel_size=1.0):
    """Build a single trimesh from a list of occupied voxel coordinates + colors."""
    unit_voxel = trimesh.creation.box(extents=(voxel_size, voxel_size, voxel_size))
    base_vertices = unit_voxel.vertices
    base_faces = unit_voxel.faces

    num_voxels = occupied_voxels.shape[0]

    all_vertices = np.concatenate(
        [base_vertices + offset for offset in occupied_voxels], axis=0
    )

    face_offsets = np.arange(num_voxels) * base_vertices.shape[0]
    all_faces = np.concatenate(
        [base_faces + offset for offset in face_offsets], axis=0
    )

    all_colors = np.concatenate(
        [
            np.tile(np.append(color, 255), (base_vertices.shape[0], 1))
            for color in colors
        ],
        axis=0,
    )

    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    mesh.visual.vertex_colors = all_colors
    return mesh


def render_voxel_scene(
    sem_scene,
    learning_map_inv,
    color_map,
    original_shape=(256, 256, 32),
    mapping="pred",
    depth_color=False,
):
    """Render a voxel semantic scene to an RGBA image with VTK.

    Args:
        sem_scene: np.ndarray of class ids. Shape can be flat or
            ``original_shape``.
        learning_map_inv: dict mapping training class id -> raw class id, or
            ``None`` if no remapping is needed.
        color_map: dict mapping class id -> [R, G, B] (0-255 or 0-1).
        original_shape: target voxel shape (X, Y, Z).
        mapping: ``'pred'`` to apply ``learning_map_inv`` first, anything else
            to use ``sem_scene`` as-is.
        depth_color: if True, color voxels by Z depth instead of class.
    """
    if learning_map_inv is not None and mapping == "pred":
        mapped_sem_scene = np.vectorize(learning_map_inv.get)(sem_scene)
    else:
        mapped_sem_scene = sem_scene

    voxel_map = mapped_sem_scene.reshape(original_shape)
    occupied_voxels = np.argwhere(voxel_map > 0)

    if occupied_voxels.shape[0] == 0:
        # Return a grey image when the scene is empty rather than crashing.
        return np.full((_VIEWPORT_HEIGHT, _VIEWPORT_WIDTH, 4), 128, dtype=np.uint8)

    if depth_color:
        cmap = cm.get_cmap("viridis")
        depth_values = occupied_voxels[:, 2]
        depth_normalized = depth_values / max(np.max(depth_values), 1)
        colors = cmap(depth_normalized)[:, :3]
    else:
        colors = np.array(
            [color_map.get(int(label), [255, 255, 255]) for label in voxel_map[voxel_map > 0]]
        )

    combined_mesh = build_voxel_mesh(occupied_voxels, colors, voxel_size=1.0)

    return _render_with_vtk(combined_mesh)


def _render_with_vtk(combined_mesh):
    vtk, numpy_support = _import_vtk()

    width, height = _VIEWPORT_WIDTH, _VIEWPORT_HEIGHT

    vertices = combined_mesh.vertices.astype(np.float32)
    faces = combined_mesh.faces.astype(np.int32)
    colors = combined_mesh.visual.vertex_colors[:, :3]

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(vertices))

    vtk_cells = vtk.vtkCellArray()
    for face in faces:
        vtk_cells.InsertNextCell(3)
        for vertex_id in face:
            vtk_cells.InsertCellPoint(int(vertex_id))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)

    if colors.max() > 1.0:
        colors_normalized = colors.astype(np.float32) / 255.0
    else:
        colors_normalized = colors.astype(np.float32)
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

    camera = renderer.GetActiveCamera()
    camera.SetPosition(_CAMERA_LOCATION)
    camera.SetFocalPoint(_LOOK_AT_POINT)
    camera.SetViewUp(_UP_VECTOR)
    camera.SetViewAngle(_VIEW_ANGLE_DEG)

    renderer.SetAmbient(0.5, 0.5, 0.5)

    light_direction = np.array([np.cos(np.pi / 2), 0, np.sin(np.pi / 2)])
    light_position = _CAMERA_LOCATION + light_direction * 1000

    light = vtk.vtkLight()
    light.SetPosition(*light_position.tolist())
    light.SetFocalPoint(*_LOOK_AT_POINT.tolist())
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
    components = vtk_array.GetNumberOfComponents()
    arr = numpy_support.vtk_to_numpy(vtk_array)
    arr = arr.reshape((height, width, components))

    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)

    renderer.RemoveAllLights()
    renderer.RemoveAllViewProps()
    render_window.RemoveRenderer(renderer)
    try:
        render_window.Finalize()
    except Exception:
        pass

    del window_to_image, vtk_image, vtk_array
    del render_window, renderer, light, actor, mapper, polydata
    del vtk_colors, vtk_cells, vtk_points
    gc.collect()
 
    return arr


def save_generated_scene(
    generated_scene,
    sample_id,
    learning_map_inv,
    color_map,
    folder_path,
    original_shape=(256, 256, 32),
    depth_color=False,
    dpi=200,
):
    """Render and save ONLY the generated voxel scene as a PNG.

    No GT, no conditioning image, no metrics. The output file is
    ``{folder_path}/{sample_id}_generated.png``.

    Returns the absolute path of the saved PNG.
    """
    os.makedirs(folder_path, exist_ok=True)
    output_path = os.path.join(folder_path, f"{sample_id}_generated.png")

    image_gen = render_voxel_scene(
        generated_scene,
        learning_map_inv,
        color_map,
        original_shape=original_shape,
        depth_color=depth_color,
    )

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.imshow(image_gen)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    plt.close("all")
    gc.collect()

    print(f"[vis] Saved generated scene: {output_path}")
    return output_path


def save_lidar_comparison(
    lidar_scene,
    generated_scene,
    gt_scene,
    sample_id,
    learning_map_inv,
    color_map,
    folder_path,
    original_shape=(256, 256, 32),
    metrics=None,
    dpi=150,
):
    """Render and save a 3-panel comparison: LiDAR | Generated | GT.

    - ``lidar_scene``: binary occupancy grid (e.g. from raw velodyne), rendered
      with depth-based coloring (no semantic remapping).
    - ``generated_scene``: diffusion reconstruction (semantic class ids).
    - ``gt_scene``: ground-truth semantic scene.
    - ``metrics``: optional dict with float keys ``'iou'`` and ``'miou'`` (in %)
      to print under the "Diffusion reconstruction" panel.

    Output file is ``{folder_path}/{sample_id}_comparison.png``.
    """
    os.makedirs(folder_path, exist_ok=True)
    output_path = os.path.join(folder_path, f"{sample_id}_comparison.png")

    image_lidar = render_voxel_scene(
        lidar_scene,
        learning_map_inv=None,
        color_map={},
        original_shape=original_shape,
        mapping=None,
        depth_color=True,
    )
    image_gen = render_voxel_scene(
        generated_scene,
        learning_map_inv,
        color_map,
        original_shape=original_shape,
    )
    image_gt = render_voxel_scene(
        gt_scene,
        learning_map_inv,
        color_map,
        original_shape=original_shape,
    )

    gen_title = "Diffusion reconstruction"
    if metrics is not None:
        iou = metrics.get("iou")
        miou = metrics.get("miou")
        parts = []
        if iou is not None:
            parts.append(f"IoU: {iou:.1f}%")
        if miou is not None:
            parts.append(f"mIoU: {miou:.1f}%")
        if parts:
            gen_title = f"{gen_title}\n{' | '.join(parts)}"

    fig, axes = plt.subplots(1, 3, figsize=(30, 6))
    for ax, img, title in zip(
        axes,
        [image_lidar, image_gen, image_gt],
        ["LiDAR (input)", gen_title, "Ground truth"],
    ):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    plt.close("all")
    gc.collect()

    print(f"[vis] Saved lidar comparison: {output_path}")
    return output_path
