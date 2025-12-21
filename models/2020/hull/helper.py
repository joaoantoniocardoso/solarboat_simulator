import numpy as np
from meshmagick import mmio, mesh, hydrostatics
from meshmagick.mesh_clipper import MeshClipper
from meshmagick.mesh import Plane
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from utils.plot import figsize


def _load_clean_mesh(stl_path: str):
    V, F = mmio.load_STL(stl_path)
    mymesh = mesh.Mesh(V, F)

    # Cleanup
    mymesh.remove_degenerated_faces(rtol=1e-2)
    mymesh.merge_duplicates(atol=1e-5)

    return mymesh


def _strip_deck(hull_mesh: mesh.Mesh):
    face_centers = hull_mesh.faces_centers
    face_normals = hull_mesh.faces_normals
    z_max = np.max(face_centers[:, 2])
    deck_mask = (face_centers[:, 2] > (z_max - 1e-3)) & (face_normals[:, 2] > 0.9)
    hull_only = hull_mesh.extract_faces(np.where(~deck_mask)[0])
    return hull_only


def compute_total_area(stl_path: str):
    """Compute total hull surface area and a wetted area proxy using a clip at deck edge."""
    mymesh = _load_clean_mesh(stl_path)
    hull_mesh = _strip_deck(mymesh)

    # Align to waterplane
    z_top = np.max(hull_mesh.vertices[:, 2])
    hull_mesh.translate([0, 0, -z_top])

    waterplane = Plane(normal=[0, 0, 1], scalar=1e-6)  # Slightly below z=0
    clipper = MeshClipper(
        hull_mesh, plane=waterplane, assert_closed_boundaries=False, verbose=False
    )

    wet_area = float(np.sum(clipper.clipped_mesh.faces_areas))
    total_area = float(np.sum(hull_mesh.faces_areas))

    assert np.isclose(wet_area, total_area)

    return dict(total_area=total_area)


def _waterline_metrics(hull_mesh: mesh.Mesh, atol: float = 1e-2):
    """Compute BWL, LWL, and AWP from waterline vertices with a fallback to overall extents."""
    verts = hull_mesh.vertices
    wl_mask = np.isclose(verts[:, 2], 0.0, atol=atol)

    if np.any(wl_mask):
        y_wl = verts[wl_mask, 1]
        x_wl = verts[wl_mask, 0]
        bwl = float(y_wl.max() - y_wl.min())
        lwl = float(x_wl.max() - x_wl.min())

        wl_points = verts[wl_mask][:, :2]
        try:
            hull = ConvexHull(wl_points)
            awp = float(hull.volume)
        except Exception:
            awp = float(0.0)
    else:
        bwl = float(verts[:, 1].max() - verts[:, 1].min())
        lwl = float(verts[:, 0].max() - verts[:, 0].min())
        awp = float(0.0)

    return bwl, lwl, awp


def compute_hull_dimensions(
    stl_path: str,
    cog_x: float,
    disp_mass: float,
    water_density_name: str = "SALT_WATER",
):
    """Compute hull dimensions (LWL, BWL, AWP) at a given operating point using hydrostatics equilibrium."""
    V, F = mmio.load_STL(stl_path)
    mymesh = mesh.Mesh(V, F)
    mymesh.remove_degenerated_faces(rtol=1e-2)
    mymesh.merge_duplicates(atol=1e-5)

    cog = np.array([cog_x, 0, 0])
    disp = disp_mass / 1000
    water_density = hydrostatics.densities.get_density(water_density_name)
    grav = 9.81
    z_corr, rotmat_corr = hydrostatics.full_equilibrium(
        mymesh,
        cog,
        disp,
        water_density,
        grav,
        reltol=1e-4,
        verbose=False,
    )
    mymesh.rotate_matrix(rotmat_corr)
    mymesh.translate_z(z_corr)

    bwl, lwl, awp = _waterline_metrics(mymesh)

    return {"LWL": lwl, "BWL": bwl, "AWP": awp}


def plot_hull_position(result, mesh):
    from matplotlib.tri import Triangulation

    V_eq = mesh.vertices
    triangles = getattr(mesh, "faces", None)
    triangles = np.asarray(triangles)[:, :3].astype(int)

    tri = Triangulation(V_eq[:, 0], V_eq[:, 2], triangles)

    fig = plt.figure(figsize=figsize(subplots=(1, 2)), constrained_layout=True)

    plt.triplot(tri, color="gray", linewidth=0.2, alpha=1, label="Hull")
    plt.axhline(0.0, color="blue", linewidth=2, alpha=0.5, label="Waterplane")

    cog_x = result.get("cog_x", 0.0)
    cog_z = result.get("cog_z", 0.0)
    plt.scatter([cog_x], [cog_z], color="red", s=20, marker="^")

    plt.xlabel("x [m]")
    plt.ylabel("z [m]")

    return fig
