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


def _waterplane_intersection_xy(
    vertices: np.ndarray, faces: np.ndarray, *, z0: float = 0.0
):
    faces = np.asarray(faces)
    faces = faces[:, :3].astype(int)

    V = np.asarray(vertices, dtype=float)
    P0 = V[faces[:, 0]]
    P1 = V[faces[:, 1]]
    P2 = V[faces[:, 2]]

    d0 = P0[:, 2] - z0
    d1 = P1[:, 2] - z0
    d2 = P2[:, 2] - z0

    pts = []

    def _edge(pa, pb, da, db):
        on_a = np.isclose(da, 0.0, atol=1e-12)
        on_b = np.isclose(db, 0.0, atol=1e-12)
        if np.any(on_a):
            pts.append(pa[on_a][:, :2])
        if np.any(on_b):
            pts.append(pb[on_b][:, :2])

        cross = (da * db) < 0.0
        if np.any(cross):
            t = da[cross] / (da[cross] - db[cross])
            pi = pa[cross] + t[:, None] * (pb[cross] - pa[cross])
            pts.append(pi[:, :2])

    _edge(P0, P1, d0, d1)
    _edge(P1, P2, d1, d2)
    _edge(P2, P0, d2, d0)

    if not pts:
        return np.zeros((0, 2), dtype=float)

    xy = np.vstack(pts)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.size == 0:
        return xy

    xy_round = np.round(xy, decimals=6)
    xy_unique = np.unique(xy_round, axis=0)
    return xy_unique


def _waterline_metrics(hull_mesh: mesh.Mesh, atol: float = 1e-2):
    """Compute BWL, LWL, and AWP from waterplane intersection points."""
    verts = hull_mesh.vertices

    xy = _waterplane_intersection_xy(verts, hull_mesh.faces, z0=0.0)
    if xy.shape[0] >= 3:
        x_wl = xy[:, 0]
        y_wl = xy[:, 1]
        bwl = float(y_wl.max() - y_wl.min())
        lwl = float(x_wl.max() - x_wl.min())
        try:
            awp = float(ConvexHull(xy).volume)
        except Exception:
            awp = float(0.0)
        return bwl, lwl, awp

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
    water_density: float = 1023.0,
):
    """Compute hull dimensions (LWL, BWL, AWP) at equilibrium."""
    V, F = mmio.load_STL(stl_path)
    mymesh = mesh.Mesh(V, F)
    mymesh.remove_degenerated_faces(rtol=1e-2)
    mymesh.merge_duplicates(atol=1e-5)

    cog = np.array([cog_x, 0, 0])
    disp = disp_mass / 1000
    water_density = float(water_density)
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


def _rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def prepare_equilibrium_hull_mesh(
    stl_path: str,
    *,
    cog_x: float,
    volume_m3: float,
    trim_deg: float = 0.0,
):
    mymesh = _load_clean_mesh(stl_path)
    mymesh = _strip_deck(mymesh)

    V0 = np.asarray(mymesh.vertices, dtype=float).copy()
    F = np.asarray(mymesh.faces, dtype=int)

    R = _rotation_matrix_y(np.deg2rad(float(trim_deg)))
    V0[:, 0] -= float(cog_x)
    V0 = V0 @ R.T
    V0[:, 0] += float(cog_x)

    plane = Plane(normal=[0, 0, 1], scalar=0.0)

    def _submerged_volume(dz: float) -> float:
        V = V0.copy()
        V[:, 2] += float(dz)
        m = mesh.Mesh(V, F)
        try:
            clipper = MeshClipper(
                m, plane=plane, assert_closed_boundaries=False, verbose=False
            )
            return float(abs(clipper.clipped_mesh.volume))
        except RuntimeError as e:
            msg = str(e)
            if "above" in msg:
                return 0.0
            if "below" in msg:
                return float("inf")
            raise

    target = float(volume_m3)
    if not np.isfinite(target) or target <= 0:
        raise ValueError(f"Invalid target volume_m3={target}")

    # Find a bracket in dz such that v(dz_lo) >= target >= v(dz_hi)
    dz_lo = 0.0
    v_lo = _submerged_volume(dz_lo)
    dz_hi = 0.0
    v_hi = v_lo

    if v_lo < target:
        dz_lo = -0.1
        for _ in range(30):
            v_lo = _submerged_volume(dz_lo)
            if v_lo >= target:
                break
            dz_lo *= 2.0
    else:
        dz_hi = 0.1
        for _ in range(30):
            v_hi = _submerged_volume(dz_hi)
            if v_hi <= target:
                break
            dz_hi *= 2.0

    if v_lo < target or v_hi > target:
        raise RuntimeError(
            f"Failed to bracket submerged volume (v_lo={v_lo}, v_hi={v_hi}, target={target})"
        )

    for _ in range(40):
        dz_mid = 0.5 * (dz_lo + dz_hi)
        v_mid = _submerged_volume(dz_mid)
        if v_mid >= target:
            dz_lo = dz_mid
        else:
            dz_hi = dz_mid

    V = V0.copy()
    V[:, 2] += float(dz_hi)
    return mesh.Mesh(V, F)


def compute_hull_metrics_at_trim_sinkage(
    base_mesh: mesh.Mesh,
    *,
    cog_x: float,
    trim_deg: float,
    sink_m: float,
):
    V0 = np.asarray(base_mesh.vertices, dtype=float)
    F = np.asarray(base_mesh.faces, dtype=int)

    V = V0.copy()

    R = _rotation_matrix_y(np.deg2rad(float(trim_deg)))
    V[:, 0] -= float(cog_x)
    V = V @ R.T
    V[:, 0] += float(cog_x)

    V[:, 2] += float(sink_m)

    mesh_dyn = mesh.Mesh(V, F)

    xy = _waterplane_intersection_xy(mesh_dyn.vertices, mesh_dyn.faces, z0=0.0)
    x_wl_min = float(np.nan)
    x_wl_max = float(np.nan)

    bwl, lwl, awp = _waterline_metrics(mesh_dyn)

    if xy.shape[0] >= 3:
        x_wl_min = float(np.min(xy[:, 0]))
        x_wl_max = float(np.max(xy[:, 0]))

    draft_keel_m = float(-np.min(mesh_dyn.vertices[:, 2]))

    draft_aft_m = float(np.nan)
    draft_fwd_m = float(np.nan)
    if np.isfinite(x_wl_min) and np.isfinite(x_wl_max) and np.isfinite(lwl) and lwl > 0:
        dx = max(0.02 * lwl, 0.05)
        aft_mask = mesh_dyn.vertices[:, 0] <= (x_wl_min + dx)
        fwd_mask = mesh_dyn.vertices[:, 0] >= (x_wl_max - dx)
        if np.any(aft_mask):
            draft_aft_m = float(-np.min(mesh_dyn.vertices[aft_mask, 2]))
        if np.any(fwd_mask):
            draft_fwd_m = float(-np.min(mesh_dyn.vertices[fwd_mask, 2]))

    if np.isfinite(draft_aft_m) and draft_aft_m <= 0:
        draft_aft_m = float(np.nan)
    if np.isfinite(draft_fwd_m) and draft_fwd_m <= 0:
        draft_fwd_m = float(np.nan)

    draft_mean_m = (
        float(np.nanmean([draft_aft_m, draft_fwd_m]))
        if (np.isfinite(draft_aft_m) or np.isfinite(draft_fwd_m))
        else draft_keel_m
    )
    if not np.isfinite(draft_mean_m) or draft_mean_m <= 0:
        draft_mean_m = draft_keel_m

    waterplane = Plane(normal=[0, 0, 1], scalar=1e-6)
    clipper = MeshClipper(
        mesh_dyn, plane=waterplane, assert_closed_boundaries=False, verbose=False
    )
    wet_area = float(np.sum(clipper.clipped_mesh.faces_areas))

    return {
        "LWL": float(lwl),
        "BWL": float(bwl),
        "AWP": float(awp),
        "x_wl_min": x_wl_min,
        "x_wl_max": x_wl_max,
        "draft_keel_m": draft_keel_m,
        "draft_mean_m": draft_mean_m,
        "draft_aft_m": draft_aft_m,
        "draft_fwd_m": draft_fwd_m,
        "wet_surface_area": wet_area,
    }


def compute_holtrop_inputs(
    stl_path: str,
    cog_x: float,
    disp_mass: float,
    water_density: float = 1023.0,
):
    """Compute a Holtrop(-Mennen) input set at equilibrium.

    Notes:
    - Drafts are derived from the equilibrium mesh relative to z=0 waterplane.
    - CM/CP are approximated from a thin midship slice.
    """

    V, F = mmio.load_STL(stl_path)
    mymesh = mesh.Mesh(V, F)
    mymesh.remove_degenerated_faces(rtol=1e-2)
    mymesh.merge_duplicates(atol=1e-5)

    cog = np.array([cog_x, 0, 0])
    disp = disp_mass / 1000
    water_density = float(water_density)
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

    hs_data = hydrostatics.compute_hydrostatics(
        mymesh,
        cog,
        water_density,
        grav,
        z_corr=z_corr,
        rotmat_corr=rotmat_corr,
        at_cog=True,
    )

    mymesh.rotate_matrix(rotmat_corr)
    mymesh.translate_z(z_corr)

    bwl, lwl, awp = _waterline_metrics(mymesh)

    verts = mymesh.vertices
    wl_mask = np.isclose(verts[:, 2], 0.0, atol=1e-2)
    x_wl_min = float(np.nan)
    x_wl_max = float(np.nan)
    if np.any(wl_mask):
        x_wl = verts[wl_mask, 0]
        x_wl_min = float(np.min(x_wl))
        x_wl_max = float(np.max(x_wl))

    x_mid = float(np.nan)
    if np.isfinite(x_wl_min) and np.isfinite(x_wl_max):
        x_mid = 0.5 * (x_wl_min + x_wl_max)

    draft_keel_m = float(-np.min(verts[:, 2]))
    draft_aft_m = float(np.nan)
    draft_fwd_m = float(np.nan)
    if np.isfinite(x_wl_min) and np.isfinite(x_wl_max) and np.isfinite(lwl) and lwl > 0:
        dx = max(0.02 * lwl, 0.05)
        aft_mask = verts[:, 0] <= (x_wl_min + dx)
        fwd_mask = verts[:, 0] >= (x_wl_max - dx)
        if np.any(aft_mask):
            draft_aft_m = float(-np.min(verts[aft_mask, 2]))
        if np.any(fwd_mask):
            draft_fwd_m = float(-np.min(verts[fwd_mask, 2]))

    draft_mean_m = (
        float(np.nanmean([draft_aft_m, draft_fwd_m]))
        if (np.isfinite(draft_aft_m) or np.isfinite(draft_fwd_m))
        else draft_keel_m
    )

    disp_volume_m3 = float(hs_data["disp_mass"]) / float(water_density)

    cwp = float(np.nan)
    cb = float(np.nan)
    cm = float(np.nan)
    cp = float(np.nan)

    if np.isfinite(lwl) and lwl > 0 and np.isfinite(bwl) and bwl > 0:
        if np.isfinite(awp) and awp > 0:
            cwp = float(awp / (lwl * bwl))
        if (
            np.isfinite(draft_mean_m)
            and draft_mean_m > 0
            and np.isfinite(disp_volume_m3)
        ):
            cb = float(disp_volume_m3 / (lwl * bwl * draft_mean_m))

    if (
        np.isfinite(x_mid)
        and np.isfinite(lwl)
        and lwl > 0
        and np.isfinite(draft_mean_m)
        and draft_mean_m > 0
    ):
        dx_mid = max(0.005 * lwl, 0.02)
        mid_mask = (np.abs(verts[:, 0] - x_mid) <= dx_mid) & (verts[:, 2] <= 0.0)
        yz = verts[mid_mask][:, [1, 2]]
        if yz.shape[0] >= 10:
            try:
                A_mid = float(ConvexHull(yz).volume)
                if np.isfinite(A_mid) and A_mid > 0 and np.isfinite(bwl) and bwl > 0:
                    cm = float(A_mid / (bwl * draft_mean_m))
            except Exception:
                cm = float(np.nan)

        if np.isfinite(cb) and cb > 0 and np.isfinite(cm) and cm > 0:
            cp = float(cb / cm)

    lcb_percent = float(np.nan)
    if np.isfinite(x_mid) and np.isfinite(lwl) and lwl > 0:
        lcb_m = float(hs_data["buoyancy_center"][0]) - x_mid
        lcb_percent = float(100.0 * lcb_m / lwl)

    return {
        "LWL": lwl,
        "BWL": bwl,
        "AWP": awp,
        "disp_mass": float(hs_data["disp_mass"]),
        "disp_volume_m3": disp_volume_m3,
        "draft_keel_m": draft_keel_m,
        "draft_mean_m": draft_mean_m,
        "draft_aft_m": draft_aft_m,
        "draft_fwd_m": draft_fwd_m,
        "lcb_percent": lcb_percent,
        "CWP": cwp,
        "CB": cb,
        "CM": cm,
        "CP": cp,
    }


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
