from itertools import product
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from galaxea_r1pro.galaxea_r1pro_0.utils.compact_spheres import compact_mesh_spheres


def covered(points, spheres):
    center = np.array([s["center"] for s in spheres])
    radius = np.array([s["radius"] for s in spheres])
    return np.min(np.linalg.norm(points[:, None] - center, axis=-1) - radius, axis=1)


def test_covers_rotated_box_volume_and_faces_with_lengthwise_slabs():
    rng = np.random.default_rng(52)
    corners = np.array(list(product([-1.0, 1.0], repeat=3))) * [0.005, 0.012, 0.05]
    rotation = Rotation.from_euler("xyz", [0.3, 0.7, 0.4]).as_matrix()
    shift = np.array([0.03, -0.1, 0.6])
    vertices = corners @ rotation.T + shift
    spheres = compact_mesh_spheres(vertices)
    samples = (
        np.vstack([corners, rng.uniform(-1, 1, (5000, 3)) * [0.005, 0.012, 0.05]]) @ rotation.T
        + shift
    )
    assert np.max(covered(samples, spheres)) <= 0
    dense_grid_count = np.prod(np.ceil(np.ptp(vertices, axis=0) / 0.01))
    assert len(spheres) < dense_grid_count / 3


def test_rigid_transform_preserves_cover_envelope():
    v = np.array(list(product([-0.01, 0.01], [-0.02, 0.02], [-0.07, 0.07])))
    r = Rotation.from_euler("xyz", [0.7, 0.2, 0.3]).as_matrix()
    t = np.array([0.2, 0.4, 0.1])
    a = compact_mesh_spheres(v)
    b = compact_mesh_spheres(v @ r.T + t)
    centers = np.array([s["center"] for s in a]) @ r.T + t
    other = np.array([s["center"] for s in b])
    assert np.max(np.min(np.linalg.norm(centers[:, None] - other, axis=-1), axis=1)) < 1e-10
    np.testing.assert_allclose(sorted(s["radius"] for s in a), sorted(s["radius"] for s in b))


@pytest.mark.parametrize("v", [[[0, 0, 0]], [[0, 0, 0], [0, 0, 0.01]]])
def test_degenerate_piece_still_has_finite_cover(v):
    result = compact_mesh_spheres(v)
    assert np.isfinite([s["radius"] for s in result]).all()
    assert np.max(covered(np.array(v), result)) <= 0


def test_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        compact_mesh_spheres([[float("nan"), 0, 0]])


def test_palm_subdivision_covers_convex_volume_with_bounded_radius():
    from galaxea_r1pro.galaxea_r1pro_0.utils.compact_spheres import palm_mesh_spheres

    rng = np.random.default_rng(83)
    corners = np.array(list(product([-0.03, 0.03], [-0.034, 0.034], [-0.027, 0.027])))
    # Cut a corner off a block: slanted hull faces exercise plane intersections.
    vertices = np.vstack(
        [corners[:-1], [[0.03, 0.034, 0.0], [0.03, 0.0, 0.027], [0.0, 0.034, 0.027]]]
    )
    result = palm_mesh_spheres(vertices)
    weights = rng.dirichlet(np.ones(len(vertices)), size=5000)
    samples = np.vstack([vertices, weights @ vertices])
    assert np.max(covered(samples, result)) <= 1e-9
    assert max(s["radius"] for s in result) <= 0.023 + 1e-9
