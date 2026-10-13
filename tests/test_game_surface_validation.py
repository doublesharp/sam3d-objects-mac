"""Final game export checks must apply to valid geometry as well as cleanup."""
import os
os.environ.setdefault("SPARSE_BACKEND", "native")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

from types import SimpleNamespace
import numpy as np
import pytest
import torch
import trimesh
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils as module
from sam3d_objects.surface_repair import RepairRejected


def _decoder_mesh(mesh):
    return SimpleNamespace(vertices=torch.tensor(mesh.vertices, dtype=torch.float32),
                           faces=torch.tensor(mesh.faces),
                           vertex_attrs=torch.ones(len(mesh.vertices), 3))


def _export(mesh, **kwargs):
    return module.to_glb(None, _decoder_mesh(mesh), game_remesh=True,
                         surface_repair=True, repair_profile="surface", repair_force=False,
                         with_mesh_postprocess=False, with_texture_baking=False,
                         verbose=False, **kwargs)


@pytest.mark.parametrize("in_place", [False, True])
def test_closed_game_mesh_cannot_move_away_from_repaired_source(monkeypatch, in_place):
    source = trimesh.creation.icosphere(subdivisions=1)
    def shifted(vertices, faces, **kwargs):
        if in_place:
            vertices[:, 0] += 1.
            return vertices, faces
        return vertices + np.array([1., 0., 0.], dtype=np.float32), faces
    monkeypatch.setattr(module, "game_remesh_mesh", shifted)
    with pytest.raises(RepairRejected, match="shape|surface") as rejected:
        _export(source)
    assert rejected.value.report["status"] == "rejected"


@pytest.mark.parametrize("requested_faces, env_faces, env_vertices, expected_faces", [
    (60, 20000, 10000, 60),
    (80, 60, 10000, 60),
    (80, 20000, 40, 80),
])
def test_closed_game_mesh_must_respect_final_export_caps(
        monkeypatch, requested_faces, env_faces, env_vertices, expected_faces):
    source = trimesh.creation.icosphere(subdivisions=1)
    monkeypatch.setenv("SAM3D_GAME_MAX_FACES", str(env_faces))
    monkeypatch.setenv("SAM3D_GAME_MAX_VERTICES", str(env_vertices))
    monkeypatch.setattr(module, "game_remesh_mesh", lambda vertices, faces, **kwargs: (vertices, faces))
    with pytest.raises(RepairRejected, match="budget") as rejected:
        _export(source, game_target_faces=requested_faces)
    report = rejected.value.report
    assert report["status"] == "rejected"
    assert report["triangles"] == len(source.faces)
    assert report["vertices"] == len(source.vertices)
    assert report["target_faces"] == expected_faces
    assert report["target_vertices"] == env_vertices


def test_closed_sphere_budget_fallback_keeps_vertices_inside_cap(monkeypatch):
    import sam3d_objects.experimental_retopo as experimental
    source = trimesh.creation.icosphere(subdivisions=5)
    monkeypatch.setenv("SAM3D_GAME_MAX_FACES", "20000")
    monkeypatch.setenv("SAM3D_GAME_MAX_VERTICES", "10000")
    def unchanged(vertices, faces, **kwargs):
        return SimpleNamespace(vertices=vertices, faces=faces,
                               quads=np.empty((0, 4), dtype=np.int64),
                               repair_faces=faces, stats={})
    monkeypatch.setattr(experimental, "retopologize", unchanged)
    appearance = SimpleNamespace(get_xyz=torch.zeros(1, 3), get_scaling=torch.ones(1, 3),
                                 get_rotation=torch.tensor([[1., 0., 0., 0.]]),
                                 get_opacity=torch.ones(1, 1), get_features=torch.zeros(1, 1, 3))
    exported = module.to_glb(appearance, _decoder_mesh(source), experimental_retopo=True,
                            experimental_source_is_clean=True, surface_repair=True,
                            repair_profile="surface", repair_force=False,
                            with_mesh_postprocess=False, with_texture_baking=False,
                            verbose=False)
    assert len(exported.vertices) <= 10000
    assert len(exported.faces) <= 20000
    assert exported.is_watertight and exported.is_winding_consistent
    assert exported.metadata["surface_repair"]["game_validation"]["status"] == "repaired"
