import importlib
import numpy as np
import pytest
import trimesh
from PIL import Image
import ply2glb
from sam3d_objects.surface_repair import geometry_report, RepairRejected


def factory():
    module = importlib.import_module('sam3d_objects.geometry_certificate')
    return module.from_validated_geometry


def box():
    mesh = trimesh.creation.box()
    return mesh, geometry_report(mesh.vertices, mesh.faces)


def test_snapshot_matches_uv_seams_face_order_and_cyclic_starts():
    mesh, report = box()
    certificate = factory()(mesh.vertices, mesh.faces, report)
    split_vertices = mesh.vertices[mesh.faces.ravel()]
    split_faces = np.arange(len(split_vertices)).reshape(-1, 3)
    split_faces = np.roll(split_faces[::-1], 1, axis=1)
    assert certificate.report_for(split_vertices, split_faces) == report


@pytest.mark.parametrize('change', ['coordinate', 'winding', 'removed_face', 'duplicate_face'])
def test_changes_invalidate_snapshot(change):
    mesh, report = box()
    certificate = factory()(mesh.vertices, mesh.faces, report)
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    if change == 'coordinate':
        vertices[0, 0] += .01
    elif change == 'winding':
        faces[0] = faces[0, ::-1]
    elif change == 'removed_face':
        faces = faces[:-1]
    else:
        faces = np.vstack([faces, faces[0]])
    assert certificate.report_for(vertices, faces) is None


def test_source_and_returned_report_mutations_do_not_change_certificate():
    mesh, report = box()
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    certificate = factory()(vertices, faces, report)
    vertices[0, 0] += 1
    faces[0] = faces[0, ::-1]
    report['strict_watertight'] = False
    first = certificate.report_for(mesh.vertices, mesh.faces)
    assert first['strict_watertight']
    first['strict_watertight'] = False
    assert certificate.report_for(mesh.vertices, mesh.faces)['strict_watertight']


def test_signed_zero_is_normalized_without_tolerance_welding():
    mesh, report = box()
    vertices = mesh.vertices.copy()
    vertices[:, 2] += .5
    certificate = factory()(vertices, mesh.faces, report)
    other = vertices.copy()
    other[other == 0] = -0.
    assert certificate.report_for(other, mesh.faces) is not None
    other[0, 2] = np.nextafter(np.float32(0), np.float32(1))
    assert certificate.report_for(other, mesh.faces) is None


@pytest.mark.parametrize('change', ['nan', 'face_index'])
def test_invalid_geometry_is_checked_even_when_certificate_exists(change):
    mesh, report = box()
    certificate = factory()(mesh.vertices, mesh.faces, report)
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    if change == 'nan':
        vertices[0, 0] = np.nan
    else:
        faces[0, 0] = len(vertices)
    with pytest.raises(ValueError):
        certificate.report_for(vertices, faces)


def test_invalid_report_does_not_create_certificate():
    mesh, report = box()
    report['strict_watertight'] = False
    assert factory()(mesh.vertices, mesh.faces, report) is None


def textured(mesh):
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), image=Image.new('RGB', (2, 2), 'white'))
    return mesh


def test_new_glb_reuses_only_exact_in_memory_certificate(monkeypatch, tmp_path):
    from sam3d_objects import surface_repair
    mesh, report = box()
    vertices = mesh.vertices @ ply2glb.EXPORT_ROTATION
    certificate = factory()(vertices, mesh.faces, report)
    textured(trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)).export(tmp_path / 'mesh.glb')
    def unexpected(*args, **kwargs):
        raise AssertionError('Identical saved geometry reran full strict validation')
    monkeypatch.setattr(surface_repair, 'geometry_report', unexpected)
    result = ply2glb.finish_source_only(str(tmp_path), str(tmp_path / 'mesh.glb'), 'mesh.glb', geometry_certificate=certificate)
    assert result['geometry'][0]['strict_watertight']
    assert result['strict_validation_reused']


def test_mismatched_certificate_runs_full_check_and_rejects_bad_mesh(tmp_path):
    mesh, report = box()
    certificate = factory()(mesh.vertices, mesh.faces, report)
    second = mesh.copy()
    second.apply_translation([.4, .2, .1])
    textured(trimesh.util.concatenate([mesh, second])).export(tmp_path / 'mesh.glb')
    with pytest.raises(RepairRejected):
        ply2glb.finish_source_only(str(tmp_path), str(tmp_path / 'mesh.glb'), 'mesh.glb', geometry_certificate=certificate)


def test_persisted_metadata_is_not_a_validation_certificate(tmp_path):
    mesh, report = box()
    second = mesh.copy(); second.apply_translation([.4, .2, .1])
    unsafe = trimesh.util.concatenate([mesh, second])
    unsafe.metadata['surface_repair'] = {'geometry': report}
    textured(unsafe).export(tmp_path / 'mesh.glb')
    with pytest.raises(RepairRejected):
        ply2glb.finish_source_only(str(tmp_path), str(tmp_path / 'mesh.glb'), 'mesh.glb')


def test_certificate_does_not_bypass_winding_checks(tmp_path):
    mesh, report = box()
    mesh.faces[0] = mesh.faces[0, ::-1]
    certificate = factory()(mesh.vertices, mesh.faces, report)
    textured(mesh).export(tmp_path / 'mesh.glb')
    with pytest.raises(RepairRejected):
        ply2glb.finish_source_only(str(tmp_path), str(tmp_path / 'mesh.glb'), 'mesh.glb', geometry_certificate=certificate)


def test_certificate_does_not_bypass_finite_uv_checks(tmp_path):
    mesh, report = box()
    certificate = factory()(mesh.vertices, mesh.faces, report)
    textured(mesh)
    mesh.visual.uv[0, 0] = np.nan
    mesh.export(tmp_path / 'mesh.glb')
    with pytest.raises(RepairRejected):
        ply2glb.finish_source_only(str(tmp_path), str(tmp_path / 'mesh.glb'), 'mesh.glb', geometry_certificate=certificate)


def test_actual_export_mints_certificate_without_persisting_it(tmp_path):
    import torch
    from types import SimpleNamespace
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb
    mesh, _ = box()
    raw = SimpleNamespace(vertices=torch.tensor(mesh.vertices, dtype=torch.float32),
                          faces=torch.tensor(mesh.faces),
                          vertex_attrs=torch.full((len(mesh.vertices), 3), .6))
    exported = to_glb(None, raw, surface_repair=True, repair_force=False,
                      with_mesh_postprocess=False, with_texture_baking=False,
                      use_vertex_color=True, verbose=False)
    certificate = exported._sam3d_geometry_certificate
    assert certificate.report_for(exported.vertices, exported.faces)['strict_watertight']
    exported.export(tmp_path / 'minted.glb')
    loaded = trimesh.load(tmp_path / 'minted.glb', force='mesh', process=False)
    assert certificate.report_for(loaded.vertices, loaded.faces)['strict_watertight']
    assert not hasattr(loaded, '_sam3d_geometry_certificate')
    assert '_sam3d_geometry_certificate' not in loaded.metadata
