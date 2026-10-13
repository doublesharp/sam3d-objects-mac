from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
import ply2glb
from PIL import Image


def _textured(mesh):
    mesh.visual=trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices),2)),image=Image.new('RGB',(2,2),'white'))
    return mesh


def _run(monkeypatch,tmp_path,mesh):
    _textured(mesh).export(tmp_path/'mesh.glb')
    monkeypatch.setenv('SAM3D_SOURCE_ONLY','1')
    monkeypatch.setattr(ply2glb,'parse_args',lambda:SimpleNamespace(folder=str(tmp_path),output=None,target_vertices=None))
    def unexpected(*args,**kwargs):
        raise AssertionError('Detailed-only reuse loaded reconstruction assets or game fitter')
    monkeypatch.setattr(ply2glb,'load_gaussian',unexpected)
    monkeypatch.setattr(ply2glb,'load_mesh_decoder',unexpected)
    monkeypatch.setattr(ply2glb,'source_glb_to_mesh_result',unexpected)
    ply2glb.main()


def test_source_only_reuses_strict_mesh_without_loading_models_or_game(monkeypatch,tmp_path):
    _run(monkeypatch,tmp_path,trimesh.creation.box())
    assert (tmp_path/'mesh.obj').exists()
    assert not (tmp_path/'mesh_game.glb').exists()
    import json
    report=json.loads((tmp_path/'export-status.json').read_text())
    assert report['status']=='detailed_complete'
    assert report['game']['status']=='not_requested'


def test_source_only_rejects_self_intersecting_existing_mesh(monkeypatch,tmp_path):
    from sam3d_objects.surface_repair import RepairRejected
    first=trimesh.creation.box();second=first.copy();second.apply_translation([.4,.2,.1])
    with pytest.raises(RepairRejected):
        _run(monkeypatch,tmp_path,trimesh.util.concatenate([first,second]))


def test_source_only_rejects_inconsistent_face_winding(monkeypatch,tmp_path):
    from sam3d_objects.surface_repair import RepairRejected
    mesh=trimesh.creation.box();mesh.faces[0]=mesh.faces[0,::-1]
    with pytest.raises(RepairRejected):
        _run(monkeypatch,tmp_path,mesh)
