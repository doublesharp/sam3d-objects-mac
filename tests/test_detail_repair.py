import numpy as np
import pytest
from scipy import ndimage
from sam3d_objects import detail_repair


def _porous_spiked_shell():
    axis=np.arange(40);x,y,z=np.meshgrid(axis,axis,axis,indexing='ij')
    solid=(x-18)**2+(y-20)**2+(z-20)**2 <= 11**2
    solid[27:38,15:17,19:21]=True
    solid[27:38,24:26,19:21]=True
    shell=solid & ~ndimage.binary_erosion(solid)
    shell[:,19:22,19:22]=False
    return shell


def test_body_core_fills_body_without_swallowing_spike_tips():
    assert hasattr(detail_repair,'body_core_from_shell')
    shell=_porous_spiked_shell()
    core=detail_repair.body_core_from_shell(shell,closing_radius=3,opening_radius=2,erosion=1)
    assert core[18,20,20]
    assert not core[36,15,20]
    assert not core[36,25,20]


def test_combined_surface_keeps_separated_outer_spikes_and_seals_body():
    assert hasattr(detail_repair,'compose_body_and_detail')
    shell=_porous_spiked_shell()
    core=detail_repair.body_core_from_shell(shell,closing_radius=3,opening_radius=2,erosion=1)
    solid=detail_repair.compose_body_and_detail(shell,core)
    assert solid[18,20,20]
    assert solid[36,15,20] and solid[36,25,20]
    assert not solid[36,20,20]


def test_empty_body_core_is_rejected_instead_of_fabricating_a_body():
    assert hasattr(detail_repair,'body_core_from_shell')
    with pytest.raises(ValueError,match='body core'):
        detail_repair.body_core_from_shell(np.zeros((24,24,24),dtype=bool),3,2,1)


def test_protected_repair_checks_face_budget_after_reduction(monkeypatch):
    import trimesh
    from sam3d_objects.surface_repair import RepairRejected
    sphere=trimesh.creation.icosphere(subdivisions=2)
    def retain_mesh(mesh, target):
        return mesh
    monkeypatch.setattr(detail_repair,'_simplify_detail',retain_mesh,raising=False)
    with pytest.raises(RepairRejected,match='budget'):
        detail_repair.repair_with_protected_detail(
            sphere.vertices,sphere.faces,resolution=64,core_resolution=32,
            closing_radius=3,opening_radius=2,erosion=1,target_faces=100)


def test_body_core_with_a_handle_is_not_qualified_as_solid():
    import trimesh
    from sam3d_objects.surface_repair import RepairRejected
    x,y,z=np.mgrid[-16:17,-16:17,-16:17]
    core=(np.sqrt(x*x+y*y)-9)**2+z*z<3**2
    sphere=trimesh.creation.icosphere(subdivisions=1,radius=2)
    with pytest.raises(RepairRejected,match='body core'):
        detail_repair._validate_detail(sphere.vertices,sphere.faces,
            sphere.vertices,sphere.faces,core,.05,np.array([-.8,-.8,-.8]),.01)


def test_removed_visible_appendage_is_rejected_even_with_identical_bounds():
    import trimesh
    from sam3d_objects.surface_repair import RepairRejected
    body=trimesh.creation.icosphere(subdivisions=2)
    arms=[]
    for y in (-.4,.4):
        arm=trimesh.creation.cone(radius=.08,height=1.,sections=16)
        arm.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
        arm.apply_translation([1.1,y,0])
        arms.append(arm)
    source=trimesh.util.concatenate([body,*arms])
    missing=trimesh.util.concatenate([body,arms[0]])
    assert np.allclose(source.bounds,missing.bounds)
    core=np.zeros((33,33,33),dtype=bool);core[13:20,13:20,13:20]=True
    with pytest.raises(RepairRejected,match='exterior'):
        detail_repair._validate_detail(source.vertices,source.faces,
            missing.vertices,missing.faces,core,.05,np.array([-.8,-.8,-.8]),.01)


def test_source_visibility_excludes_enclosed_cavity_walls():
    import trimesh
    outer=trimesh.creation.box(extents=[2,2,2])
    inner=trimesh.creation.box(extents=[.5,.5,.5])
    source=trimesh.util.concatenate([outer,inner])
    visible=detail_repair._source_visibility(source.vertices,source.faces,source.vertices,.01)
    assert visible[:len(outer.vertices)].all()
    assert not visible[len(outer.vertices):].any()
