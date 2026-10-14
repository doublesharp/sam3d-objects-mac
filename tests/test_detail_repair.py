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


@pytest.mark.parametrize('shape', [(3,4,2),(17,19,15)])
@pytest.mark.parametrize('radius', [0,1,4,6])
@pytest.mark.parametrize('kind', ['empty','full','random','boundary'])
def test_spherical_closing_exact_finite_boundary(shape,radius,kind):
    from sam3d_objects import cpu_repair_utils
    assert hasattr(cpu_repair_utils,'spherical_closing')
    field=np.zeros(shape,dtype=bool)
    if kind=='full':field[:]=True
    if kind=='random':field=np.random.default_rng(7).random(shape)>.7
    if kind=='boundary':field[0]=True;field[-1]=True
    expected=ndimage.binary_closing(field,structure=detail_repair._ball(radius))
    assert np.array_equal(cpu_repair_utils.spherical_closing(field,radius),expected)

@pytest.mark.parametrize('kind', ['empty','degenerate','large','small'])
def test_banded_shell_matches_actual_full_open3d_distances(kind):
    from sam3d_objects import cpu_repair_utils
    from sam3d_objects.surface_repair import _scene,_distance
    assert hasattr(cpu_repair_utils,'sample_shell')
    vertices=np.array([[0,0,0],[.7,0,0],[0,.3,.2]],dtype=np.float32)
    faces=np.array([[0,1,2]],dtype=np.int64)
    if kind=='empty':faces=np.empty((0,3),dtype=np.int64)
    if kind=='degenerate':vertices[:]=0
    if kind=='large':vertices*=100
    scene=_scene(vertices,faces)
    pitch=.071;origin=np.array([-.213,-.142,-.071],dtype=np.float32);shape=(17,11,7)
    full=detail_repair._sample_shell(scene,pitch,origin,shape)
    band=cpu_repair_utils.sample_shell(scene,pitch,origin,shape,_distance,vertices,faces)
    assert np.array_equal(full,band)


@pytest.mark.parametrize('direction', [-1,0,1])
def test_banded_shell_threshold_ties_and_float32_neighbors(direction):
    from sam3d_objects import cpu_repair_utils
    from sam3d_objects.surface_repair import _scene,_distance
    # Exact binary pitch and translated origin keep the threshold tie representable.
    pitch=.125
    origin=np.array([4.,-2.,8.],dtype=np.float32)
    height=np.float32(origin[2]+pitch*.75)
    if direction:
        height=np.nextafter(height,np.float32(-np.inf if direction<0 else np.inf))
    vertices=np.array([[3.,-3.,height],[6.,-3.,height],[3.,0.,height],[6.,0.,height]],dtype=np.float32)
    faces=np.array([[0,1,2],[1,3,2]],dtype=np.int64)
    scene=_scene(vertices,faces)
    shape=(5,5,4)
    full=detail_repair._sample_shell(scene,pitch,origin,shape)
    band=cpu_repair_utils.sample_shell(scene,pitch,origin,shape,_distance,vertices,faces)
    assert np.array_equal(full,band)
    assert full.any() and (~full).any()
    assert full[:,:,0].all() if direction<=0 else not full[:,:,0].any()
    assert not full[:,:,-1].any()
