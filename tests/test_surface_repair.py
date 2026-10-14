import numpy as np
import pytest
import trimesh
from sam3d_objects.surface_repair import RepairRejected, repair_surface, validate_repair


def test_closed_mesh_is_preserved_without_remeshing():
    mesh=trimesh.creation.box()
    result=repair_surface(mesh.vertices,mesh.faces,resolution=48)
    np.testing.assert_array_equal(result.vertices,mesh.vertices)
    np.testing.assert_array_equal(result.faces,mesh.faces)
    assert result.report['status']=='unchanged'


def test_punctured_surface_becomes_closed():
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=.5)
    faces=mesh.faces[1:]
    result=repair_surface(mesh.vertices,faces,resolution=48,closing_radius=1)
    clean=trimesh.Trimesh(result.vertices,result.faces,process=False)
    assert clean.is_watertight and clean.is_winding_consistent
    assert result.report['after']['boundary_edges']==0
    assert result.report['after']['nonmanifold_edges']==0
    assert result.report['status']=='repaired'
    assert result.report['surface_error_p95']<.03


def test_thin_appendage_survives_repair():
    body=trimesh.creation.box(extents=[.5,.5,.5])
    leg=trimesh.creation.cylinder(radius=.028,height=.48,sections=12)
    rotation=trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0])
    leg.apply_transform(rotation);leg.apply_translation([.46,-.18,0])
    mesh=trimesh.util.concatenate([body,leg]);faces=np.delete(mesh.faces,len(body.faces)+int(np.argmin(leg.area_faces)),axis=0)
    result=repair_surface(mesh.vertices,faces,resolution=96,closing_radius=1)
    assert result.vertices[:,0].max()>.675
    assert result.report['bounds_retention_min']>.95
    assert result.report['source_error_max']<.06
    assert trimesh.Trimesh(result.vertices,result.faces,process=False).is_watertight


def test_quality_gate_rejects_missing_appendage():
    body=trimesh.creation.box(extents=[.5,.5,.5])
    tip=trimesh.creation.icosphere(subdivisions=1,radius=.02)
    tip.apply_translation([.7,-.2,0])
    mesh=trimesh.util.concatenate([body,tip])
    with pytest.raises(RepairRejected,match='shape|bounds|detail'):
        validate_repair(mesh.vertices,mesh.faces,body.vertices,body.faces)


def test_invalid_geometry_is_rejected():
    mesh=trimesh.creation.box();vertices=np.asarray(mesh.vertices).copy();vertices[0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        repair_surface(vertices,mesh.faces)


def test_surface_extraction_avoids_exact_half_voxel_contacts():
    from sam3d_objects.surface_repair import extract_closed_surface, topology
    # At exactly 0.5 this field creates three edges shared by four triangles.
    solid=np.pad(np.random.default_rng(6).random((5,5,5))>.55,2)
    vertices,faces=extract_closed_surface(solid,1.,np.zeros(3))
    stats=topology(vertices,faces)
    assert stats['boundary_edges']==0 and stats['nonmanifold_edges']==0
    assert stats['watertight']


def test_glb_export_repairs_before_export_and_keeps_vertex_colors(tmp_path):
    import torch
    from types import SimpleNamespace
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb
    from sam3d_objects.surface_repair import topology
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=.5)
    raw=SimpleNamespace(vertices=torch.tensor(mesh.vertices,dtype=torch.float32),
                        faces=torch.tensor(mesh.faces[1:]),
                        vertex_attrs=torch.full((len(mesh.vertices),3),.6))
    exported=to_glb(None,raw,surface_repair=True,repair_resolution=48,repair_closing_radius=1,
                    repair_report_path=str(tmp_path/'report.json'),
                    with_mesh_postprocess=False,with_texture_baking=False,use_vertex_color=True,verbose=False)
    assert topology(np.asarray(exported.vertices),np.asarray(exported.faces))['watertight']
    assert len(exported.visual.vertex_colors)==len(exported.vertices)
    assert (tmp_path/'report.json').exists()


def test_fast_texture_bake_keeps_uv_color_orientation(monkeypatch):
    import torch
    from types import SimpleNamespace
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils as module
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
    axis=np.linspace(-1.,1.,15,dtype=np.float32);xx,yy=np.meshgrid(axis,axis)
    points=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)]).astype(np.float32)
    colors=np.column_stack([(points[:,0]+1)/2,(points[:,1]+1)/2,np.full(len(points),.25)]).astype(np.float32)
    gs=Gaussian(aabb=[-1,-1,-1,2,2,2],device='cpu');gs.from_xyz(torch.from_numpy(points))
    gs._features_dc=torch.from_numpy(((colors-.5)/.28209479177387814)[:,None,:])
    gs._scaling=torch.zeros(len(points),3);gs._rotation=torch.zeros(len(points),4);gs._opacity=torch.zeros(len(points),1)
    vertices=np.array([[-.8,-.8,0],[.8,-.8,0],[.8,.8,0],[-.8,.8,0]],dtype=np.float32)
    mesh=SimpleNamespace(vertices=torch.from_numpy(vertices),faces=torch.tensor([[0,1,2],[0,2,3]]),vertex_attrs=torch.ones(4,3))
    def legacy(*args,**kwargs):raise AssertionError('Fast Gaussian mode unexpectedly used legacy multiview rendering')
    monkeypatch.setattr(module,'render_multiview',legacy)
    exported=module.to_glb(gs,mesh,texture_mode='gaussian',texture_size=128,
                           surface_repair=False,with_mesh_postprocess=False,verbose=False)
    # Barycentric samples inside each face avoid atlas padding and verify both axes.
    vertices_before_rotation=np.asarray(exported.vertices)@np.array([[1,0,0],[0,0,-1],[0,1,0]]).T
    positions=vertices_before_rotation[exported.faces].mean(1)
    uvs=exported.visual.uv[exported.faces].mean(1)
    sampled=trimesh.visual.color.uv_to_color(uvs,exported.visual.material.baseColorTexture)[:,:3]/255
    expected=np.column_stack([(positions[:,0]+1)/2,(positions[:,1]+1)/2,np.full(len(positions),.25)])
    np.testing.assert_allclose(sampled,expected,atol=.10)
    assert exported.visual.material.metallicFactor==0


def test_large_opening_is_not_accepted_as_a_thin_wrapper():
    mesh=trimesh.creation.box()
    faces=mesh.faces[mesh.face_normals[:,2]<.9]
    with pytest.raises(RepairRejected,match='volume|shell|interior'):
        repair_surface(mesh.vertices,faces,resolution=48,closing_radius=1)


def test_game_optimization_receives_repaired_source_and_retains_budget(monkeypatch):
    import torch
    from types import SimpleNamespace
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils as module
    from sam3d_objects.surface_repair import topology
    mesh=trimesh.creation.icosphere(subdivisions=2)
    bounded_mesh=trimesh.creation.icosphere(subdivisions=1)
    raw=SimpleNamespace(vertices=torch.tensor(mesh.vertices,dtype=torch.float32),faces=torch.tensor(mesh.faces[1:]),vertex_attrs=torch.ones(len(mesh.vertices),3))
    received=[]
    def bounded(vertices,faces,**kwargs):
        received.append(topology(vertices,faces))
        return np.asarray(bounded_mesh.vertices,dtype=np.float32),np.asarray(bounded_mesh.faces,dtype=np.int64)
    monkeypatch.setattr(module,'game_remesh_mesh',bounded)
    exported=module.to_glb(None,raw,game_remesh=True,game_target_faces=80,surface_repair=True,
                           repair_resolution=48,repair_closing_radius=1,with_mesh_postprocess=False,
                           with_texture_baking=False,verbose=False)
    assert received[0]['watertight'], 'Repair must happen before game optimization'
    assert len(exported.faces)<=80


@pytest.mark.parametrize('unreliable_volume',['zero','inconsistent_winding'])
def test_large_opening_is_rejected_even_without_reliable_source_volume(unreliable_volume):
    mesh=trimesh.creation.box()
    if unreliable_volume=='zero':
        faces=mesh.faces[np.abs(mesh.face_normals[:,0])<.9]
    else:
        faces=mesh.faces[mesh.face_normals[:,2]<.9].copy();faces[0]=faces[0,::-1]
    with pytest.raises(RepairRejected,match='large|interior|shell'):
        repair_surface(mesh.vertices,faces,resolution=48,closing_radius=1)


def test_adaptive_repair_profile_is_rotation_invariant_for_thin_objects():
    from sam3d_objects.surface_repair import choose_repair_settings
    mesh=trimesh.creation.box(extents=[1.,.9,.25])
    rotated=mesh.copy();rotated.apply_transform(trimesh.transformations.rotation_matrix(.63,[1,2,3]))
    assert choose_repair_settings(mesh.vertices)==(224,2)
    assert choose_repair_settings(rotated.vertices)==(224,2)


def test_adaptive_repair_profile_uses_stronger_closing_for_compact_objects():
    from sam3d_objects.surface_repair import choose_repair_settings
    mesh=trimesh.creation.icosphere(subdivisions=1)
    assert choose_repair_settings(mesh.vertices)==(192,4)


def test_forced_repair_counts_interior_sealed_by_closing_itself():
    source=trimesh.creation.box(extents=[1,.08,1])
    result=repair_surface(source.vertices,source.faces,force=True,resolution=128,closing_radius=6)
    assert result.report['interior_voxels']>0
    assert result.report['after']['watertight']
    assert result.report['after']['boundary_edges']==result.report['after']['nonmanifold_edges']==0
    assert result.report['surface_error_p95']<.035


def test_closed_edges_do_not_allow_geometric_self_intersections():
    first=trimesh.creation.box()
    second=trimesh.creation.box();second.apply_translation([.4,.2,.1])
    overlap=trimesh.util.concatenate([first,second])
    assert overlap.is_watertight
    with pytest.raises(RepairRejected,match='intersect'):
        validate_repair(overlap.vertices,overlap.faces,overlap.vertices,overlap.faces)


def test_unchanged_path_rejects_self_intersecting_closed_sources():
    first=trimesh.creation.box()
    second=trimesh.creation.box();second.apply_translation([.4,.2,.1])
    overlap=trimesh.util.concatenate([first,second])
    try:
        result=repair_surface(overlap.vertices,overlap.faces,resolution=48,closing_radius=2)
    except RepairRejected:
        return
    assert result.report['status'] != 'unchanged'
    assert result.report['geometry']['strict_watertight']


def test_game_cleanup_repairs_duplicate_surface_and_enforces_budget():
    from sam3d_objects.surface_repair import repair_game_surface, geometry_report
    source=trimesh.creation.icosphere(subdivisions=1)
    duplicate=np.vstack([source.faces,source.faces[:1]])
    result=repair_game_surface(source.vertices,source.faces,source.vertices,duplicate,max_faces=len(source.faces))
    assert geometry_report(result.vertices,result.faces)['strict_watertight']
    assert len(result.faces)<=len(source.faces)
    with pytest.raises(RepairRejected,match='budget'):
        repair_game_surface(source.vertices,source.faces,source.vertices,duplicate,max_faces=4)
