import os
os.environ.setdefault('SPARSE_BACKEND','native')
os.environ.setdefault('LIDRA_SKIP_INIT','true')
import numpy as np
import pytest
from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import postprocess_mesh


def test_cleanup_keeps_large_component_and_removes_disconnected_small_piece():
    # The large repeated-index workload also covers MPS's large bincount failure.
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[10,0,0],[11,0,0],[10,1,0]],dtype=np.float32)
    faces=np.concatenate([np.tile([0,1,2],(106000,1)),np.tile([3,4,5],(4,1))]).astype(np.int64)
    clean_vertices,clean_faces=postprocess_mesh(vertices,faces,simplify=False,fill_holes=False)
    assert len(clean_faces)==106000
    assert len(clean_vertices)==3
    assert np.isfinite(clean_vertices).all()
    assert clean_faces.max()<len(clean_vertices)


def test_cleanup_routes_topology_to_cpu(monkeypatch):
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils as module
    original=module.utils3d.torch.compute_connected_components
    def checked_components(faces,*args,**kwargs):
        # Model the observed unsupported MPS topology path at the dependency boundary.
        if faces.device.type=='mps':
            raise RuntimeError('MPS topology grouping failed on the real crab mesh')
        return original(faces,*args,**kwargs)
    monkeypatch.setattr(module.utils3d.torch,'compute_connected_components',checked_components)
    test_cleanup_keeps_large_component_and_removes_disconnected_small_piece()


@pytest.mark.parametrize("reject_alternate", [False, True])
def test_simplification_does_not_open_a_closed_surface(monkeypatch, reject_alternate):
    import trimesh
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils as module
    from sam3d_objects.surface_repair import topology
    source=trimesh.creation.icosphere(subdivisions=2)
    def damaged_decimation(poly,*args,**kwargs):
        triangles=poly.faces.reshape(-1,4)[:-1,1:]
        return module.pv.PolyData(poly.points,np.c_[np.full(len(triangles),3),triangles])
    monkeypatch.setattr(module.pv.PolyData,'decimate',damaged_decimation)
    if reject_alternate:
        original=module._open3d_mesh_from_arrays
        class DamagedAlternate:
            def simplify_quadric_decimation(self,**kwargs):
                mesh=original(source.vertices,source.faces)
                mesh.remove_triangles_by_index([0])
                return mesh
        monkeypatch.setattr(module,'_open3d_mesh_from_arrays',lambda *args:DamagedAlternate())
    vertices,faces=postprocess_mesh(np.asarray(source.vertices,dtype=np.float32),source.faces,
                                   simplify_ratio=.8,fill_holes=False)
    report=topology(vertices,faces)
    assert report['watertight'] and report['winding_consistent']
    assert report['boundary_edges']==report['nonmanifold_edges']==0
    if reject_alternate:
        assert len(faces)==len(source.faces)
        assert np.allclose(vertices,source.vertices)



def test_extreme_reduction_of_small_closed_mesh_retains_valid_surface():
    import trimesh
    from sam3d_objects.surface_repair import topology
    source=trimesh.creation.box()
    vertices,faces=postprocess_mesh(np.asarray(source.vertices,dtype=np.float32),source.faces,
                                   simplify_ratio=.95,fill_holes=False)
    report=topology(vertices,faces)
    assert report['watertight'] and report['winding_consistent']
    assert np.isfinite(vertices).all()
    assert report['boundary_edges']==report['nonmanifold_edges']==0


def test_sliver_collapse_preserves_thin_closed_surface(monkeypatch):
    import trimesh
    from sam3d_objects.surface_repair import topology
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils as module
    monkeypatch.delenv('SAM3D_GAME_COLLAPSE_SLIVERS',raising=False)
    source=trimesh.creation.box(extents=[1,.001,1])
    mesh=module._open3d_mesh_from_arrays(source.vertices,source.faces)
    result=module._collapse_skinny_open3d_triangles(mesh)
    vertices=np.asarray(result.vertices);faces=np.asarray(result.triangles)
    report=topology(vertices,faces)
    assert report['watertight'] and report['winding_consistent']
    assert report['boundary_edges']==report['nonmanifold_edges']==0
    assert np.ptp(vertices,axis=0)[1] >= .00097
