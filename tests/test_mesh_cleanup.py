import os
os.environ.setdefault('SPARSE_BACKEND','native')
os.environ.setdefault('LIDRA_SKIP_INIT','true')
import numpy as np
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
