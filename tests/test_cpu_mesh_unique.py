import torch
import pytest
from sam3d_objects import cpu_mesh_utils

@pytest.mark.parametrize("rows", [torch.tensor([[2,-1],[1,3],[2,-1],[3,1],[1,3]],dtype=torch.int32), torch.empty((0,3),dtype=torch.int64), torch.tensor([[-2**63,0],[2**63-1,0]],dtype=torch.int64), torch.arange(24).reshape(4,6)[:,::2], torch.tensor([[1,2],[2,1],[1,2]])])
def test_packed_unique_matches_rows_inverse_counts(rows):
    assert hasattr(cpu_mesh_utils, 'unique_rows')
    actual=cpu_mesh_utils.unique_rows(rows,return_inverse=True,return_counts=True)
    expected=torch.unique(rows,dim=0,return_inverse=True,return_counts=True)
    for a,b in zip(actual,expected): assert torch.equal(a,b)
