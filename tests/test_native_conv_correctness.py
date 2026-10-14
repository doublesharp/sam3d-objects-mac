import os
os.environ.setdefault('SPARSE_BACKEND','native')
os.environ.setdefault('LIDRA_SKIP_INIT','true')
import pytest
import torch
import torch.nn.functional as F
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv.conv_native import _subm_forward, _sparse_forward

@pytest.mark.parametrize('device',['cpu','mps'])
def test_submanifold_matches_dense_convolution(device):
    if device=='mps' and not torch.backends.mps.is_available():
        pytest.skip('Requires MPS')
    coords=torch.tensor([[0,0,0,0],[0,1,1,1],[0,1,1,2],[0,2,1,1],[1,1,1,1]],dtype=torch.long)
    generator=torch.Generator().manual_seed(12)
    feats=torch.randn(len(coords),2,generator=generator)
    weight=torch.randn(3,3,3,2,3,generator=generator)
    bias=torch.randn(3,generator=generator)
    dense=torch.zeros(2,2,4,4,4)
    for coord,feat in zip(coords,feats):
        b,z,y,x=coord.tolist();dense[b,:,z,y,x]=feat
    expected_dense=F.conv3d(dense,weight.permute(4,3,0,1,2),bias,padding=1)
    expected=torch.stack([expected_dense[b,:,z,y,x] for b,z,y,x in coords.tolist()])
    actual=_subm_forward(feats.to(device),coords.to(device),weight.to(device),bias.to(device)).cpu()
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)


def test_strided_kernel_one_accumulates_voxels():
    coords=torch.tensor([[0,0,0,0],[0,0,0,1],[0,2,0,0]])
    feats=torch.tensor([[1.],[2.],[4.]])
    out,idx,spatial=_sparse_forward(feats,coords,torch.ones(1,1,1,1,1),None,(2,2,2),0,[3,1,2],1)
    assert idx.tolist()==[[0,0,0,0],[0,1,0,0]]
    torch.testing.assert_close(out,torch.tensor([[3.],[4.]]))
    assert spatial==[2,1,1]


@pytest.mark.parametrize('device',['cpu','mps'])
def test_neighbor_cache_handles_feature_updates_and_coordinate_changes(device):
    if device=='mps' and not torch.backends.mps.is_available():
        pytest.skip('Requires MPS')
    coords=torch.tensor([[0,0,0,0],[0,0,0,1],[0,1,0,0]],dtype=torch.long)
    feats=torch.tensor([[1.],[2.],[3.]])
    weight=torch.arange(27,dtype=torch.float32).reshape(3,3,3,1,1)
    coords=coords.to(device);feats=feats.to(device);weight=weight.to(device)
    cache={}
    for current_coords,current_feats in [
        (coords,feats),(coords.clone(),feats*2),(coords[[2,0,1]],feats[[2,0,1]]),
        (coords+torch.tensor([0,4,0,0],device=device),feats),
    ]:
        expected=_subm_forward(current_feats,current_coords,weight,None)
        actual=_subm_forward(current_feats,current_coords,weight,None,neighbor_cache=cache)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    original=_subm_forward(feats,coords,weight,None,neighbor_cache=cache)
    coords[0,3]=3
    expected=_subm_forward(feats,coords,weight,None)
    actual=_subm_forward(feats,coords,weight,None,neighbor_cache=cache)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert not torch.equal(original,actual)
