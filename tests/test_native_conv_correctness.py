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


def test_vectorized_coordinate_lookup_preserves_last_duplicate_and_large_sparse_keys():
    import numpy as np
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv import conv_native
    assert hasattr(conv_native, '_CoordinateLookup'), 'The vectorized coordinate lookup is missing'
    coords = np.array([[2, 1 << 60, 0, -7], [0, -3, 8, 1], [2, 1 << 60, 0, -7]], dtype=np.int64)
    queries = np.array([[2, 1 << 60, 0, -7], [0, -3, 8, 1], [2, 1 << 60, 1, -7], [-1, 0, 0, 0]], dtype=np.int64)
    np.testing.assert_array_equal(conv_native._CoordinateLookup(coords).lookup(queries), [2, 1, -1, -1])
    np.testing.assert_array_equal(conv_native._CoordinateLookup(coords[:0]).lookup(queries), [-1] * 4)


def _original_pairs(coords, kernel, stride, spatial=None, inverse=False):
    """Small loop oracle for the native definitions, including last duplicate wins."""
    import itertools
    rows = [tuple(row) for row in coords.tolist()]
    coord_map = {row: i for i, row in enumerate(rows)}
    offsets = list(itertools.product(*(range(k) for k in kernel)))
    candidates = []
    for offset in offsets:
        shifted = []
        for b, *xyz in rows:
            result = tuple(v * s + k - size // 2 if inverse else v // s + k - size // 2
                           for v, s, k, size in zip(xyz, stride, offset, kernel))
            if inverse and min(result) < 0:
                continue
            if spatial is not None and any(v < 0 or v >= limit for v, limit in zip(result, spatial)):
                continue
            shifted.append((b, *result))
        candidates.append(shifted)
    output = sorted(set(row for chunk in candidates for row in chunk))
    output_map = {row: i for i, row in enumerate(output)}
    pairs = []
    for offset in offsets:
        src, dst = [], []
        if inverse:
            for di, (b, *xyz) in enumerate(output):
                shifted = [v - k + size // 2 for v, k, size in zip(xyz, offset, kernel)]
                if any(v % s for v, s in zip(shifted, stride)):
                    continue
                row = (b, *(v // s for v, s in zip(shifted, stride)))
                if row in coord_map:
                    src.append(coord_map[row]); dst.append(di)
        else:
            for row in rows:
                b, *xyz = row
                shifted = (b, *(v // s + k - size // 2 for v, s, k, size in zip(xyz, stride, offset, kernel)))
                if shifted in output_map:
                    src.append(coord_map[row]); dst.append(output_map[shifted])
        pairs.append((src, dst))
    return output, pairs


@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('spatial', [None, [3, 4, 2]])
def test_vectorized_expansion_matches_original_unsorted_multibatch_boundaries(inverse, spatial):
    import numpy as np
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv import conv_native
    assert hasattr(conv_native, '_expanded_coordinate_pairs'), 'The vectorized expansion is missing'
    coords = np.array([[1, 2, 0, 1], [0, 0, 0, 0], [0, 5, 1, 3], [1, 2, 0, 1], [0, -2, 2, 0]], dtype=np.int64)
    kernel, stride = (3, 2, 1), (2, 3, 2)
    # Inverse ignores upper spatial bounds in the existing implementation.
    bounds = None if inverse else spatial
    expected_coords, expected_pairs = _original_pairs(coords, kernel, stride, bounds, inverse)
    actual_coords, actual_pairs = conv_native._expanded_coordinate_pairs(coords, kernel, stride, bounds, inverse=inverse)
    np.testing.assert_array_equal(actual_coords, expected_coords)
    for actual, expected in zip(actual_pairs, expected_pairs):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
    empty_coords, empty_pairs = conv_native._expanded_coordinate_pairs(coords[:0], kernel, stride, bounds, inverse=inverse)
    assert empty_coords.shape == (0, 4)
    assert len(empty_pairs) == 6
    assert all(len(src) == len(dst) == 0 for src, dst in empty_pairs)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_native_expansion_numerics_match_original_pair_order(device, inverse, empty):
    import numpy as np
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv.conv_native import SparseInverseConv3d
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv.native_backend import NativeSparseData
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Requires MPS')
    coords = torch.tensor([[1, 2, 0, 1], [0, 0, 0, 0], [0, 5, 1, 3], [1, 2, 0, 1], [0, 0, 2, 0]], dtype=torch.long)
    if empty:
        coords = coords[:0]
    g = torch.Generator().manual_seed(83)
    feats = torch.randn(len(coords), 3, generator=g).to(device)
    weight = torch.randn(3, 3, 3, 3, 2, generator=g).to(device)
    bias = torch.randn(2, generator=g).to(device)
    stride = (2, 3, 2)
    bounds = None if inverse else [3, 2, 3]
    expected_coords, pairs = _original_pairs(coords.numpy(), (3, 3, 3), stride, bounds, inverse)
    expected = torch.zeros(len(expected_coords), 2, device=device)
    oracle_weight = weight.permute(0, 1, 2, 4, 3).contiguous().permute(0, 1, 2, 4, 3) if inverse else weight
    for W, (src, dst) in zip(oracle_weight.reshape(27, 3, 2), pairs):
        if src:
            expected.index_add_(0, torch.tensor(dst, device=device), feats[torch.tensor(src, device=device)] @ W)
    if len(expected):
        expected += bias
    expected_spatial = (np.array(expected_coords)[:, 1:].max(axis=0) + 1).tolist() if expected_coords else [0, 0, 0]
    if inverse:
        module = SparseInverseConv3d(3, 2, 3, stride=stride).to(device)
        module.weight.data.copy_(weight.permute(0, 1, 2, 4, 3))
        module.bias.data.copy_(bias)
        data = NativeSparseData(feats, coords.to(device).to(torch.int32), [6, 3, 4], 2)
        x = SparseTensor(data, shape=torch.Size([2, 3]), scale=stride)
        x.register_spatial_cache(f'conv_{stride}_unsorted_data', x.data)
        x.register_spatial_cache(f'conv_{stride}_sort_bwd', torch.arange(len(coords), device=device))
        out = module(x)
        actual, actual_coords, actual_spatial = out.feats, out.coords, out.data.spatial_shape
        assert out._scale == (1, 1, 1)
        assert out._spatial_cache is x._spatial_cache
    else:
        actual, actual_coords, actual_spatial = _sparse_forward(feats, coords.to(device), weight, bias, stride, 1, [5, 4, 5], 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [tuple(row) for row in actual_coords.cpu().tolist()] == expected_coords
    assert actual_spatial == expected_spatial


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_inverse_matches_dense_transposed_convolution(device):
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv.conv_native import SparseInverseConv3d
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Requires MPS')
    coords = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 0], [1, 2, 1, 0]], dtype=torch.long)
    g = torch.Generator().manual_seed(92)
    feats = torch.randn(3, 2, generator=g)
    weight = torch.randn(3, 3, 3, 4, 2, generator=g)
    bias = torch.randn(4, generator=g)
    dense = torch.zeros(2, 2, 3, 2, 3)
    for (b, z, y, x), feat in zip(coords.tolist(), feats):
        dense[b, :, z, y, x] = feat
    dense_output = F.conv_transpose3d(dense, weight.permute(4, 3, 0, 1, 2), bias, stride=2, padding=1, output_padding=1)
    module = SparseInverseConv3d(2, 4, 3, stride=2).to(device)
    module.weight.data.copy_(weight); module.bias.data.copy_(bias)
    x = SparseTensor(feats.to(device), coords.to(device), scale=(2, 2, 2))
    x.register_spatial_cache('conv_(2, 2, 2)_unsorted_data', x.data)
    x.register_spatial_cache('conv_(2, 2, 2)_sort_bwd', torch.arange(3, device=device))
    actual = module(x)
    expected = torch.stack([dense_output[b, :, z, y, x] for b, z, y, x in actual.coords.cpu().tolist()])
    torch.testing.assert_close(actual.feats.cpu(), expected, rtol=1e-4, atol=1e-5)


def test_coordinate_lookup_rejects_outside_bounds_without_packed_key_alias():
    import numpy as np
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv import conv_native
    coords = np.array([[0, 0, 0, 0], [0, 0, 1, 0]], dtype=np.int64)
    lookup = conv_native._CoordinateLookup(coords)
    # The fast keys must use one integer per sparse row, not a spatial grid.
    assert lookup.keys.dtype == np.dtype(np.int64), 'Compact coordinate keys are missing'
    queries = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.int64)
    np.testing.assert_array_equal(lookup.lookup(queries), [-1, -1, -1, 1])


def test_coordinate_lookup_falls_back_when_coordinate_volume_exceeds_int64():
    import numpy as np
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv import conv_native
    coords = np.array([[0, -(1 << 62), 0, 0], [2, 1 << 62, 2, 3]], dtype=np.int64)
    lookup = conv_native._CoordinateLookup(coords)
    assert lookup.keys.dtype.fields is not None
    np.testing.assert_array_equal(lookup.lookup(coords[::-1]), [1, 0])


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_submanifold_preserves_duplicate_lookup_and_cache_kernel_invalidation(device):
    import itertools
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Requires MPS')
    coords = torch.tensor([[1, 2, 0, 1], [0, 0, 0, 0], [1, 2, 0, 1], [0, 0, 0, 1]], device=device)
    feats = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
    cache = {}
    for kernel in [(1, 1, 1), (3, 2, 1), (1, 1, 1)]:
        weight = torch.arange(2 * 3 * kernel[0] * kernel[1] * kernel[2], dtype=torch.float32, device=device).reshape(*kernel, 2, 3) / 16
        bias = torch.tensor([0.25, -0.5, 1.], device=device)
        rows = [tuple(row) for row in coords.cpu().tolist()]
        coord_map = {row: i for i, row in enumerate(rows)}
        expected = torch.zeros(4, 3, device=device)
        for offset in itertools.product(*(range(k) for k in kernel)):
            src, neighbors = [], []
            for i, (b, *xyz) in enumerate(rows):
                row = (b, *(value + k - size // 2 for value, k, size in zip(xyz, offset, kernel)))
                if row in coord_map:
                    src.append(i); neighbors.append(coord_map[row])
            if src:
                expected.index_add_(0, torch.tensor(src, device=device), feats[torch.tensor(neighbors, device=device)] @ weight[offset])
        expected += bias
        actual = _subm_forward(feats, coords, weight, bias, neighbor_cache=cache)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert cache['entry'][0] == kernel
        prior_pairs = cache['entry'][3]
        _subm_forward(feats * 2, coords.clone(), weight, bias, neighbor_cache=cache)
        assert cache['entry'][3] is prior_pairs


def test_submanifold_neighbor_maps_are_shared_across_layers(monkeypatch):
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv import conv_native
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
    coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]], dtype=torch.int32)
    feats = torch.randn(3, 2)
    layers = [conv_native.SparseConv3d(2, 3, 3).eval() for _ in range(2)]
    x = SparseTensor(feats, coords)
    expected = [conv_native._subm_forward(feats * scale, x.coords.long(), layer.weight, layer.bias)
                for layer, scale in zip(layers, [1, 2])]
    builds = []
    original = conv_native._CoordinateLookup
    def lookup(indices):
        builds.append(len(indices))
        return original(indices)
    monkeypatch.setattr(conv_native, '_CoordinateLookup', lookup)
    with torch.no_grad():
        for layer, scale, reference in zip(layers, [1, 2], expected):
            actual = layer(x.replace(feats * scale))
            torch.testing.assert_close(actual.feats, reference, rtol=0, atol=0)
            assert actual._spatial_cache is x._spatial_cache
    assert builds == [3], 'Layers sharing coordinates should build the neighbor map once'


@pytest.mark.parametrize('inference', [False, True])
def test_shared_neighbor_maps_validate_coordinate_changes_and_kernel(inference):
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv import conv_native
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
    with torch.inference_mode(inference), torch.no_grad():
        x = SparseTensor(torch.randn(3, 2), torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]]))
        layers = [conv_native.SparseConv3d(2, 3, 3).eval(), conv_native.SparseConv3d(2, 3, 1).eval()]
        for layer in layers:
            layer(x)
        for coords in [x.coords[[2, 0, 1]], x.coords + torch.tensor([0, 2, 0, 0]), x.coords]:
            current = x.replace(x.feats * 2, coords=coords)
            for layer in layers:
                expected = conv_native._subm_forward(current.feats, current.coords.long(), layer.weight, layer.bias)
                torch.testing.assert_close(layer(current).feats, expected, rtol=0, atol=0)
        x.coords[0, 3] = 3
        for layer in layers:
            expected = conv_native._subm_forward(x.feats, x.coords.long(), layer.weight, layer.bias)
            torch.testing.assert_close(layer(x).feats, expected, rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_shared_neighbor_maps_handle_device_and_scale_transitions(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Requires MPS')
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv import conv_native
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
    shared = {}
    coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
    feats = torch.randn(3, 2)
    fine = SparseTensor(feats, coords, scale=(1, 1, 1), spatial_cache=shared)
    coarse = SparseTensor(feats * 2, coords + torch.tensor([0, 0, 2, 0]),
                          scale=(2, 2, 2), spatial_cache=shared)
    layer = conv_native.SparseConv3d(2, 3, 3).eval()
    with torch.no_grad():
        for target in ['cpu', device, 'cpu']:
            layer.to(target)
            for source in [fine, coarse, fine]:
                x = source.to(target)
                expected = conv_native._subm_forward(x.feats, x.coords.long(), layer.weight, layer.bias)
                actual = layer(x)
                torch.testing.assert_close(actual.feats, expected, rtol=0, atol=0)
                assert actual.device == x.device
                assert actual._scale == source._scale
                assert actual._spatial_cache is shared
