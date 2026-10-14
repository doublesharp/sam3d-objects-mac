# Copyright (c) Meta Platforms, Inc. and affiliates.
# Pure-PyTorch implementations of SubMConv3d / SparseConv3d / SparseInverseConv3d
# for the "native" backend (CPU / MPS, no spconv / CUDA required).
#
# Weight layout mirrors spconv: [K, K, K, in_channels, out_channels]
# Coordinate layout: [N, 4] — (batch, z, y, x)
#
# Coordinate indexing uses sparse NumPy keys and keeps kernel accumulation
# order identical to the native reference implementation.

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from .native_backend import NativeSparseData

# Lazy import to avoid circular dependency (sparse.__init__ imports conv.__init__)
def _get_SparseTensor():
    from ..basic import SparseTensor
    return SparseTensor

def _get_DEBUG():
    from .. import DEBUG
    return DEBUG

__all__ = ["SparseConv3d", "SparseInverseConv3d"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COORD_DTYPE = np.dtype([(name, np.int64) for name in ("b", "z", "y", "x")])


def _coordinate_keys(coords: np.ndarray) -> np.ndarray:
    """Lexicographic keys without packed-key overflow or a dense spatial grid."""
    return np.ascontiguousarray(coords, dtype=np.int64).view(_COORD_DTYPE).reshape(-1)


class _CoordinateLookup:
    """Sparse sorted lookup retaining the original dict's last-duplicate rule."""
    def __init__(self, coords: np.ndarray):
        coords = np.asarray(coords, dtype=np.int64)
        self.bounds = None
        if len(coords):
            lower = coords.min(axis=0)
            upper = coords.max(axis=0)
            extents = [int(high) - int(low) + 1 for low, high in zip(lower, upper)]
            volume = 1
            for extent in extents:
                volume *= extent
            if volume <= np.iinfo(np.int64).max:
                self.bounds = (lower, upper, extents)
        keys = self._encode(coords)
        self.order = np.argsort(keys, kind="stable")
        self.keys = keys[self.order]

    def _encode(self, coords: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return _coordinate_keys(coords)
        lower, _, extents = self.bounds
        shifted = coords - lower
        keys = shifted[:, 0].copy()
        for axis in range(1, 4):
            keys *= extents[axis]
            keys += shifted[:, axis]
        return keys

    def lookup(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.int64)
        result = np.full(len(coords), -1, dtype=np.int64)
        if len(self.keys):
            if self.bounds is None:
                rows = np.arange(len(coords))
            else:
                lower, upper, _ = self.bounds
                # Reject out-of-bounds queries before encoding to prevent aliasing.
                rows = np.flatnonzero(np.all((coords >= lower) & (coords <= upper), axis=1))
            queries = self._encode(coords[rows])
            positions = np.searchsorted(self.keys, queries, side="right") - 1
            valid = positions >= 0
            valid[valid] &= self.keys[positions[valid]] == queries[valid]
            result[rows[valid]] = self.order[positions[valid]]
        return result


def _expanded_coordinate_pairs(
    coords: np.ndarray,
    kernel_shape: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    spatial_bounds=None,
    *,
    inverse=False,
):
    """Sorted sparse output union and per-kernel source/destination row pairs.

    Temporary storage scales with input voxels times kernel volume, never with
    spatial volume. Strided pairs retain input order and duplicate contributions;
    inverse pairs retain sorted output order with one contribution per output.
    """
    coords = np.asarray(coords, dtype=np.int64)
    base = coords.copy()
    if inverse:
        base[:, 1:] *= np.asarray(stride)
    else:
        base[:, 1:] //= np.asarray(stride)
    sources = _CoordinateLookup(coords).lookup(coords)
    chunks, source_chunks = [], []
    Kz, Ky, Kx = kernel_shape
    for kz in range(Kz):
        for ky in range(Ky):
            for kx in range(Kx):
                shifted = base.copy()
                shifted[:, 1:] += (kz - Kz // 2, ky - Ky // 2, kx - Kx // 2)
                valid = np.ones(len(coords), dtype=bool)
                if inverse or spatial_bounds is not None:
                    valid &= np.all(shifted[:, 1:] >= 0, axis=1)
                if spatial_bounds is not None:
                    valid &= np.all(shifted[:, 1:] < np.asarray(spatial_bounds), axis=1)
                chunks.append(shifted[valid])
                source_chunks.append(sources[valid])
    all_coords = np.concatenate(chunks)
    union = _CoordinateLookup(all_coords)
    first = np.ones(len(union.keys), dtype=bool)
    first[1:] = union.keys[1:] != union.keys[:-1]
    output_coords = all_coords[union.order[first]]
    destinations = np.empty(len(all_coords), dtype=np.int64)
    destinations[union.order] = np.cumsum(first) - 1
    pairs, offset = [], 0
    for chunk, src in zip(chunks, source_chunks):
        dst = destinations[offset:offset + len(chunk)]
        offset += len(chunk)
        if inverse:
            # np.unique sorts destinations just like the old fine-coordinate loop.
            dst, first = np.unique(dst, return_index=True)
            src = src[first]
        pairs.append((src, dst))
    return output_coords, pairs


def _subm_forward(
    feats: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    neighbor_cache: Optional[dict] = None,
) -> torch.Tensor:
    """Submanifold convolution with a bounded, per-module neighbor-map cache."""
    Kz, Ky, Kx, Cin, Cout = weight.shape
    kernel_shape = (Kz, Ky, Kx)
    cpu_idx = indices.cpu().numpy()
    entry = neighbor_cache.get("entry") if neighbor_cache is not None else None
    # Compare values, not tensor identity: diffusion reconstructs identical
    # coordinate tensors every step. Snapshotting also detects in-place edits.
    if (entry is not None and entry[0] == kernel_shape and entry[1] == feats.device
            and np.array_equal(entry[2], cpu_idx)):
        pairs = entry[3]
    else:
        lookup = _CoordinateLookup(cpu_idx)
        pairs = []
        for kz in range(Kz):
            for ky in range(Ky):
                for kx in range(Kx):
                    shifted = cpu_idx.copy()
                    shifted[:, 1:] += (kz - Kz // 2, ky - Ky // 2, kx - Kx // 2)
                    neighbors = lookup.lookup(shifted)
                    src_idx = np.flatnonzero(neighbors >= 0)
                    pairs.append((
                        torch.tensor(src_idx, dtype=torch.long, device=feats.device),
                        torch.tensor(neighbors[src_idx], dtype=torch.long, device=feats.device),
                    ))
        if neighbor_cache is not None:
            # One atomic replacement; concurrent callers retain their local entry.
            neighbor_cache["entry"] = (kernel_shape, feats.device, cpu_idx.copy(), pairs)

    out = torch.zeros(feats.shape[0], Cout, dtype=feats.dtype, device=feats.device)
    pair_index = 0
    for kz in range(Kz):
        for ky in range(Ky):
            for kx in range(Kx):
                src_t, nb_t = pairs[pair_index]
                pair_index += 1
                if src_t.numel():
                    contrib = (feats[nb_t] @ weight[kz, ky, kx]).to(out.dtype)
                    out.index_add_(0, src_t, contrib)
    if bias is not None:
        out = out + bias
    return out


def _sparse_forward(
    feats: torch.Tensor,       # [N, Cin]
    indices: torch.Tensor,     # [N, 4] int64  (b,z,y,x)
    weight: torch.Tensor,      # [Kz, Ky, Kx, Cin, Cout]
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int, int],
    padding: int,
    spatial_shape: list,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Strided sparse conv.  Returns (out_feats, out_indices, out_spatial_shape).
    out_indices are int64 sorted by batch.
    """
    Kz, Ky, Kx, Cin, Cout = weight.shape
    cpu_idx = indices.cpu().numpy()
    bounds = [(size + padding * 2 - kernel) // step + 1
              for size, kernel, step in zip(spatial_shape, (Kz, Ky, Kx), stride)] if spatial_shape else None
    out_coords, pairs = _expanded_coordinate_pairs(cpu_idx, (Kz, Ky, Kx), stride, bounds)
    if not len(out_coords):
        empty_idx = torch.zeros((0, 4), dtype=torch.int64, device=feats.device)
        empty_f = torch.zeros((0, Cout), dtype=feats.dtype, device=feats.device)
        return empty_f, empty_idx, [0, 0, 0]

    out_idx_t = torch.tensor(out_coords, dtype=torch.int64, device=feats.device)
    out_feats = torch.zeros(len(out_coords), Cout, dtype=feats.dtype, device=feats.device)
    pair_index = 0
    for kz in range(Kz):
        for ky in range(Ky):
            for kx in range(Kx):
                src_idx, dst_idx = pairs[pair_index]
                pair_index += 1
                if len(src_idx):
                    src_t = torch.tensor(src_idx, dtype=torch.long, device=feats.device)
                    dst_t = torch.tensor(dst_idx, dtype=torch.long, device=feats.device)
                    contrib = (feats[src_t] @ weight[kz, ky, kx]).to(out_feats.dtype)
                    out_feats.index_add_(0, dst_t, contrib)

    if bias is not None:
        out_feats = out_feats + bias

    out_spatial = (out_coords[:, 1:].max(axis=0) + 1).tolist()

    return out_feats, out_idx_t, out_spatial


# ---------------------------------------------------------------------------
# Module wrappers (mirror conv_spconv.py interface)
# ---------------------------------------------------------------------------

class SparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        dilation: int = 1,
        padding=None,
        bias: bool = True,
        indice_key=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.stride = (
            tuple(stride) if isinstance(stride, (list, tuple))
            else (stride, stride, stride)
        )
        self.padding = padding if padding is not None else 0
        self._is_subm = all(s == 1 for s in self.stride) and padding is None
        self._subm_neighbor_cache = {}

        K = kernel_size
        # Weight: [K, K, K, in_ch, out_ch]  (matches spconv checkpoint layout)
        self.weight = nn.Parameter(torch.empty(K, K, K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight.view(-1, in_channels, out_channels).permute(2, 1, 0))

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        feats = x.feats.float()          # keep fp32 for stability; re-cast after
        indices = x.coords.long()        # [N,4] int64 for indexing

        if self._is_subm:
            cache_key = f"native_subm_neighbors_{self.kernel_size}"
            shared_cache = x.get_spatial_cache(cache_key)
            if shared_cache is None:
                shared_cache = self._subm_neighbor_cache
                x.register_spatial_cache(cache_key, shared_cache)
            self._subm_neighbor_cache = shared_cache
            out_feats = _subm_forward(feats, indices, self.weight.float(),
                                       self.bias.float() if self.bias is not None else None,
                                       neighbor_cache=self._subm_neighbor_cache)
            out_feats = out_feats.to(x.feats.dtype)
            new_data = NativeSparseData(out_feats, x.coords, x.data.spatial_shape, x.data.batch_size)
            new_data.indice_dict = x.data.indice_dict
            out = _get_SparseTensor()(new_data, shape=torch.Size([x.shape[0], self.out_channels]),
                               layout=x.layout, scale=x._scale,
                               spatial_cache=x._spatial_cache)
            return out

        # Strided conv
        pad = self.padding if isinstance(self.padding, int) else self.padding[0]
        out_feats, out_idx, out_spatial = _sparse_forward(
            feats, indices, self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.stride, pad,
            x.data.spatial_shape, x.data.batch_size,
        )
        out_feats = out_feats.to(x.feats.dtype)
        out_idx_i32 = out_idx.to(torch.int32)

        # Sort by batch (match spconv output layout)
        sort_fwd = out_idx_i32[:, 0].argsort()
        sort_bwd = torch.zeros_like(sort_fwd).scatter_(
            0, sort_fwd, torch.arange(sort_fwd.shape[0], device=sort_fwd.device))
        sorted_feats = out_feats[sort_fwd]
        sorted_idx = out_idx_i32[sort_fwd]

        unsorted_data = NativeSparseData(out_feats, out_idx_i32, out_spatial, x.data.batch_size)
        new_data = NativeSparseData(sorted_feats, sorted_idx, out_spatial, x.data.batch_size)
        new_data.indice_dict = x.data.indice_dict

        out = _get_SparseTensor()(
            new_data,
            shape=torch.Size([x.shape[0], self.out_channels]),
            layout=None,
            scale=tuple(s * stride for s, stride in zip(x._scale, self.stride)),
            spatial_cache=x._spatial_cache,
        )
        out.register_spatial_cache(f"conv_{self.stride}_unsorted_data", unsorted_data)
        out.register_spatial_cache(f"conv_{self.stride}_sort_bwd", sort_bwd)
        return out


class SparseInverseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        dilation: int = 1,
        bias: bool = True,
        indice_key=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.stride = (
            tuple(stride) if isinstance(stride, (list, tuple))
            else (stride, stride, stride)
        )

        K = kernel_size
        # Transposed weight reuses the same layout [K, K, K, out_ch, in_ch]
        # (inverse: out_ch of forward becomes in_ch of inverse)
        self.weight = nn.Parameter(torch.empty(K, K, K, out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight.view(-1, out_channels, in_channels).permute(2, 1, 0))

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        spatial_changed = any(s != 1 for s in self.stride)

        if spatial_changed:
            unsorted_data: NativeSparseData = x.get_spatial_cache(
                f"conv_{self.stride}_unsorted_data")
            sort_bwd = x.get_spatial_cache(f"conv_{self.stride}_sort_bwd")
            # Recover original (unsorted) order, then apply features
            data = unsorted_data.replace_feature(x.feats[sort_bwd])
            if _get_DEBUG():
                assert torch.equal(data.indices, x.coords[sort_bwd]), \
                    "SparseInverseConv3d: coord recovery mismatch"
        else:
            data = x.data

        # data.indices: [N,4] int32 — these are the *downsampled* coords.
        # We need to scatter back to the original (finer) resolution.
        # Recover original fine coords from unsorted_data's indice_dict is not
        # available in the native backend; instead we reconstruct by upsizing:
        # fine_coord = coarse_coord * stride + kernel_offset
        # and accumulating into a dense output of the original spatial shape.
        #
        # For the conv transpose semantics expected here, we use a simplified
        # scatter approach: for each coarse voxel, iterate kernel offsets,
        # compute fine voxel, and scatter-add weight contribution.

        coarse_idx = data.indices.long()   # [N, 4]
        feats = data.features.float()      # [N, Cin]
        Kz, Ky, Kx, Cout, Cin = self.weight.shape
        cpu_coarse = coarse_idx.cpu().numpy()
        fine_coords, pairs = _expanded_coordinate_pairs(
            cpu_coarse, (Kz, Ky, Kx), self.stride, inverse=True)
        if not len(fine_coords):
            empty_idx = torch.zeros((0, 4), dtype=torch.int32, device=x.feats.device)
            empty_f = torch.zeros((0, self.out_channels), dtype=x.feats.dtype, device=x.feats.device)
            new_data = NativeSparseData(empty_f, empty_idx, [0, 0, 0], x.data.batch_size)
            return _get_SparseTensor()(new_data,
                                shape=torch.Size([x.shape[0], self.out_channels]),
                                layout=None,
                                scale=tuple(s // stride for s, stride in zip(x._scale, self.stride)),
                                spatial_cache=x._spatial_cache)

        fine_idx_t = torch.tensor(fine_coords, dtype=torch.int64, device=x.feats.device)
        out_feats = torch.zeros(len(fine_coords), self.out_channels, dtype=feats.dtype, device=feats.device)
        pair_index = 0
        for kz in range(Kz):
            for ky in range(Ky):
                for kx in range(Kx):
                    src_idx, dst_idx = pairs[pair_index]
                    pair_index += 1
                    if len(src_idx):
                        src_t = torch.tensor(src_idx, dtype=torch.long, device=feats.device)
                        dst_t = torch.tensor(dst_idx, dtype=torch.long, device=feats.device)
                        contrib = (feats[src_t] @ self.weight[kz, ky, kx].T).to(out_feats.dtype)
                        out_feats.index_add_(0, dst_t, contrib)

        if self.bias is not None:
            out_feats = out_feats + self.bias.float()

        out_feats = out_feats.to(x.feats.dtype)
        fine_idx_i32 = fine_idx_t.to(torch.int32)

        fine_spatial = (fine_coords[:, 1:].max(axis=0) + 1).tolist()

        new_data = NativeSparseData(out_feats, fine_idx_i32, fine_spatial, x.data.batch_size)
        out = _get_SparseTensor()(
            new_data,
            shape=torch.Size([x.shape[0], self.out_channels]),
            layout=None,
            scale=tuple(s // stride for s, stride in zip(x._scale, self.stride)),
            spatial_cache=x._spatial_cache,
        )
        return out
