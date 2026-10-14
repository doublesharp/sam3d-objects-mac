"""Bounded attention reuse for one immutable inference conditioning run."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
from weakref import WeakKeyDictionary
import torch

_run = ContextVar("sam3d_attention_run", default=None)
_branch = ContextVar("sam3d_attention_branch", default=None)

class RunCache:
    def __init__(self):
        self.entries = WeakKeyDictionary()

@contextmanager
def attention_cache_scope():
    cache = RunCache() if os.environ.get("SAM3D_ATTENTION_CACHE", "1") != "0" else None
    token = _run.set(cache)
    branch_token = _branch.set(None)
    try:
        yield cache
    finally:
        if cache is not None:
            cache.entries.clear()
        _branch.reset(branch_token)
        _run.reset(token)

def cached_attention_run(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with attention_cache_scope():
            return fn(*args, **kwargs)
    return wrapped

def tensor_key(t):
    try:
        version = t._version
    except RuntimeError:  # inference tensors are immutable within this run
        version = None
    return (id(t), version, tuple(t.shape), t.dtype, t.device)

@contextmanager
def conditioning_branch(raw, zero=False, supported=True):
    token = _branch.set((raw, bool(zero)) if supported and isinstance(raw, torch.Tensor) else None)
    try:
        yield
    finally:
        _branch.reset(token)

def cache_slot(module, x, context, supported=True):
    cache, branch = _run.get(), _branch.get()
    if cache is None or branch is None or module.training or torch.is_grad_enabled() or not supported:
        return None
    if not isinstance(context, torch.Tensor):
        return None
    raw, zero = branch
    feats = x if isinstance(x, torch.Tensor) else x.feats
    key = (tensor_key(raw), tensor_key(context)[1:],
           tuple(feats.shape), feats.dtype, feats.device,
           torch.is_autocast_enabled(feats.device.type),
           torch.get_autocast_dtype(feats.device.type),
           tuple(tensor_key(p) for p in module.parameters()))
    entries = cache.entries.setdefault(module, {})
    slot = entries.get(zero)
    if slot is None or slot[0] != key:
        slot = (key, {"zero": zero}, raw)  # retain identity, prevent allocator/id reuse
        entries[zero] = slot
    return slot[1]
