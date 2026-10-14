import contextlib
import importlib
import pytest
import torch
from sam3d_objects.model.backbone.tdfy_dit.modules.attention.modules import MultiHeadAttention
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.modules import SparseMultiHeadAttention
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor

PATH = 'sam3d_objects.model.backbone.tdfy_dit.modules.attention_cache'
def api():
    try:
        return importlib.import_module(PATH)
    except ModuleNotFoundError:
        return None

def scope():
    return api().attention_cache_scope() if api() else contextlib.nullcontext()
def branch(c, zero=False):
    return api().conditioning_branch(c, zero=zero) if api() else contextlib.nullcontext()

@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('zero', [False, True])
def test_exact_reuse_alternating_branches(sparse, zero):
    torch.manual_seed(1)
    cls = SparseMultiHeadAttention if sparse else MultiHeadAttention
    m = cls(8, 2, type='cross').eval()
    c = torch.randn(1, 4, 8)
    x1, x2 = torch.randn(1, 3, 8), torch.randn(1, 3, 8)
    def x(v, offset):
        return SparseTensor(v[0], torch.tensor([[0, offset+i, 0, 0] for i in range(3)])) if sparse else v
    calls = []
    hook = m.to_kv.register_forward_hook(lambda *a: calls.append(1))
    with torch.no_grad():
        expected = [m(x(v, i), c * 0 if zero else c) for i,v in enumerate([x1,x2])]
        calls.clear()
        with scope():
            for i,v in enumerate([x1,x2]):
                with branch(c, zero):
                    actual = m(x(v, i), c * 0 if zero else c)
                torch.testing.assert_close(actual.feats if sparse else actual, expected[i].feats if sparse else expected[i], rtol=0, atol=0)
                if sparse:
                    assert actual.coords[0,1] == i
        assert len(calls) == 1
    hook.remove()

@pytest.mark.parametrize('norm', [False,True])
def test_conditional_q_fresh_and_parameter_condition_invalidation(norm):
    m = MultiHeadAttention(8,2,type='cross',qk_rms_norm=norm).eval()
    c, x = torch.randn(1,4,8), torch.randn(1,3,8)
    calls=[]
    m.to_kv.register_forward_hook(lambda *a: calls.append(1))
    with torch.no_grad(), scope():
        for _ in range(2):
            with branch(c): m(x,c)
        assert len(calls)==1
        c.add_(1)
        with branch(c): m(x,c)
        m.to_kv.weight.add_(1)
        with branch(c): m(x,c)
        assert len(calls)==3

@pytest.mark.parametrize('training,grad', [(True,False),(False,True)])
def test_unsafe_execution_bypasses(training, grad):
    m = MultiHeadAttention(8,2,type='cross').train(training)
    c,x=torch.randn(1,4,8),torch.randn(1,3,8)
    calls=[]
    m.to_kv.register_forward_hook(lambda *a: calls.append(1))
    with torch.set_grad_enabled(grad), scope(), branch(c):
        m(x,c);m(x,c)
    assert len(calls)==2

def test_scope_cleanup_nested_exception_and_escape_hatch(monkeypatch):
    a=api()
    assert a is not None, 'Production run cache API required'
    m=MultiHeadAttention(8,2,type='cross').eval()
    c,x=torch.randn(1,4,8),torch.randn(1,3,8)
    calls=[]
    m.to_kv.register_forward_hook(lambda *args:calls.append(1))
    with torch.no_grad(), a.attention_cache_scope() as outer, branch(c):
        expected=m(x,c)
        assert m in outer.entries
        with pytest.raises(ValueError):
            with a.attention_cache_scope() as inner, branch(c):
                torch.testing.assert_close(m(x,c),expected,rtol=0,atol=0)
                assert m in inner.entries
                assert len(calls)==2
                raise ValueError()
        assert not inner.entries
        assert m in outer.entries
        torch.testing.assert_close(m(x,c),expected,rtol=0,atol=0)
        assert len(calls)==2
    assert not outer.entries
    assert m.to_kv.weight is not None
    monkeypatch.setenv('SAM3D_ATTENTION_CACHE','0')
    with a.attention_cache_scope() as disabled:
        assert disabled is None

@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('norm',[False,True])
def test_zero_outputs_exact_across_queries(seed,norm):
    torch.manual_seed(seed)
    m=MultiHeadAttention(16,4,type='cross',qk_rms_norm=norm).eval()
    c=torch.randn(1,7,16)
    xs=[torch.randn(1,9,16) for _ in range(4)]
    with torch.no_grad():
        expected=[m(x,c*0) for x in xs]
        with scope(),branch(c,True):
            for x,e in zip(xs,expected):
                torch.testing.assert_close(m(x,c*0),e,rtol=0,atol=0)

def test_alternation_two_bounded_slots_weak_ownership_and_inference_tensors():
    import gc, weakref
    m=MultiHeadAttention(8,2,type='cross').eval()
    calls=[]
    m.to_kv.register_forward_hook(lambda *a: calls.append(1))
    with torch.inference_mode(),scope() as cache:
        c=torch.randn(1,4,8);x=torch.randn(1,3,8)
        for _ in range(3):
            for zero in [False,True]:
                with branch(c,zero):m(x,c*0 if zero else c)
        assert len(calls)==2
        assert len(cache.entries[m])==2
        ref=weakref.ref(m);del m;gc.collect()
        assert ref() is None
        assert len(cache.entries)==0

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_dtype_and_shape_changes_invalidate(dtype):
    m=MultiHeadAttention(8,2,type='cross').eval().to(dtype)
    c=torch.randn(1,4,8,dtype=dtype)
    calls=[];m.to_kv.register_forward_hook(lambda *a:calls.append(1))
    with torch.no_grad(),scope(),branch(c):
        for n in [3,3,5,5]:m(torch.randn(1,n,8,dtype=dtype),c)
        assert len(calls)==2

def test_unsupported_branch_bypasses():
    m=MultiHeadAttention(8,2,type='cross').eval()
    c,x=torch.randn(1,4,8),torch.randn(1,3,8)
    calls=[];m.to_kv.register_forward_hook(lambda *a:calls.append(1))
    with torch.no_grad(),scope(),api().conditioning_branch(c,supported=False):
        m(x,c);m(x,c)
    assert len(calls)==2

@pytest.mark.parametrize('sparse', [False,True])
@pytest.mark.parametrize('tokens', [1,7])
def test_zero_branch_reuses_kv_with_fresh_queries_and_outputs(sparse,tokens):
    cls=SparseMultiHeadAttention if sparse else MultiHeadAttention
    m=cls(8,2,type='cross').eval()
    c=torch.zeros(1,tokens,8)
    xs=[torch.randn(1,3,8) for _ in range(2)]
    if sparse:
        xs=[SparseTensor(x[0],torch.tensor([[0,i+offset,0,0] for i in range(3)])) for offset,x in enumerate(xs)]
        for i,x in enumerate(xs):x.register_spatial_cache('current',i)
    calls=[];m.to_out.register_forward_hook(lambda *a:calls.append(1))
    q_calls=[];m.to_q.register_forward_hook(lambda *a:q_calls.append(1))
    kv_calls=[];m.to_kv.register_forward_hook(lambda *a:kv_calls.append(1))
    with torch.no_grad():
        expected=[m(x,c) for x in xs]
        calls.clear();q_calls.clear();kv_calls.clear()
        with scope() as cache,branch(c,True):
            for i,x in enumerate(xs):
                actual=m(x,c)
                torch.testing.assert_close(actual.feats if sparse else actual,expected[i].feats if sparse else expected[i],rtol=0,atol=0)
                if sparse:
                    assert torch.equal(actual.coords,x.coords)
                    assert actual._spatial_cache is x._spatial_cache
                    assert actual.get_spatial_cache('current')==i
            assert 'kv' in cache.entries[m][True][1]
            assert 'output' not in cache.entries[m][True][1]
    assert len(calls)==len(q_calls)==2
    assert len(kv_calls)==1

@pytest.mark.parametrize('sparse',[False,True])
@pytest.mark.parametrize('force_zero,cfg,custom',[(True,False,False),(True,True,False),(False,True,False),(True,False,True)])
def test_real_wrapper_preserves_raw_condition_identity_before_cast(sparse,force_zero,cfg,custom):
    from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow import SLatFlowModelTdfyWrapper
    from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper
    class Mapping(torch.nn.Module):
        def to_input(self,x):return x
        def to_output(self,x):return x
    args=dict(in_channels=8,model_channels=8,cond_channels=8,out_channels=8,num_blocks=1,num_heads=2,qk_rms_norm=True,force_zeros_cond=force_zero)
    if custom:args['condition_embedder']=torch.nn.Identity()
    if sparse:
        m=SLatFlowModelTdfyWrapper(resolution=8,patch_size=2,num_io_res_blocks=1,io_block_channels=[8],**args).eval()
        attn=m.blocks[0].cross_attn
    else:
        m=SparseStructureFlowTdfyWrapper(latent_mapping={'shape':Mapping(),'pose':Mapping()},**args).eval()
        attn=m.blocks[0].cross_attn['shape']
    c=torch.randn(1,4,8,dtype=torch.float64);x=torch.randn(1,3,8);t=torch.ones(1)
    coords=[[0,i,0,0] for i in range(3)]
    def run():return m(x,t,c,coords,cfg=cfg) if sparse else m({'shape':x,'pose':x},t,c,cfg=cfg)
    calls=[];attn.to_kv.register_forward_hook(lambda *a:calls.append(1))
    with torch.no_grad():
        expected=run()
        calls.clear()
        with scope():
            for _ in range(2):
                result=run()
                if not sparse:result,expect=result['shape'],expected['shape']
                else:expect=expected
                torch.testing.assert_close(result,expect,rtol=0,atol=0)
    assert len(calls)==(2 if custom or (cfg and not force_zero) else 1)

def test_autocast_changes_invalidate_projection_dtype():
    m=MultiHeadAttention(8,2,type='cross').eval()
    c,x=torch.randn(1,4,8),torch.randn(1,3,8)
    with torch.no_grad():
        baseline=m(x,c)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            autocast_baseline=m(x,c)
        with scope(),branch(c):
            torch.testing.assert_close(m(x,c),baseline,rtol=0,atol=0)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                torch.testing.assert_close(m(x,c),autocast_baseline,rtol=0,atol=0)

def test_versioned_cast_context_change_invalidates():
    m=MultiHeadAttention(8,2,type='cross').eval()
    raw=torch.randn(1,4,8,dtype=torch.float64)
    c=raw.float();x=torch.randn(1,3,8)
    calls=[];m.to_kv.register_forward_hook(lambda *a:calls.append(1))
    with torch.no_grad(),scope(),branch(raw):
        m(x,c)
        c.add_(1)
        expected=m(x,c)  # must project changed version
    assert len(calls)==2

@pytest.mark.parametrize('sparse',[False,True])
def test_biased_single_token_nonfinite_query_matches_uncached(sparse):
    torch.manual_seed(73)
    cls=SparseMultiHeadAttention if sparse else MultiHeadAttention
    m=cls(8,2,type='cross',ctx_channels=8,qkv_bias=True).eval()
    c=torch.zeros(1,1,8)
    good=torch.randn(1,4,8)
    bad=good.clone();bad[0,0,0]=float('nan')
    def query(feats,offset):
        if not sparse:return feats
        result=SparseTensor(feats[0],torch.tensor([[0,i+offset,0,0] for i in range(4)]))
        result.register_spatial_cache('current',offset)
        return result
    x,y=query(good,0),query(bad,1)
    with torch.no_grad():
        expected=m(y,c)
        with scope(),branch(c,True):
            m(x,c)
            actual=m(y,c)
        torch.testing.assert_close(actual.feats if sparse else actual,expected.feats if sparse else expected,rtol=0,atol=0,equal_nan=True)
        if sparse:
            assert torch.equal(actual.coords,y.coords)
            assert actual._spatial_cache is y._spatial_cache
