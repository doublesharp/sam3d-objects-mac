# SkinTokens CPU and MPS inference patch

This package preserves the inference port against upstream SkinTokens commit
`273b691d35989d71cd17ff2895fdc735097b92d1`. It includes SDPA attention,
an instance-local empty-history repetition guard, complete skin-token grammar,
explicit conditioning seeds, a local runner, rig validation, and CPU regressions.
Training behavior is outside this patch's scope.

Apply to a clean checkout of the pinned revision. Run these commands from the
SAM repository root. An existing prepared runtime does not need reapplication.

```sh
SAM3D_ROOT="$PWD"
git clone https://github.com/VAST-AI-Research/SkinTokens.git .cache/rigging/SkinTokens
git -C .cache/rigging/SkinTokens checkout 273b691d35989d71cd17ff2895fdc735097b92d1
git -C .cache/rigging/SkinTokens apply --check "$SAM3D_ROOT/integrations/skintokens/changes.patch"
git -C .cache/rigging/SkinTokens apply "$SAM3D_ROOT/integrations/skintokens/changes.patch"
```

Use a separate Python 3.11 environment at `.cache/rigging/runtime`, or configure
`SAM3D_RIG_PYTHON`. `requirements-runtime.txt` records the relevant package
versions in the tested runtime; it is not a complete dependency lock or evidence
of a fresh installation. Follow upstream checkpoint instructions to place
`grpo_1400.ckpt` under `experiments/articulation_xl_quantization_256_token_4/`.
The checkpoint and runtime are not included in Git. The SDPA port does not
require FlashAttention. Run from the patched SkinTokens root so its configs and
checkpoint paths resolve correctly.

CPU checks, using the prepared runtime's Python:

```sh
PYTHONPATH=. python test_generation_scores.py --cpu-only
PYTHONPATH=. python test_token_grammar.py
PYTHONPATH=. python test_rig_validation.py
PYTHONPATH=. python test_generation_configuration.py
PYTHONPATH=. python test_conditioning_seed.py
python run_local.py --help
```

The existing SAM integration consumes `run_local.py` and its result contract.
See [rigging](../../docs/rigging.md) for full-quality conditioning, resource
limits, exact skeleton overrides, and separate GLB candidate export. Direct
runner defaults are lightweight research settings; use the integrated launcher
for pipeline candidates. Structural validity does not establish complete anatomy
or acceptable deformation. Original assets must remain available for comparison.
Upstream code retains the license in `LICENSE.upstream`.
