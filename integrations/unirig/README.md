# UniRig portable skeleton inference patch

This experimental package targets upstream UniRig commit
`6793c6640ff01c8fb389f3993434124bb43d2933`. It preserves deterministic CPU
farthest-point sampling, lazy imports for optional skinning dependencies, SDPA
attention, an instance-local empty-history repetition guard, strict checkpoint
loading, a skeleton runner, and CPU checks. It does not provide the PTv3 skinning
runtime or skin weights, and is not wired into the default SAM rig stage.

Apply to a clean checkout of the pinned revision, from the SAM repository root:

```sh
SAM3D_ROOT="$PWD"
git clone https://github.com/VAST-AI-Research/UniRig.git .cache/rigging/UniRig
git -C .cache/rigging/UniRig checkout 6793c6640ff01c8fb389f3993434124bb43d2933
git -C .cache/rigging/UniRig apply --check "$SAM3D_ROOT/integrations/unirig/changes.patch"
git -C .cache/rigging/UniRig apply "$SAM3D_ROOT/integrations/unirig/changes.patch"
```

Use a separate Python 3.11 runtime. `requirements-runtime.txt` records relevant
versions from the tested environment, not a complete dependency lock or a
verified clean installation. The skeleton path avoids FlashAttention,
`torch_cluster`, `spconv`, and `torch_scatter`. Skinning still needs its upstream
optional dependencies.

Download model files separately with the Hugging Face CLI. From the patched
UniRig root:

```sh
hf download VAST-AI/UniRig skeleton/articulation-xl_quantization_256/model.ckpt \
  --revision 36842e2b5947e9e60f89275b83208c8e74071c63 --local-dir experiments
hf download facebook/opt-350m config.json \
  --revision 08ab08cc4b72ff5593870b5d527cf4230323703c --local-dir models/opt-350m
mkdir -p ../unirig-evidence
PYTHONPATH=. python test_ar_portability.py
PYTHONPATH=. python test_fps_cpu.py
PYTHONPATH=. python test_strict_load.py
PYTHONPATH=. python test_ar_forward_cpu.py
python run_ar_local.py --help
```

The FPS check compares exact indices against an algorithm reference. It does
not execute the native `torch_cluster` binary. Load and forward checks require
the downloaded checkpoint and only establish parameter/shape/finite checks.
Use `run_ar_local.py --input /absolute/model.glb --output /absolute/new-directory`
from the patched root. A complete skeleton with a valid hierarchy remains an
anatomical candidate. It has no skin weights or automatic animation.
Upstream code retains the license in `LICENSE.upstream`.
