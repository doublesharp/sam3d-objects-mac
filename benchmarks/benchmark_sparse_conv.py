"""Compare native sparse convolution with the pinned upstream implementation.

Run from the repository root with SPARSE_BACKEND=native. Both implementations
receive identical inputs; this measures one CPU convolution, not an entire run.
"""
import os
os.environ.setdefault('SPARSE_BACKEND', 'native')
os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
import argparse
import json
import statistics
import subprocess
import time
import torch
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv import conv_native

parser = argparse.ArgumentParser()
parser.add_argument('--base', default='2ed940d1edbc86d3aae8c96dcbde3e8437b18f07')
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
source = subprocess.check_output([
    'git', 'show', args.base + ':sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/conv_native.py',
], text=True)
namespace = {'__name__': conv_native.__name__, '__package__': conv_native.__package__}
exec(compile(source, '<upstream-conv>', 'exec'), namespace)
coords = torch.cartesian_prod(torch.arange(8), torch.arange(8), torch.arange(8))
coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.long), coords], 1)
feats = torch.randn(len(coords), 4, generator=torch.Generator().manual_seed(8))
weights = torch.randn(3, 3, 3, 4, 4, generator=torch.Generator().manual_seed(9))
results = {}
outputs = {}
for label, fn in [('upstream', namespace['_subm_forward']), ('optimized', conv_native._subm_forward)]:
    times = []
    with torch.inference_mode():
        for _ in range(args.repeats):
            start = time.perf_counter()
            outputs[label] = fn(feats, coords, weights, None)
            times.append(time.perf_counter() - start)
    results[label + '_seconds'] = statistics.median(times)
torch.testing.assert_close(outputs['upstream'], outputs['optimized'], rtol=0, atol=0)
results.update(voxels=len(coords), channels=4, repeats=args.repeats,
               speedup=results['upstream_seconds'] / results['optimized_seconds'],
               output='bitwise identical')
print(json.dumps(results, indent=2))
