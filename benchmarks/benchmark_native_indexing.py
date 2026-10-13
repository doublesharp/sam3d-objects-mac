"""Compare cold/warm native sparse CPU kernels with a pinned implementation.

This measures convolution kernels, not generation or a whole mesh decode.
"""
import os
os.environ.setdefault('SPARSE_BACKEND', 'native')
os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
import torch
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv import conv_native
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='9d476de')
    parser.add_argument('--sizes', default='512,4096,16384')
    parser.add_argument('--channels', default='8,64')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source = subprocess.check_output(['git', 'show', args.base + ':sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/conv_native.py'], text=True, cwd=Path(__file__).resolve().parents[1])
    baseline = {'__name__': conv_native.__name__, '__package__': conv_native.__package__}
    exec(compile(source, '<baseline-conv>', 'exec'), baseline)
    results = []
    for size in map(int, args.sizes.split(',')):
        side = max(4, int((size * 4) ** (1 / 3)) + 1)
        g = torch.Generator().manual_seed(2026 + size)
        linear = torch.randperm(side ** 3, generator=g)[:size]
        xyz = torch.stack([linear // (side ** 2), linear // side % side, linear % side], 1)
        coords = torch.cat([torch.zeros(size, 1, dtype=torch.long), xyz], 1)
        for channels in map(int, args.channels.split(',')):
            feats = torch.randn(size, channels, generator=g)
            weight = torch.randn(3, 3, 3, channels, channels, generator=g)
            bias = torch.randn(channels, generator=g)
            x = SparseTensor(feats, coords)
            x.register_spatial_cache('conv_(2, 2, 2)_unsorted_data', x.data)
            x.register_spatial_cache('conv_(2, 2, 2)_sort_bwd', torch.arange(size))
            for operation in ['subm_cold', 'subm_warm', 'strided', 'inverse']:
                calls = []
                for label, namespace in [('baseline', baseline), ('candidate', vars(conv_native))]:
                    cache = {}
                    if operation == 'inverse':
                        module = namespace['SparseInverseConv3d'](channels, channels, 3, stride=2)
                        module.weight.data.copy_(weight)
                        module.bias.data.copy_(bias)
                        def call(m=module):
                            output = m(x)
                            return output.feats, output.coords, output.data.spatial_shape
                    elif operation == 'strided':
                        call = lambda ns=namespace: ns['_sparse_forward'](feats, coords, weight, bias, (2, 2, 2), 1, [side] * 3, 1)
                    else:
                        call = lambda ns=namespace, c=cache: (ns['_subm_forward'](feats, coords, weight, bias, neighbor_cache=c if operation == 'subm_warm' else None),)
                    calls.append((label, call))
                timings = {label: [] for label, _ in calls}
                outputs = {}
                with torch.inference_mode():
                    for label, call in calls:
                        call()
                    for repeat in range(args.repeats):
                        for label, call in calls[::1 if repeat % 2 == 0 else -1]:
                            start = time.perf_counter()
                            outputs[label] = call()
                            timings[label].append(time.perf_counter() - start)
                for a, b in zip(outputs['baseline'], outputs['candidate']):
                    if torch.is_tensor(a):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    else:
                        assert a == b
                entry = dict(voxels=size, channels=channels, side=side, operation=operation,
                             samples_seconds=timings, median_seconds={k: statistics.median(v) for k, v in timings.items()},
                             output='bitwise identical')
                entry['speedup'] = entry['median_seconds']['baseline'] / entry['median_seconds']['candidate']
                results.append(entry)
                print(json.dumps(entry), flush=True)
    evidence = dict(base=args.base, threads=args.threads, repeats=args.repeats,
                    device='cpu', interference='Other user generation/export workers may be active.',
                    torch_version=torch.__version__, results=results)
    if args.output:
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
