"""CPU baseline/candidate decode of one existing latent, without generation.

Original forward methods are bound only to the benchmark's baseline instances.
No imported production module is overwritten and no existing artifact is changed.
"""
import os
os.environ.setdefault('SPARSE_BACKEND', 'native')
os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.conv import conv_native
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor
from sam3d_objects.model.io import load_model_from_checkpoint


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def run(label, args, baseline):
    torch.manual_seed(1)
    started = time.perf_counter()
    config = OmegaConf.load(str(args.config))
    config.device = 'cpu'
    config.use_fp16 = False
    model = instantiate(config)
    model = load_model_from_checkpoint(model, str(args.checkpoint), strict=True,
                                       device='cpu', freeze=True, eval=True, state_dict_key=None)
    if label == 'baseline':
        for module in model.modules():
            if isinstance(module, (conv_native.SparseConv3d, conv_native.SparseInverseConv3d)):
                method = baseline[type(module).__name__].forward
                module.forward = method.__get__(module, type(module))
    load_seconds = time.perf_counter() - started
    saved = torch.load(args.latent, map_location='cpu', weights_only=False)
    latent = SparseTensor(saved['feats'].float(), saved['coords'], shape=saved['shape'])
    conv_timings = []
    def pre_hook(module, inputs):
        module._benchmark_start = time.perf_counter()
    def post_hook(module, inputs, output):
        conv_timings.append(dict(module=module._benchmark_name, voxels=len(inputs[0].coords),
                                 in_channels=module.in_channels, out_channels=module.out_channels,
                                 seconds=time.perf_counter() - module._benchmark_start))
    for name, module in model.named_modules():
        if isinstance(module, (conv_native.SparseConv3d, conv_native.SparseInverseConv3d)):
            module._benchmark_name = name
            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)
    started = time.perf_counter()
    with torch.inference_mode():
        meshes = model(latent)
    elapsed = time.perf_counter() - started
    outputs = [dict(vertices=mesh.vertices.cpu(), faces=mesh.faces.cpu(),
                    vertex_attrs=mesh.vertex_attrs.cpu() if mesh.vertex_attrs is not None else None)
               for mesh in meshes]
    path = args.output.parent / ('sparse_decode_' + label + '.pt')
    torch.save(outputs, path)
    entry = dict(label=label, load_seconds=load_seconds, decode_seconds=elapsed,
                 convolution_seconds=sum(row['seconds'] for row in conv_timings),
                 conv_layers=conv_timings, latent_voxels=len(latent.coords),
                 output_file=str(path), meshes=[dict(vertices=len(out['vertices']), faces=len(out['faces']),
                 finite_vertices=bool(torch.isfinite(out['vertices']).all()),
                 vertices_sha256=digest(out['vertices']), faces_sha256=digest(out['faces']),
                 attrs_sha256=digest(out['vertex_attrs']) if out['vertex_attrs'] is not None else None) for out in outputs])
    del model, latent, saved, meshes, outputs
    gc.collect()
    print(json.dumps(entry), flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='b02e1d29baf35498bad36a0fe17c83d5ac0cc20b')
    parser.add_argument('--latent', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    source = subprocess.check_output(['git', 'show', args.base + ':sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/conv_native.py'], text=True, cwd=Path(__file__).resolve().parents[1])
    baseline = {'__name__': conv_native.__name__, '__package__': conv_native.__package__}
    exec(compile(source, '<baseline-conv>', 'exec'), baseline)
    entries = [run(label, args, baseline) for label in ['baseline', 'candidate']]
    a, b = [torch.load(row['output_file'], weights_only=False) for row in entries]
    assert len(a) == len(b)
    for first, second in zip(a, b):
        for name in ['vertices', 'faces', 'vertex_attrs']:
            if first[name] is None:
                assert second[name] is None
            else:
                torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    evidence = dict(base=args.base, device='cpu', dtype='float32', threads=args.threads,
                    latent=str(args.latent), checkpoint=str(args.checkpoint), config=str(args.config),
                    interference='Existing user export/generation workers active; one sample per implementation, baseline first.',
                    results=entries, output='bitwise identical vertices, faces, and vertex attributes',
                    speedup=entries[0]['decode_seconds'] / entries[1]['decode_seconds'])
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(dict(speedup=evidence['speedup'], output=evidence['output'])), flush=True)


if __name__ == '__main__':
    main()
