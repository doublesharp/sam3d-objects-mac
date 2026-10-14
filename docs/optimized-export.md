# Optimized Apple Silicon export

The normal `run.sh` flow and `ply2glb.py` now use automatic surface repair and a direct Gaussian texture bake. Disconnected debris is removed on CPU before simplification. A closed source must remain closed and pass shape checks after reduction; failed reductions try Open3D and then retain the original geometry if necessary. Triangle-collapse cleanup also rejects changes that break closure or erase thin features. Generated source meshes are repaired before UV unwrapping. Game meshes are fitted to that repaired source, so their polygon budget and sidecars remain consistent.

Run the full image workflow with `./run.sh`, or reuse a saved reconstruction:

```sh
SPARSE_BACKEND=native .venv/bin/python ply2glb.py outputs/your-object
```

Diffusion uses MPS on Apple Silicon. Mesh decoding defaults to the verified CPU path because large MPS decodes encountered Metal command-buffer errors. Repair uses CPU Open3D distance queries, SciPy closing/flood fill, and marching cubes. The direct color bake samples the saved Gaussian appearance instead of rendering dozens of views. CUDA mesh decoding remains available on CUDA hosts.

Repair skips healthy meshes. Automatic settings use principal-axis thickness, independent of object rotation: thin objects use a 224-cell longest axis and a two-cell closing radius; compact objects use 192 and four. The geometry is checked for finite values, closed manifold edges, consistent winding, bounds retention, bidirectional surface-area error, and interior-volume preservation. Unresolved large openings or insufficient enclosed interior cause a rejection rather than a fabricated successful result. Reports are saved as `mesh_repair_report.json` and `mesh_game_repair_report.json` and included in GLB metadata.

Intentional cavities, thin walls, and small features still need visual inspection. A rejected large missing surface needs a new reconstruction or a deliberate modeling edit.

## Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAM3D_SURFACE_REPAIR` | `1` | Set `0` to disable automatic repair. |
| `SAM3D_REPAIR_RESOLUTION` | `auto` | Override the longest grid axis, 32–320. |
| `SAM3D_REPAIR_RADIUS` | `auto` | Override closing radius, 0–6 cells. |
| `SAM3D_TEXTURE_MODE` | `gaussian` | Use `average` for the prior multiview texture bake, or `opt`/`fast` for older bake modes. |
| `SAM3D_EXPERIMENTAL_COLOR_NEIGHBORS` | `16` | Local Gaussian samples per texel. |
| `SAM3D_MESH_DECODE_DEVICE` | `auto` | Force `cpu`, `mps`, or `cuda` for controlled experiments. |
| `SAM3D_CPU_DECODE_VOXELS` | `0` | On MPS hosts, keep automatic mesh decoding on CPU at or above this voxel count. |

Existing loadable `mesh.glb` files are reused as fitting inputs. Set `SAM3D_GAME_REBUILD_SOURCE_GLB=1` only when deliberately rebuilding that file from `slat.pt`. Save an original copy first if it must be retained.

## Measurements on M2 Max

The same crab image, mask, seed 9, and 10 diffusion steps per stage took 1151.3 seconds in the earlier local run and 240.49 seconds after optimization, a 4.79× generation speedup. Voxel coordinates, latent features, and all 778,304 Gaussian properties were bitwise identical. This comparison covers image-to-splat generation, including model loading, not a claim about every stage or every input.

The repaired, textured, rigged crab export took 64.73 seconds from saved reconstruction data. The octopus took 185.94 seconds to generate and 146.81 seconds for mesh decoding and textured export. Its final automatic export from the cached decoded mesh took 48.38 seconds. Both detailed models have 160,000 triangles and passed glTF validation with zero errors and warnings. These are individual local runs; background load and object complexity affect timings.

For further tuning, measure model loading, sparse diffusion, latent diffusion, Gaussian decoding, mesh decoding, repair, UV unwrapping, and texturing separately. Reuse decoded geometry for appearance and repair experiments. Optimize the measured bottleneck while retaining an output comparison and geometry/appearance checks.
