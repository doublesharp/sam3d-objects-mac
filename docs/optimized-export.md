# Optimized Apple Silicon export

The normal `run.sh` flow and `ply2glb.py` now use automatic surface repair and a direct Gaussian texture bake. Disconnected debris is removed on CPU before simplification. A closed source must remain closed and pass shape checks after reduction; failed reductions try Open3D and then retain the original geometry if necessary. Triangle-collapse cleanup also rejects changes that break closure or erase thin features. Generated source meshes are repaired before UV unwrapping. Game meshes are fitted to that repaired source, so their polygon budget and sidecars remain consistent.

Run the full image workflow with `./run.sh`, or reuse a saved reconstruction:

```sh
SPARSE_BACKEND=native .venv/bin/python ply2glb.py outputs/your-object
```

Diffusion uses MPS on Apple Silicon. Mesh decoding defaults to the verified CPU path because large MPS decodes encountered Metal command-buffer errors. Repair uses CPU Open3D distance queries, SciPy closing/flood fill, and marching cubes. The direct color bake samples the saved Gaussian appearance instead of rendering dozens of views. CUDA mesh decoding remains available on CUDA hosts.

Repair skips sources that pass strict manifold, orientation and self-intersection checks. Automatic settings use principal-axis thickness, independent of object rotation: thin objects use a 224-cell longest axis and a two-cell closing radius; compact objects use 192 and four. The geometry is checked for finite values, closed manifold edges and vertices, no triangle self-intersections, consistent winding, bounds retention, bidirectional surface-area error, and interior-volume preservation. Unresolved large openings or insufficient enclosed interior cause a rejection rather than a fabricated successful result. Reports are saved as `mesh_repair_report.json` and `mesh_game_repair_report.json` and included in GLB metadata.

Intentional cavities, thin walls, and small features still need visual inspection. A rejected large missing surface needs a new reconstruction or a deliberate modeling edit. Rigging follows geometry repair; a rig specialized for one animal should not be applied to another.

## Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAM3D_SOURCE_ONLY` | `0` | Set `1` to export/reuse the detailed GLB and OBJ without fitting or texturing a game LOD. |
| `SAM3D_REPAIR_PROFILE` | `surface` | Select `protected-detail` for a separate body core and finer exterior detail. |
| `SAM3D_SURFACE_REPAIR` | `1` | Set `0` to disable automatic repair. |
| `SAM3D_REPAIR_RESOLUTION` | `auto` | Override the longest grid axis: surface 32–320, protected-detail 64–320. |
| `SAM3D_REPAIR_RADIUS` | `auto` | Override closing radius: surface 0–6, protected-detail 1–12 coarse cells. |
| `SAM3D_REPAIR_FORCE` | `0` | Opt in to cavity cleanup even when a source mesh is already closed. |
| `SAM3D_TEXTURE_MODE` | `gaussian` | Use `average` for the prior multiview texture bake, or `opt`/`fast` for older bake modes. |
| `SAM3D_EXPERIMENTAL_COLOR_NEIGHBORS` | `16` | Local Gaussian samples per texel. |
| `SAM3D_MESH_DECODE_DEVICE` | `auto` | Force `cpu`, `mps`, or `cuda` for controlled experiments. |
| `SAM3D_CPU_DECODE_VOXELS` | `0` | On MPS hosts, keep automatic mesh decoding on CPU at or above this voxel count. |

Existing loadable `mesh.glb` files are reused as fitting inputs. Set `SAM3D_GAME_REBUILD_SOURCE_GLB=1` only when deliberately rebuilding that file from `slat.pt`. Save an original copy first if it must be retained. Forced cleanup of an existing source also needs this rebuild flag; otherwise the source is reused. Force applies once to the detailed source, after component cleanup and before game fitting. It skips preliminary polygon reduction so repair sees the original surface; the repair stage retains its 160,000-face limit. It can change intentional cavities and needs visual inspection.

For a porous reconstruction, a separately qualified stronger cleanup can be requested:

```sh
SAM3D_GAME_REBUILD_SOURCE_GLB=1 SAM3D_REPAIR_FORCE=1 \
SAM3D_REPAIR_RESOLUTION=128 SAM3D_REPAIR_RADIUS=6 \
SPARSE_BACKEND=native .venv/bin/python ply2glb.py outputs/your-object
```

This profile was checked on the whale's unintended side tunnels. It is an explicit override rather than a universal default. Surface fidelity and bounds checks still apply. Interior accounting includes volume sealed by closing as well as flood fill.

## Detailed-only export

Set `SAM3D_SOURCE_ONLY=1` when the detailed GLB is the desired result. New reconstructions stop after that GLB and its OBJ/texture sidecars. A cached `mesh.glb` is strictly checked for geometry, winding and finite UVs, without loading inference/appearance assets or baking another texture. `export-status.json` records `detailed_complete` and game `not_requested`; existing game artifacts are preserved.

```sh
SAM3D_SOURCE_ONLY=1 SAM3D_REPAIR_PROFILE=protected-detail \
SPARSE_BACKEND=native .venv/bin/python ply2glb.py outputs/your-object
```

The generated hedgehog detailed source passed strict geometry and front/back inspection. Its 20,000-triangle reduction still failed closure after local cleanup and is not qualified. Detailed-only export avoids that optional stage without accepting a defective game mesh. This does not change the default combined export.

## Body core with protected detail

For porous bodies with spikes or curled appendages, choose an explicit profile:

```sh
SAM3D_REPAIR_PROFILE=protected-detail \
SPARSE_BACKEND=native .venv/bin/python ply2glb.py outputs/your-object
```

This uses a 96-cell coarse body core beneath a 256-cell exterior field. The default coarse closing/opening radii are eight cells, with three cells of erosion. Signed-distance interpolation smooths the core between coarse cells; sub-voxel smoothing removes ambiguous fine-grid contacts. The export checks body-core topology and containment, visible exterior error, retained bounds, embedded manifoldness, triangle intersections and the 160,000-face cap. Small intersections introduced by reduction are cleaned on the bounded candidate, then rechecked against the source.

Exterior checks use 26 source-only ray directions plus convex-hull extrema. This avoids treating hidden cavity walls as lost outer detail. It is a bounded visibility proxy, not proof for every possible view or anatomy. Tests reject a removed appendage even when the bounds are unchanged. This profile needs front/back inspection for a new object class; it is not a universal replacement for the standard repair. It does not recover image detail absent from the Gaussian appearance.

Existing `mesh.glb` is still reused by default. Back it up and set `SAM3D_GAME_REBUILD_SOURCE_GLB=1` when deliberately rebuilding that source. The body profile runs once on the detailed source; game fitting reuses the strict closed surface with standard checks. Game cleanup also checks intersections, source fidelity and its budget, and regenerates matching normals and triangular fallback sidecars after local geometry changes.

## Measurements on M2 Max

The same crab image, mask, seed 9, and 10 diffusion steps per stage took 1151.3 seconds in the earlier local run and 240.49 seconds after optimization, a 4.79× generation speedup. Voxel coordinates, latent features, and all 778,304 Gaussian properties were bitwise identical. This comparison covers image-to-splat generation, including model loading, not a claim about every stage or every input.

The repaired, textured, rigged crab export took 64.73 seconds from saved reconstruction data. The octopus took 185.94 seconds to generate and 146.81 seconds for mesh decoding and textured export. Its final automatic export from the cached decoded mesh took 48.38 seconds. Both detailed models have 160,000 triangles and passed glTF validation with zero errors and warnings. These are individual local runs; background load and object complexity affect timings.

For further tuning, measure model loading, sparse diffusion, latent diffusion, Gaussian decoding, mesh decoding, repair, UV unwrapping, and texturing separately. Reuse decoded geometry for appearance and repair experiments. Optimize the measured bottleneck while retaining an output comparison and geometry/appearance checks.
