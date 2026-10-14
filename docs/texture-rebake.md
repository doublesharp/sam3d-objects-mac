# Rebake a saved texture atlas

Finished models use a 4096-pixel atlas; previews use 2048 pixels. New exports
default to `--texture-profile final`. Use `--texture-profile preview` or
`SAM3D_TEXTURE_PROFILE=preview` for a 2K preview. `SAM3D_TEXTURE_SIZE` explicitly
overrides either profile and must be an integer from 64 through 8192.

Existing detailed GLBs are reused by `ply2glb.py`, including their saved textures.
Changing the profile does not rebake a reused source. Use the texture-only tool
below to create a separate 4K version without decoding or repairing it again.

`tools/rebake_texture.py` samples saved Gaussian colors into the existing GLB UV atlas at a new resolution. It runs on CPU and loads no inference model. Geometry, normals, UVs, skinning, animation, metadata and the complete original binary buffer remain unchanged. The output embeds a newly baked PNG and also writes it separately. Keeping the original buffer means the GLB retains its previous atlas bytes as well.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  /path/to/sam-3d-mps/.venv/bin/python tools/rebake_texture.py \
  /path/to/completed/mesh.glb /path/to/completed/splat.ply \
  /path/to/new/mesh-4k.glb --size 4096 --workers 2
```

The PNG defaults to the output GLB name with `.png`. Use `--png /path/to/atlas.png` to choose a separate path. Existing outputs are refused unless `--replace` is explicit. The tool always refuses paths resolving to either input, even with `--replace`.

The texture-only tool defaults to 4096 pixels. Pass `--size 2048` for a preview.

`--workers` controls the existing baker's KD-tree queries. The tool defaults to two workers; the shared baker's original default stays `-1`, all workers. BLAS and OpenMP thread limits can bound other imported libraries. The rasterizer and inpainting still allocate arrays proportional to the square of the atlas size. Supported sizes are integers from 64 through 8192. Run one rebake at a time when another generation job is active.

The initial scope is standard SAM GLB output with one embedded buffer, one mesh, one triangle primitive, one material, a base-color atlas using untransformed `TEXCOORD_0`, and identity node transforms. Dense floating-point positions, normals and UVs plus unsigned triangle indices are required. References and counts must be integers; offsets must be component-aligned. Vertex attributes use 4-byte alignment, and explicit vertex strides must be multiples of four from 4 through 252 bytes and wide enough for an element. Indices must be tightly packed without an explicit stride. Sparse or normalized mesh accessors, morph targets, required extensions, compressed primitives, texture transforms, external buffers and external base-color images are rejected. Skin and animation data are retained without posing or changing them; nonidentity rest node transforms remain unsupported.

Standard SAM exports its Gaussian z-up geometry to GLB y-up by rotation. The rebake reverses that rotation for sampling and reverses trimesh's exported UV-v convention before calling the existing baker. The source normals keep the original seam behavior. No anatomy-specific heuristics, generation, repair, simplification or UV unwrapping occur.

Higher pixel counts sample the same Gaussian appearance more densely. They cannot add appearance detail absent from the splats. Check crops and the model in your viewer before deciding whether the larger atlas is useful. A 2K rebake can be compared to the source embedded atlas to check orientation independently of resolution.

The CLI prints JSON with baker time, total time, output byte counts and peak process RSS. Total time excludes interpreter startup; RSS includes imports and GLB packaging. CPU-worker limits are configuration controls, not proof that a concurrent generation suffered no slowdown.
