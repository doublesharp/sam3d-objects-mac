# Performance behavior and qualification

## Integrated CPU changes

Native CPU sparse indexing and CPU memory-mapped checkpoint loading avoid GPU
allocation during load. CPU mesh extraction uses exact integer row keys instead
of repeated general row sorting. Spherical occupancy closing and conservative
shell bands reduce repair work while retaining the production geometry gates.
These changes preserve mesh order and strict geometry/fidelity validation.

Fresh source-only exports may reuse an in-memory geometry certificate after the
known export axis rotation. Reloaded oriented geometry must match exactly after
UV seam deduplication and triangle normalization. Reversed winding and duplicate
faces remain distinct. Finite geometry, indices, winding and UV checks still run.
Changed geometry and existing files use full validation. Certificates are not
persisted or trusted from metadata.

## Headless generation startup

Notebook plotting, Gradio UI, and scene-composition dependencies load when their
corresponding visualization helper is called. Command-line inference imports its
core image and model dependencies directly.

## Generation attention cache

Eligible attention layers cache immutable conditioning K/V within one pipeline
run. The cache never crosses runs and is bypassed for unsupported inputs.
`SAM3D_ATTENTION_CACHE=0` disables it. Both conditioning branches reuse K/V
only. Every query, attention operation and output projection runs afresh,
including single-token conditioning and non-finite queries. These changes do not change sampler seeds, steps or generation
profiles.

## Folder overlap

The full `./run.sh` folder flow automatically overlaps one fresh generator and
one CPU exporter for direct Gaussian textures on hosts with at least 64 GiB RAM
and 32 GiB currently available, with surface repair enabled. Disabling repair
selects serial because legacy hole cleanup can use the accelerator. Unknown
memory and explicit MPS/CUDA export settings fall back to serial. A completed-generation queue holds at most one
item. A new overlap is suspended under low available memory; export drains and
the next item runs serially without a memory wait inside the scheduler.

Use `SAM3D_PIPELINED_BATCH=0 ./run.sh` to select serial. Standalone `cli.py` is
unchanged: it generates splats and latents without export. Other texture modes
retain the serial flow and their existing device settings.

Overlapping export forces `SAM3D_EXPORT_DEVICE=cpu` and
`SAM3D_MESH_DECODE_DEVICE=cpu`. It uses two PyTorch and BLAS threads, one PyTorch
interop thread, and two Gaussian KD-tree workers. Standalone export retains its
normal resource defaults; optional `SAM3D_EXPORT_THREADS`,
`SAM3D_EXPORT_INTEROP_THREADS`, and `SAM3D_GAUSSIAN_KD_WORKERS` expose those budgets.
The thread settings require positive integers; KD workers also accepts `-1`.
Normal repair-distance threads retain their existing budget.

Profiles, source-only/complete output, texture settings and game targets are
inherited. Failure records preserve successful generation files. Failed items
are logged and later items continue; the runner returns failure if any stage
failed. Ctrl-C terminates and reaps both child process groups. Run-specific export
markers prevent the shell from duplicating an export.

## Native CPU benchmarks

The scripts in benchmarks/benchmark_native_indexing.py and
benchmarks/benchmark_cached_sparse_decode.py default to a pinned, reachable
pre-indexing implementation from this branch. Use --base to compare another
compatible Git revision. Both compare outputs exactly. Kernel timings and
saved-latent CPU decode timings do not measure end-to-end generation.

## Experimental paths and timing limits

CPU segmented window attention remains opt-in through
`SAM3D_CPU_WINDOW_ATTENTION=segmented`; `legacy` remains the default. Full MPS mesh
upsampling is not enabled by these changes. MPS generation chunk sizing remains
configurable and model residency is not introduced.

Stage speedups do not imply full-pipeline speedups. Earlier extraction, repair,
and batch prototype measurements do not establish production runtime guarantees.
A single matched CPU sampler pair showed an exact-output improvement, but broader
inputs and complete exports still require separate qualification. Machine-specific
commands, raw timings, generated files and runtime parity reports belong in local
evaluation artifacts rather than maintained product documentation.
