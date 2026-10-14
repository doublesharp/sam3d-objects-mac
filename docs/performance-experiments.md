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

## Generation attention cache

Eligible attention layers cache immutable conditioning K/V within one pipeline
run. The cache never crosses runs and is bypassed for unsupported inputs.
`SAM3D_ATTENTION_CACHE=0` disables it. Both conditioning branches reuse K/V
only. Every query, attention operation and output projection runs afresh,
including single-token conditioning and non-finite queries. These changes do not change sampler seeds, steps or generation
profiles.

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
