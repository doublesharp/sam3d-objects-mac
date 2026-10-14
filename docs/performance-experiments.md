# Performance behavior and qualification

## Integrated CPU changes

Native CPU sparse indexing and CPU memory-mapped checkpoint loading avoid GPU
allocation during load. These changes preserve convolution order and outputs.

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
