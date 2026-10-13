# Apple Silicon performance evidence

The active checkout is `sam-3d-mps`; experiments and new artifacts live in
`.worktrees/future-generation-performance/outputs/performance/`. Measurements
below are single runs on the 96 GiB M2 Max, with shared machine load noted.
Stage timings are not matched end-to-end upstream comparisons.

## Integrated improvements

Native CPU sparse indexing and CPU memory-mapped checkpoint loads were integrated
as `8c7a315`. The cached crab CPU decode comparison measured 264.43 to 159.05 seconds
(1.66 times faster), with bitwise identical vertices, faces and attributes. This
pair had shared machine load. The original combined runtime suite passed 150 tests with
9 skips.

Fresh source-only exports can reuse the strict geometry report produced during
repair. An in-memory certificate snapshots the exact oriented geometry after the
known export axis rotation. Reloaded geometry must match exactly after removing
UV seam duplication and normalizing triangle order and cyclic starts. Duplicate
faces and reversed winding remain distinct. Finite geometry, indices, winding
and UV checks still run. Changed geometry and existing GLBs run the full strict
checker; certificates are never loaded from file metadata or persisted.

The large whale source had 216,912 welded vertices and 434,632 faces. Its complete
cached-geometry CPU export, including repair, UV unwrap, 4K bake and files, took
555.81 seconds before reuse. The final reload/check/sidecar step took approximately
162 seconds within that run. A matched standalone comparison on the same exported geometry measured
58.33 seconds without reuse and 4.00 seconds with reuse (14.57 times faster for
this step, saving 54.33 seconds). Exact snapshot/match took 0.43 seconds in a
separate measurement. Replacing this step in the prior 555.81-second cached export
would imply roughly 10 percent less export time; that full-path saving remains
an estimate because the complete export was not rerun. Shared load explains why
the standalone baseline differs from the earlier full-export phase. Pair results
are recorded under `outputs/performance/validation/whale-validation-*.json`.
These timings exclude neural decoding. The GLB passed Khronos validation with zero errors and
zero warnings, and strict checks reported no self-intersection pairs.

## Experiments retained in the worktree

CPU window attention can be selected with `SAM3D_CPU_WINDOW_ATTENTION=segmented`;
`legacy` remains the default. Each window uses CPU SDPA without a global dense
mask. With native indexing plus this attention path, cached crab decode measured
221.14 to 75.79 seconds (2.92 times), and whale 71.79 to 38.86 seconds (1.85 times).
The baselines explicitly load pre-optimization code from `9d476de`.

These outputs are finite with matching vertex/face counts, but are not bitwise
identical. Crab changed four quad diagonals and whale one. Vertex differences
were below 2e-6 and attribute differences below 2.4e-5. Crab passed a conservative
sampled local surface-distance bound of 0.1 voxel; whale exceeded that bound
(0.000456 versus 0.000391 normalized units). Its full exported GLB nevertheless
passed strict geometry and file validation. This does not qualify the attention
path as a new default. Four CPU threads were slower than two in the whale trial;
this single observation does not establish a universal thread optimum.

MPS generation with chunk size 128 took 246.16 seconds versus 120.96 with 1024,
using the same dragon photo, mask, seed and ten steps. The first run overlapped
CPU export, while the second had less load. Occupancy also changed from 13,082
to 13,126 voxels. The result is a promising candidate, not a measured 2x default
speedup or output-equivalence proof. Chunk size remains configurable.

Keeping models resident produced materially different latent features despite
an identical seed. Residency remains unqualified pending RNG/lazy-init analysis.

## Metal mesh-decoder failure

The diagnostic completed all twelve FP32 neural transformer blocks in about
17.7 seconds, then last recorded `begin upsample_0` before the laptop restarted.
The October 13 panic reported an AGX secure-unmap firmware timeout, with no memory
pressure or swap exhaustion. This implicates the GPU upsampling boundary but does
not identify the exact operation. Full MPS mesh upsampling must not be retried as
part of this bounded optimization pass. The diagnostic permits CPU upsampling;
a custom Metal decoder is separate work.

## Scope and verification

The bounded pass integrates measured CPU wins and removes repeated work while
preserving output checks. Texture-quality and learned-rigging work is coordinated
with the other SAM chat. GPU use requires its reservation release. Existing
artifacts and checkpoints are preserved; generated evidence is ignored by Git.

The certificate implementation passed independent review and focused tests for
exact matching, modified geometry, invalid indices/values, immutable snapshots,
GLB metadata exclusion, actual exporter roundtrip, winding and nonfinite UVs.
The full worktree suite passed 105 tests with 10 skips in the sandbox. Before workflow separation, the original combined checkout passed all 175 tests with device access
(no skips). This qualifies the integrated code and its regression checks; it
does not constitute a matched end-to-end performance measurement.
A matched full upstream-to-current generation/export run remains outstanding;
previous roughly 4x end-to-end estimates should not be presented as measurements.
