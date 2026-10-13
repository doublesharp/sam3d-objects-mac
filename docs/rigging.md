# Experimental learned rigging

`run.sh` invokes an optional learned rig stage after each successful static GLB converter process exits. The stage can infer a varied joint tree and skin weights with the separately prepared SkinTokens runtime. It never applies the procedural crab rig to other models.

Every output is a candidate awaiting anatomical and deformation review. Complete tokens, a valid tree, normalized weights and a correct bind pose do not establish that a model found every limb. A detailed whale can pass those structural checks while omitting one flipper. Humanoids, other animals and new object poses need their own qualification.

## Use after static export

```sh
./run.sh
./run.sh glb outputs/example

# Other batch orchestration: run after the ply2glb process exits.
.venv/bin/python tools/auto_rig.py outputs/example

# Finish static assets without loading a rig model.
SAM3D_RIG_MODE=off ./run.sh
```

The standalone helper returns zero for ordinary rig failures so a completed static export remains successful. Callers must inspect `rig_status.json`, especially `status`, `structurally_valid`, `anatomically_qualified` and `error`. Invalid command line arguments still return a nonzero exit code.

| Status | Meaning |
| --- | --- |
| `off` | Rig stage disabled. |
| `skipped_missing_runtime` | Optional runtime or runner missing. Static export retained. |
| `failed` | Unsupported source, model failure, timeout, invalid rig or unsafe output. Static export retained. |
| `candidate_generated` | Separate candidate passed CPU structure, source preservation and serialized bind/rest checks. Anatomical review is pending. |

The stage writes `mesh_rig_candidate.glb`, `rig_status.json` and `rig_candidate_data/` containing model results, NPZ arrays and `worker.log`. `mesh.glb` remains the finished static asset. Existing candidates or diagnostic directories cause refusal; use a fresh output folder for another attempt. Source/output paths through symlinks are refused. Status updates replace the ordinary status file atomically and never follow a status symlink.

## Runtime configuration

The runtime is optional and is not downloaded by the stage. Model source, pretrained weights and runtime packages remain outside Git.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `SAM3D_RIG_MODE` | `auto` | `auto` attempts a candidate when runtime exists; `off` skips. |
| `SAM3D_RIG_MODEL_DIR` | `.cache/rigging/SkinTokens` | Prepared source root with `run_local.py`, its configs and checkpoints. |
| `SAM3D_RIG_PYTHON` | `.cache/rigging/runtime/bin/python` | Prepared runtime Python. |
| `SAM3D_RIG_SKELETON` | unset | Optional exact skeleton JSON in source mesh coordinates. |
| `SAM3D_RIG_BEAMS` | `10` | Positive integer search width. A one-beam fast override can miss limbs. |
| `SAM3D_RIG_MAX_TIME_SECONDS` | `120` | Maximum model generation time, with a hard 120-second ceiling. |
| `SAM3D_RIG_TIMEOUT_SECONDS` | `240` | Outer worker limit including load, generation and skin decode, with a hard 240-second ceiling. |

Paths default relative to the project installation, not the current shell directory. The helper also accepts `--mode`, `--skeleton`, `--beams`, `--max-time`, `--timeout` and `--device cpu|mps`. The prepared MPS runner must cap its process memory fraction at `0.45`. The launcher overrides only the worker environment with MPS high `0.9`, low `0.8`, fallback enabled and two CPU threads. It requests full-quality point conditioning, the glTF-to-model basis transform, float32 and ten beams. A nonblocking installation-wide file lock permits one learned worker at a time. Time limits must be finite and positive, generation at most 120 seconds, outer timeout at most 240 seconds and outer timeout at least the generation limit. Shorter limits are supported. A timeout kills and reaps the worker's process group.

## Exact joint override

```json
{
  "joints": [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
  "parents": [-1, 0],
  "joint_names": ["root", "tip"]
}
```

Joints are finite source GLB coordinates. Parents contain one integer per joint, exactly one `-1` root, valid indices and no cycles. The runner may quantize joints for conditioning, then must restore the supplied coordinates and retain identical parents. The launcher rejects any difference before candidate export. Zero-length bones or joints outside the source's bounds envelope still fail structural validation. An override supplies anatomy; it does not establish that inferred attachment weights deform correctly.

## Runner contract and CPU gates

The external command executes from the configured model source root:

```sh
RIG_PYTHON MODEL_DIR/run_local.py \
  --input ABS_OUTPUT_DIR/mesh.glb --output ABS_OUTPUT_DIR/rig_candidate_data \
  --device mps --dtype float32 --threads 2 --beams 10 \
  --full-quality --gltf-to-blender --max-time 120 \
  [--skeleton ABS_SKELETON_JSON]
```

The runner writes `result.json` with explicit `status: inference_passed`, token count, final/global EOS, skeleton EOS, skin vocabulary start, `tokens_per_skin: 4`, skeleton bone count, skin token count and grammar flags. `rig.npz` must contain source-order `vertices`, `faces`, source-coordinate `joints`, integer `parents`, dense normalized `weights` and the complete integer `tokens` sequence. A zero exit code alone is insufficient. Missing model grammar identifiers fail clearly; update an older research runner before using the integration.

CPU checks independently require one skeleton EOS delimiter, a terminal global EOS, four valid skin tokens per joint, metadata agreement, a rooted acyclic skeleton, finite nonnegative normalized weights, source geometry equality and bounded nonzero bones. Input GLBs must have one embedded buffer, one uncompressed triangle primitive, an identity scene-root mesh node, no instances, morph targets or existing skin. Unsupported scenes remain static assets.

Export appends attributes, joints and inverse bind matrices to the original document and binary buffer. Geometry, indices, UVs, materials, textures and images retain their source values; the original binary is an exact prefix. Dense weights reduce to four influences and renormalize for glTF, with maximum/mean discarded mass in status. Fewer than four joints receive zero-weight padding. The serialized skin must preserve the source rest pose and bind identity. The candidate has no automatic animation or anatomical approval.
