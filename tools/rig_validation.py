"""CPU structural checks for learned rig candidates. These do not qualify anatomy."""
import numpy as np


def validate_skeleton(joints, parents):
    joints = np.asarray(joints, dtype=float)
    raw = np.asarray(parents)
    if joints.ndim != 2 or joints.shape[1] != 3 or not len(joints) or not np.isfinite(joints).all():
        raise ValueError('joints must be nonempty finite Nx3 coordinates')
    if len(joints) > 65535:
        raise ValueError('joint count exceeds unsigned-short skin indices')
    if raw.shape != (len(joints),) or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError('parents must contain one integer per joint')
    parents = raw.astype(np.int64)
    if np.any((parents < -1) | (parents >= len(joints))):
        raise ValueError('parent index is outside the skeleton')
    if np.count_nonzero(parents == -1) != 1:
        raise ValueError('skeleton must have one root')
    for joint in range(len(joints)):
        seen = set()
        current = joint
        while current != -1:
            if current in seen:
                raise ValueError('skeleton contains a parent cycle')
            seen.add(current)
            current = int(parents[current])
    return joints, parents


def validate_rig(rig, source_vertices, source_faces):
    vertices = np.asarray(source_vertices, dtype=float)
    faces = np.asarray(source_faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('source vertices must be nonempty finite Nx3 coordinates')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or not np.isfinite(faces).all() or not np.equal(faces, np.floor(faces)).all() or np.any((faces < 0) | (faces >= len(vertices))):
        raise ValueError('source faces must be valid triangle indices')
    if np.shape(rig['vertices']) != vertices.shape or not np.array_equal(rig['vertices'], vertices):
        raise ValueError('model source vertices differ from original POSITION ordering')
    if np.shape(rig['faces']) != faces.shape or not np.array_equal(rig['faces'], faces):
        raise ValueError('model source faces differ from original triangle ordering')
    joints, parents = validate_skeleton(rig['joints'], rig['parents'])
    span = float(np.ptp(vertices, axis=0).max())
    if span <= 0:
        raise ValueError('source geometry has zero bounds')
    # A modest envelope rejects runaway decoded coordinates, not anatomical errors.
    tolerance = span * .1
    if np.any(joints < vertices.min(0)-tolerance) or np.any(joints > vertices.max(0)+tolerance):
        raise ValueError('joints lie outside source bounds envelope')
    nonroot = np.flatnonzero(parents >= 0)
    if np.any(np.linalg.norm(joints[nonroot]-joints[parents[nonroot]], axis=1) <= span*1e-8):
        raise ValueError('skeleton has a zero-length bone')
    weights = np.asarray(rig['weights'], dtype=float)
    if weights.shape != (len(vertices), len(joints)) or not np.isfinite(weights).all():
        raise ValueError('weights must be finite with one row per vertex and column per joint')
    if np.any(weights < 0) or np.any(weights > 1+1e-6):
        raise ValueError('weights must be nonnegative and at most one')
    if not np.allclose(weights.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError('weights must be normalized')
    return {'vertices':len(vertices), 'joints':len(joints), 'anatomically_qualified':False}


def validate_override(rig, override):
    joints, parents = validate_skeleton(override['joints'], override['parents'])
    if not np.array_equal(rig['joints'], joints) or not np.array_equal(rig['parents'], parents):
        raise ValueError('generated skeleton differs from exact supplied coordinates or parents')


def validate_tokens(tokens, result, joint_count):
    tokens = np.asarray(tokens)
    if result.get('status') != 'inference_passed':
        raise ValueError('model did not report inference_passed')
    if tokens.ndim != 1 or not len(tokens) or not np.isfinite(tokens).all() or not np.equal(tokens, np.floor(tokens)).all():
        raise ValueError('tokens must be a complete integer sequence')
    if result.get('tokens') != len(tokens) or result.get('final_token') != int(tokens[-1]) or result.get('tokens_eos') != int(tokens[-1]):
        raise ValueError('missing final EOS or token count mismatch')
    if not all(result.get(key) is True for key in ('skeleton_eos_present','global_eos_present','skin_tokens_valid')):
        raise ValueError('model reports incomplete skeleton or skin token grammar')
    fields = ('skeleton_eos','skin_vocab_start','tokens_per_skin','tokens_eos')
    if any(not isinstance(result.get(key),int) or isinstance(result.get(key),bool) for key in fields):
        raise ValueError('missing model token grammar identifiers')
    if result['tokens_per_skin'] != 4:
        raise ValueError('unsupported model skin tokens per joint')
    delimiter = np.flatnonzero(tokens == result['skeleton_eos'])
    if len(delimiter) != 1 or delimiter[0] == 0:
        raise ValueError('missing or repeated skeleton EOS delimiter')
    skin_tokens = tokens[int(delimiter[0])+1:-1]
    if len(skin_tokens) != joint_count*4 or np.any(skin_tokens < result['skin_vocab_start']) or np.any(skin_tokens >= result['tokens_eos']):
        raise ValueError('incomplete skin token sequence or invalid vocabulary')
    if np.any(tokens[:-1] == result['tokens_eos']):
        raise ValueError('global EOS occurs before sequence end')
    if result.get('skin_token_count') != joint_count*4 or result.get('counted_skeleton_bones') != joint_count:
        raise ValueError('skin token or skeleton count does not match decoded joints')
