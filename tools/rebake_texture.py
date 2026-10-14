"""Rebake one standard SAM GLB atlas from saved Gaussian appearance, on CPU.

The original GLB binary is retained verbatim. Only the base-color texture
reference changes; a new PNG, texture and buffer view are appended.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import resource
import struct
import sys
import tempfile
import time

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.special import expit


def read_glb(path):
    raw = Path(path).read_bytes()
    if len(raw) < 28:
        raise ValueError('Truncated GLB')
    magic, version, length = struct.unpack_from('<III', raw)
    if magic != 0x46546C67 or version != 2 or length != len(raw):
        raise ValueError('Expected a valid GLB 2 file')
    chunks = []
    offset = 12
    while offset < length:
        if offset + 8 > length:
            raise ValueError('Truncated GLB chunk')
        size, kind = struct.unpack_from('<II', raw, offset)
        offset += 8
        if size % 4 or offset + size > length:
            raise ValueError('Invalid GLB chunk size')
        chunks.append((kind, raw[offset:offset + size]))
        offset += size
    if [kind for kind, _ in chunks] != [0x4E4F534A, 0x004E4942]:
        raise ValueError('Expected JSON and one embedded BIN chunk only')
    return json.loads(chunks[0][1]), chunks[1][1]


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _referenced(items, index, name):
    index = _integer(index, name)
    if index >= len(items):
        raise ValueError(f'{name} is out of range')
    return items[index]


def accessor_array(document, binary, index, *, vertex_attribute=False, indices=False):
    """Decode a dense, legally aligned accessor without changing its bytes."""
    try:
        accessor = _referenced(document['accessors'], index, 'accessor reference')
        view = _referenced(document['bufferViews'], accessor['bufferView'], 'bufferView reference')
        if 'sparse' in accessor or accessor.get('normalized', False):
            raise ValueError('Sparse or normalized mesh accessors are unsupported')
        types = {5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}
        widths = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
        component = _integer(accessor['componentType'], 'componentType')
        dtype = np.dtype(types[component])
        width = widths[accessor['type']]
        count = _integer(accessor['count'], 'accessor count', 1)
        start = _integer(view.get('byteOffset', 0), 'bufferView byteOffset')
        local_offset = _integer(accessor.get('byteOffset', 0), 'accessor byteOffset')
        view_length = _integer(view['byteLength'], 'bufferView byteLength', 1)
        buffer_length = _integer(document['buffers'][0]['byteLength'], 'buffer byteLength', 1)
        buffer_index = _integer(view.get('buffer', 0), 'buffer reference')
        element_size = width * dtype.itemsize
        stride = element_size
        if 'byteStride' in view:
            stride = _integer(view['byteStride'], 'bufferView byteStride', 4)
            if indices or stride > 252 or stride % 4 or stride < element_size:
                raise ValueError('Invalid glTF byteStride; indices must be tightly packed')
        if (local_offset % dtype.itemsize or (start + local_offset) % dtype.itemsize or
                start % dtype.itemsize):
            raise ValueError('Accessor offsets must be component-aligned')
        if vertex_attribute and (local_offset % 4 or (start + local_offset) % 4 or stride % 4):
            raise ValueError('Vertex attributes must be 4-byte aligned')
        required = local_offset + (count - 1) * stride + element_size
        if (buffer_index != 0 or buffer_length > len(binary) or required > view_length or
                start + view_length > buffer_length):
            raise ValueError('Invalid mesh accessor bounds or stride')
        return np.ndarray((count, width), dtype=dtype, buffer=binary,
                          offset=start + local_offset, strides=(stride, dtype.itemsize)).copy()
    except (KeyError, IndexError, TypeError, OverflowError) as exc:
        raise ValueError('Unsupported or malformed mesh accessor') from exc


def _mesh_inputs(document, binary):
    buffers = document.get('buffers', [])
    if (len(buffers) != 1 or 'uri' in buffers[0] or
            type(buffers[0].get('byteLength')) is not int or
            not 0 < buffers[0]['byteLength'] <= len(binary)):
        raise ValueError('Expected one embedded GLB buffer')
    if document.get('extensionsRequired'):
        raise ValueError('Required GLB extensions are unsupported')
    meshes = document.get('meshes', [])
    if len(meshes) != 1 or len(meshes[0].get('primitives', [])) != 1:
        raise ValueError('Expected one mesh with one triangle primitive')
    for node in document.get('nodes', []):
        for field, identity in [('matrix', np.eye(4).ravel()), ('translation', [0, 0, 0]),
                                ('rotation', [0, 0, 0, 1]), ('scale', [1, 1, 1])]:
            if field in node:
                value = np.asarray(node[field])
                if value.shape != np.shape(identity) or not np.array_equal(value, identity):
                    raise ValueError('Only identity node transforms are supported')
    primitive = meshes[0]['primitives'][0]
    if primitive.get('mode', 4) != 4 or primitive.get('extensions') or primitive.get('targets'):
        raise ValueError('Only dense triangles without mesh extensions or morph targets are supported')
    if len(document.get('materials', [])) != 1 or primitive.get('material') != 0:
        raise ValueError('Expected one material and base-color atlas')
    try:
        info = document['materials'][0]['pbrMetallicRoughness']['baseColorTexture']
        if info.get('texCoord', 0) != 0 or info.get('extensions'):
            raise ValueError('Only untransformed TEXCOORD_0 base-color textures are supported')
        texture = _referenced(document['textures'], info['index'], 'texture reference')
        if texture.get('extensions'):
            raise ValueError('Texture extensions are unsupported')
        image = _referenced(document['images'], texture['source'], 'image reference')
        if 'uri' in image or 'bufferView' not in image:
            raise ValueError('Expected an embedded base-color image')
        _referenced(document['bufferViews'], image['bufferView'], 'image bufferView reference')
        attrs = primitive['attributes']
        vertices = accessor_array(document, binary, attrs['POSITION'], vertex_attribute=True)
        normals = accessor_array(document, binary, attrs['NORMAL'], vertex_attribute=True)
        uv = accessor_array(document, binary, attrs['TEXCOORD_0'], vertex_attribute=True)
        indices = accessor_array(document, binary, primitive['indices'], indices=True)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('Missing supported positions, normals, UVs, indices or base-color image') from exc
    if (vertices.shape[1] != 3 or normals.shape != vertices.shape or
            uv.shape != (len(vertices), 2) or indices.shape[1] != 1 or indices.size % 3):
        raise ValueError('Invalid mesh attribute dimensions or triangle indices')
    if any(x.dtype.kind != 'f' or not np.isfinite(x).all() for x in (vertices, normals, uv)):
        raise ValueError('Positions, normals and UVs must be finite floating-point arrays')
    if np.any(np.linalg.norm(normals, axis=1) < 1e-8):
        raise ValueError('Mesh normals must be nonzero')
    if np.any(uv < 0) or np.any(uv > 1):
        raise ValueError('Atlas UVs must lie in [0,1]')
    if indices.dtype.kind != 'u' or indices.max() >= len(vertices):
        raise ValueError('Invalid triangle indices')
    faces = indices.reshape(-1, 3).astype(np.int64)
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError('Triangle indices repeat a vertex')
    # Standard SAM exports row vectors as z-up @ rotation. Reverse it here.
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    vertices, normals = vertices @ rotation.T, normals @ rotation.T
    # Trimesh exports v as 1-v; the existing baker already flips its image.
    uv[:, 1] = 1 - uv[:, 1]
    return vertices, faces, uv, normals, info, texture


def _splat_inputs(path):
    try:
        vertex = PlyData.read(path)['vertex'].data
        def columns(names):
            return np.column_stack([vertex[name] for name in names]).astype(np.float32)
        points = columns(['x', 'y', 'z'])
        colors = columns([f'f_dc_{i}' for i in range(3)]) * 0.28209479177387814 + 0.5
        raw_opacity = np.asarray(vertex['opacity'], dtype=np.float32)
        if not np.isfinite(raw_opacity).all():
            raise ValueError('Gaussian opacity must be finite')
        opacity = expit(raw_opacity)
        with np.errstate(over='ignore', under='ignore'):
            scales = np.exp(columns([f'scale_{i}' for i in range(3)]))
        rotations = columns([f'rot_{i}' for i in range(4)])
    except (KeyError, IndexError) as exc:
        raise ValueError('Expected standard Gaussian PLY appearance fields') from exc
    if (not len(points) or any(not np.isfinite(x).all() for x in (points, colors, opacity, scales, rotations)) or
            np.any(scales <= 0) or np.any(np.linalg.norm(rotations, axis=1) < 1e-8)):
        raise ValueError('Gaussian appearance must be nonempty and finite with positive scales and valid rotations')
    # The existing baker filters opacity, so ensure it has something to sample.
    threshold = float(os.environ.get('SAM3D_EXPERIMENTAL_COLOR_OPACITY', '0.05'))
    if not np.any(opacity >= threshold):
        raise ValueError('No Gaussians meet the baker opacity threshold')
    return points, colors, opacity, scales, rotations


def _atomic_write(path, payload, replace):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(payload)
        if replace:
            os.replace(temporary, path)
        else:
            # link is atomic and refuses a destination that appeared during baking.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rebake(source_glb, splat_ply, output_glb, output_png, *, texture_size=4096, workers=2, replace=False):
    start = time.perf_counter()
    if not isinstance(texture_size, int) or not 64 <= texture_size <= 8192:
        raise ValueError('texture_size must be an integer from 64 to 8192')
    if not isinstance(workers, int) or (workers != -1 and workers < 1):
        raise ValueError('workers must be a positive integer or -1')
    source, splat, output, png = map(Path, (source_glb, splat_ply, output_glb, output_png))
    resolved = [p.resolve() for p in (source, splat, output, png)]
    if resolved[2] in resolved[:2] or resolved[3] in resolved[:2] or resolved[2] == resolved[3]:
        raise ValueError('Output paths must be distinct and cannot overwrite source files')
    for destination in (output, png):
        if destination.exists() and not replace:
            raise FileExistsError(f'{destination} exists; use --replace explicitly')
        if destination.is_symlink():
            raise ValueError('Output symlinks are unsupported')
    document, original_binary = read_glb(source)
    vertices, faces, uv, normals, info, old_texture = _mesh_inputs(document, original_binary)
    appearance = _splat_inputs(splat)
    # Read-only dependency use, no model loading and no accelerator initialization.
    os.environ.setdefault('LIDRA_SKIP_INIT', '1')
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sam3d_objects.experimental_retopo.normal_bake import bake_gaussian_color_texture
    baker_start = time.perf_counter()
    image = bake_gaussian_color_texture(vertices, faces, uv, normals, *appearance,
                                         texture_size=texture_size, workers=workers)
    baker_seconds = time.perf_counter() - baker_start
    encoded_png = io.BytesIO()
    Image.fromarray(image).save(encoded_png, format='PNG')
    png_bytes = encoded_png.getvalue()
    binary = bytearray(original_binary)
    binary.extend(b'\0' * (-len(binary) % 4))
    view_index = len(document['bufferViews'])
    document['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(png_bytes)})
    binary.extend(png_bytes)
    image_index = len(document['images'])
    document['images'].append({'bufferView': view_index, 'mimeType': 'image/png'})
    texture = copy.deepcopy(old_texture)
    texture['source'] = image_index
    info['index'] = len(document['textures'])
    document['textures'].append(texture)
    document['buffers'][0]['byteLength'] = len(binary)
    json_bytes = json.dumps(document, separators=(',', ':')).encode('utf-8')
    json_bytes += b' ' * (-len(json_bytes) % 4)
    binary.extend(b'\0' * (-len(binary) % 4))
    payload = (struct.pack('<III', 0x46546C67, 2, 28 + len(json_bytes) + len(binary)) +
               struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes +
               struct.pack('<II', len(binary), 0x004E4942) + binary)
    # Neither source is opened for writing. Existing output files require replace.
    _atomic_write(png, png_bytes, replace)
    try:
        _atomic_write(output, payload, replace)
    except Exception:
        if not replace:
            png.unlink(missing_ok=True)
        raise
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {'source_glb': str(source.resolve()), 'splat_ply': str(splat.resolve()),
            'output_glb': str(output.resolve()), 'output_png': str(png.resolve()),
            'texture_size': texture_size, 'workers': workers, 'vertices': len(vertices),
            'triangles': len(faces), 'gaussians': len(appearance[0]),
            'baker_seconds': baker_seconds, 'total_seconds': time.perf_counter() - start,
            'glb_bytes': len(payload), 'png_bytes': len(png_bytes),
            'peak_rss_bytes': int(usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss * 1024)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_glb', type=Path)
    parser.add_argument('splat_ply', type=Path)
    parser.add_argument('output_glb', type=Path)
    parser.add_argument('--png', type=Path, help='Separate PNG output; defaults to output GLB stem.png')
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=2, help='KD tree query workers; -1 uses all CPUs')
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    try:
        stats = rebake(args.source_glb, args.splat_ply, args.output_glb,
                       args.png or args.output_glb.with_suffix('.png'),
                       texture_size=args.size, workers=args.workers, replace=args.replace)
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, f'error: {exc}\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
