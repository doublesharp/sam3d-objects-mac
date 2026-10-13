import copy
import io
import json
import struct
import importlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement


def pack_glb(path, document, binary):
    encoded = json.dumps(document).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary += b'\0' * (-len(binary) % 4)
    path.write_bytes(struct.pack('<III', 0x46546c67, 2, 28 + len(encoded) + len(binary)) +
                     struct.pack('<II', len(encoded), 0x4e4f534a) + encoded +
                     struct.pack('<II', len(binary), 0x004e4942) + binary)


def unpack_glb(path):
    raw = path.read_bytes()
    length = struct.unpack_from('<I', raw, 12)[0]
    return json.loads(raw[20:20 + length]), raw[28 + length:]


@pytest.fixture
def source(tmp_path):
    binary = bytearray()
    doc = {'asset': {'version': '2.0', 'extras': {'opaque': ['keep', 3]}},
           'extras': {'source': 'fixture'}, 'buffers': [{'byteLength': 0}],
           'bufferViews': [], 'accessors': [], 'nodes': [{'mesh': 0}, {}],
           'scenes': [{'nodes': [0, 1]}], 'scene': 0,
           'materials': [{'name': 'keep material', 'extras': {'tag': 5},
                         'pbrMetallicRoughness': {'baseColorTexture': {'index': 0},
                                                  'metallicFactor': 0.3, 'roughnessFactor': 0.8}}],
           'textures': [{'source': 0, 'sampler': 0}], 'samplers': [{'wrapS': 33071}],
           'images': [], 'meshes': []}

    def accessor(array, kind, component=5126):
        binary.extend(b'\0' * (-len(binary) % 4))
        vi = len(doc['bufferViews'])
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': array.nbytes})
        binary.extend(array.tobytes())
        ai = len(doc['accessors'])
        doc['accessors'].append({'bufferView': vi, 'componentType': component, 'count': len(array), 'type': kind})
        return ai

    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0, -1], [0, 0, -1]], dtype='<f4')
    attrs = {'POSITION': accessor(vertices, 'VEC3'),
             'NORMAL': accessor(np.tile([0, 1, 0], (4, 1)).astype('<f4'), 'VEC3'),
             'TEXCOORD_0': accessor(np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype='<f4'), 'VEC2'),
             'JOINTS_0': accessor(np.zeros((4, 4), dtype='<u2'), 'VEC4', 5123),
             'WEIGHTS_0': accessor(np.tile([1, 0, 0, 0], (4, 1)).astype('<f4'), 'VEC4')}
    indices = accessor(np.array([0, 1, 2, 0, 2, 3], dtype='<u2'), 'SCALAR', 5123)
    doc['meshes'] = [{'name': 'keep mesh', 'primitives': [{'attributes': attrs, 'indices': indices, 'material': 0}]}]
    doc['skins'] = [{'joints': [1], 'inverseBindMatrices': accessor(np.eye(4, dtype='<f4').reshape(1, 16), 'MAT4')}]
    doc['nodes'][0]['skin'] = 0
    time = accessor(np.array([0, 1], dtype='<f4'), 'SCALAR')
    values = accessor(np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype='<f4'), 'VEC4')
    doc['animations'] = [{'samplers': [{'input': time, 'output': values}],
                           'channels': [{'sampler': 0, 'target': {'node': 1, 'path': 'rotation'}}]}]
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), 'gray').save(buf, format='PNG')
    doc['images'] = [{'bufferView': len(doc['bufferViews']), 'mimeType': 'image/png'}]
    doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(buf.getvalue())})
    binary.extend(buf.getvalue())
    doc['buffers'][0]['byteLength'] = len(binary)
    glb = tmp_path / 'source.glb'
    pack_glb(glb, doc, bytes(binary))
    names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    ply = np.zeros(81, dtype=[(n, '<f4') for n in names])
    xx, yy = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9))
    ply['x'], ply['y'] = xx.ravel(), yy.ravel()
    colors = np.column_stack([yy.ravel(), xx.ravel(), np.ones(81) * 0.1])
    for i in range(3):
        ply[f'f_dc_{i}'] = (colors[:, i] - 0.5) / 0.28209479177387814
        ply[f'scale_{i}'] = np.log(0.01 if i == 2 else 0.1)
    ply['opacity'], ply['rot_0'] = 5, 1
    splat = tmp_path / 'splat.ply'
    PlyData([PlyElement.describe(ply, 'vertex')]).write(splat)
    return glb, splat, doc, bytes(binary)


def tool():
    # An explicit assertion supplies a useful red failure before implementation exists.
    assert (Path(__file__).parents[1] / 'tools/rebake_texture.py').exists(), 'texture rebake tool is missing'
    return importlib.import_module('tools.rebake_texture')


def test_rebake_preserves_source_and_all_original_accessor_bytes(source, tmp_path):
    glb, splat, original, binary = source
    before = glb.read_bytes(), splat.read_bytes()
    out, png = tmp_path / 'new.glb', tmp_path / 'new.png'
    stats = tool().rebake(glb, splat, out, png, texture_size=128, workers=1)
    assert Image.open(png).size == (128, 128)
    doc, data = unpack_glb(out)
    assert data[:len(binary)] == binary
    for key in ('accessors', 'meshes', 'nodes', 'skins', 'animations', 'asset', 'extras', 'scenes', 'samplers'):
        assert doc[key] == original[key]
    material = copy.deepcopy(doc['materials'][0])
    material['pbrMetallicRoughness']['baseColorTexture']['index'] = 0
    assert material == original['materials'][0]
    assert doc['bufferViews'][:-1] == original['bufferViews']
    image = doc['images'][doc['textures'][doc['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['index']]['source']]
    view = doc['bufferViews'][image['bufferView']]
    assert data[view['byteOffset']:view['byteOffset'] + view['byteLength']] == png.read_bytes()
    assert before == (glb.read_bytes(), splat.read_bytes())
    assert stats['baker_seconds'] > 0


def test_y_up_positions_and_raw_gltf_uv_orientation(source, tmp_path):
    glb, splat, *_ = source
    tool().rebake(glb, splat, tmp_path / 'new.glb', tmp_path / 'new.png', texture_size=64, workers=1)
    image = np.array(Image.open(tmp_path / 'new.png'))
    # Gaussian y becomes exported -z, and raw glTF v=0 addresses image top.
    assert image[5, 32, 0] > 210 and image[58, 32, 0] < 45
    assert image[32, 58, 1] > 210 and image[32, 5, 1] < 45


@pytest.mark.parametrize('target', ['new.glb', 'new.png'])
def test_refuses_overwrite_before_baking(source, tmp_path, target):
    glb, splat, *_ = source
    path = tmp_path / target
    path.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        tool().rebake(glb, splat, tmp_path / 'new.glb', tmp_path / 'new.png', texture_size=64)
    assert path.read_bytes() == b'keep'


def test_replace_and_source_protection(source, tmp_path):
    glb, splat, *_ = source
    out, png = tmp_path / 'new.glb', tmp_path / 'new.png'
    out.write_bytes(b'old'); png.write_bytes(b'old')
    tool().rebake(glb, splat, out, png, texture_size=64, workers=1, replace=True)
    assert Image.open(png).size == (64, 64)
    with pytest.raises(ValueError, match='source'):
        tool().rebake(glb, splat, glb, png, texture_size=64, replace=True)


@pytest.mark.parametrize('case', ['transform', 'multi', 'external', 'uv_transform', 'missing_normal', 'bad_indices', 'nan_uv'])
def test_rejects_unsupported_or_invalid_mesh(source, tmp_path, case):
    glb, splat, doc, binary = source
    if case == 'transform': doc['nodes'][0]['translation'] = [1, 0, 0]
    if case == 'multi': doc['meshes'].append(copy.deepcopy(doc['meshes'][0]))
    if case == 'external': doc['buffers'][0]['uri'] = 'external.bin'
    if case == 'uv_transform': doc['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['extensions'] = {'KHR_texture_transform': {'offset': [0.1, 0]}}
    if case == 'missing_normal': del doc['meshes'][0]['primitives'][0]['attributes']['NORMAL']
    if case in ('bad_indices', 'nan_uv'):
        primitive = doc['meshes'][0]['primitives'][0]
        ai = primitive['indices'] if case == 'bad_indices' else primitive['attributes']['TEXCOORD_0']
        offset = doc['bufferViews'][doc['accessors'][ai]['bufferView']]['byteOffset']
        binary = bytearray(binary)
        struct.pack_into('<H' if case == 'bad_indices' else '<f', binary, offset, 999 if case == 'bad_indices' else float('nan'))
    pack_glb(glb, doc, bytes(binary))
    with pytest.raises(ValueError):
        tool().rebake(glb, splat, tmp_path / 'new.glb', tmp_path / 'new.png', texture_size=64)
    assert not (tmp_path / 'new.glb').exists()


@pytest.mark.parametrize('size', [0, 32, 8193])
def test_rejects_unbounded_dimensions(source, tmp_path, size):
    glb, splat, *_ = source
    with pytest.raises(ValueError, match='size'):
        tool().rebake(glb, splat, tmp_path / 'new.glb', tmp_path / 'new.png', texture_size=size)


def test_workers_preserve_baker_output(source, tmp_path):
    glb, splat, *_ = source
    for workers in (1, 2):
        tool().rebake(glb, splat, tmp_path / f'{workers}.glb', tmp_path / f'{workers}.png', texture_size=64, workers=workers)
    assert (tmp_path / '1.png').read_bytes() == (tmp_path / '2.png').read_bytes()
    with pytest.raises(ValueError, match='workers'):
        tool().rebake(glb, splat, tmp_path / 'bad.glb', tmp_path / 'bad.png', texture_size=64, workers=0)


def test_baker_worker_parameter_keeps_default_and_rejects_invalid_counts():
    import inspect
    from sam3d_objects.experimental_retopo.normal_bake import bake_gaussian_color_texture
    assert inspect.signature(bake_gaussian_color_texture).parameters['workers'].default == -1
    with pytest.raises(ValueError, match='workers'):
        bake_gaussian_color_texture(*([None] * 9), workers=0)


@pytest.mark.parametrize('case', ['strided', 'truncated', 'negative_offset', 'sparse', 'normalized', 'zero_normal', 'uv_bounds', 'bad_scale', 'bad_rotation', 'empty_opacity'])
def test_additional_invalid_input_rejection(source, tmp_path, case):
    glb, splat, doc, binary = source
    if case in ('strided', 'negative_offset', 'sparse', 'normalized'):
        ai = doc['meshes'][0]['primitives'][0]['attributes']['POSITION']
        accessor = doc['accessors'][ai]
        if case == 'strided': doc['bufferViews'][accessor['bufferView']]['byteStride'] = 10000
        if case == 'negative_offset': accessor['byteOffset'] = -1
        if case == 'sparse': accessor['sparse'] = {}
        if case == 'normalized': accessor['normalized'] = True
    if case in ('zero_normal', 'uv_bounds'):
        name = 'NORMAL' if case == 'zero_normal' else 'TEXCOORD_0'
        ai = doc['meshes'][0]['primitives'][0]['attributes'][name]
        offset = doc['bufferViews'][doc['accessors'][ai]['bufferView']]['byteOffset']
        binary = bytearray(binary)
        if case == 'zero_normal': struct.pack_into('<fff', binary, offset, 0, 0, 0)
        else: struct.pack_into('<f', binary, offset, 1.1)
    if case in ('bad_scale', 'bad_rotation', 'empty_opacity'):
        data = PlyData.read(splat)['vertex'].data.copy()
        if case == 'bad_scale': data['scale_0'] = float('inf')
        if case == 'bad_rotation':
            for i in range(4): data[f'rot_{i}'] = 0
        if case == 'empty_opacity': data['opacity'] = -100
        PlyData([PlyElement.describe(data, 'vertex')]).write(splat)
    pack_glb(glb, doc, bytes(binary))
    if case == 'truncated': glb.write_bytes(glb.read_bytes()[:-1])
    with pytest.raises(ValueError):
        tool().rebake(glb, splat, tmp_path / 'new.glb', tmp_path / 'new.png', texture_size=64)


def test_rejects_nonfinite_raw_gaussian_opacity(source, tmp_path):
    glb, splat, *_ = source
    data = PlyData.read(splat)['vertex'].data.copy()
    data['opacity'][0] = float('inf')
    PlyData([PlyElement.describe(data, 'vertex')]).write(splat)
    with pytest.raises(ValueError, match='finite'):
        tool().rebake(glb, splat, tmp_path / 'new.glb', tmp_path / 'new.png', texture_size=64)


@pytest.mark.parametrize('case', [
    'negative_accessor', 'bool_accessor', 'float_accessor',
    'negative_view', 'bool_view', 'float_view',
    'bool_count', 'float_count', 'zero_count',
    'unaligned_view', 'unaligned_local', 'unaligned_combined',
    'bool_offset', 'float_offset', 'bool_length', 'float_length',
    'stride_13', 'stride_256', 'bool_stride', 'float_stride',
])
def test_dense_accessor_rejects_illegal_references_layout_and_types(case):
    # Two real float VEC3 payloads; oversized backing data means invalid strides
    # are tested independently of the accessor's end-of-buffer bounds.
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype='<f4')
    binary = bytearray(600)
    binary[:array.nbytes] = array.tobytes()
    accessor = {'bufferView': 0, 'componentType': 5126, 'count': 2, 'type': 'VEC3'}
    view = {'buffer': 0, 'byteOffset': 0, 'byteLength': len(binary)}
    doc = {'buffers': [{'byteLength': len(binary)}], 'bufferViews': [view], 'accessors': [accessor]}
    index = 0
    if case.endswith('accessor'):
        index = {'negative_accessor': -1, 'bool_accessor': False, 'float_accessor': 0.0}[case]
    if case.endswith('view') and case != 'unaligned_view':
        accessor['bufferView'] = {'negative_view': -1, 'bool_view': False, 'float_view': 0.0}[case]
    if case.endswith('count'):
        accessor['count'] = {'bool_count': True, 'float_count': 2.0, 'zero_count': 0}[case]
    if case == 'unaligned_view':
        view['byteOffset'] = 1; view['byteLength'] -= 1
        binary[1:1 + array.nbytes] = array.tobytes()
    if case == 'unaligned_local': accessor['byteOffset'] = 1
    if case == 'unaligned_combined':
        view['byteOffset'] = 2; view['byteLength'] -= 2; accessor['byteOffset'] = 2
    if case.endswith('offset'):
        accessor['byteOffset'] = {'bool_offset': False, 'float_offset': 0.0}[case]
    if case.endswith('length'):
        # A bool length does not fit VEC3, but remains an explicit schema case.
        view['byteLength'] = {'bool_length': True, 'float_length': 600.0}[case]
    if case.startswith('stride_'):
        stride = int(case.split('_')[1]); view['byteStride'] = stride
        binary[stride:stride + 12] = array[1].tobytes()
    if case in ('bool_stride', 'float_stride'):
        view['byteStride'] = True if case == 'bool_stride' else 12.0
    with pytest.raises(ValueError):
        tool().accessor_array(doc, bytes(binary), index)


@pytest.mark.parametrize('case', [
    'negative_position', 'negative_position_view', 'bool_position',
    'negative_texture', 'bool_texture', 'float_texture',
    'negative_image', 'bool_image', 'float_image', 'negative_image_view',
    'strided_indices',
])
def test_rebake_rejects_illegal_reference_or_index_stride(source, tmp_path, case):
    glb, splat, doc, binary = source
    primitive = doc['meshes'][0]['primitives'][0]
    position = primitive['attributes']['POSITION']
    if case == 'negative_position':
        doc['accessors'].append(copy.deepcopy(doc['accessors'][position]))
        primitive['attributes']['POSITION'] = -1
    if case == 'negative_position_view':
        accessor = doc['accessors'][position]
        doc['bufferViews'].append(copy.deepcopy(doc['bufferViews'][accessor['bufferView']]))
        accessor['bufferView'] = -1
    if case == 'bool_position': primitive['attributes']['POSITION'] = True
    info = doc['materials'][0]['pbrMetallicRoughness']['baseColorTexture']
    if case.endswith('texture'):
        info['index'] = {'negative_texture': -1, 'bool_texture': False, 'float_texture': 0.0}[case]
    if case in ('negative_image', 'bool_image', 'float_image'):
        doc['textures'][0]['source'] = {'negative_image': -1, 'bool_image': False, 'float_image': 0.0}[case]
    if case == 'negative_image_view': doc['images'][0]['bufferView'] = -1
    if case == 'strided_indices':
        binary = bytearray(binary)
        binary.extend(b'\0' * (-len(binary) % 4))
        offset = len(binary)
        payload = np.zeros((6, 2), dtype='<u2')
        payload[:, 0] = [0, 1, 2, 0, 2, 3]
        binary.extend(payload.tobytes())
        accessor = doc['accessors'][primitive['indices']]
        accessor['bufferView'] = len(doc['bufferViews'])
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': offset, 'byteLength': payload.nbytes, 'byteStride': 4})
        doc['buffers'][0]['byteLength'] = len(binary)
    pack_glb(glb, doc, bytes(binary))
    with pytest.raises(ValueError):
        tool().rebake(glb, splat, tmp_path / 'new.glb', tmp_path / 'new.png', texture_size=64)
    assert not (tmp_path / 'new.glb').exists()
    assert not (tmp_path / 'new.png').exists()


def test_dense_accessor_keeps_legal_interleaved_float_layout():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype='<f4')
    binary = bytearray(40)
    binary[8:20] = array[0].tobytes()
    binary[24:36] = array[1].tobytes()
    doc = {'buffers': [{'byteLength': 40}],
           'bufferViews': [{'buffer': 0, 'byteOffset': 4, 'byteLength': 36, 'byteStride': 16}],
           'accessors': [{'bufferView': 0, 'byteOffset': 4, 'componentType': 5126, 'count': 2, 'type': 'VEC3'}]}
    np.testing.assert_array_equal(tool().accessor_array(doc, bytes(binary), 0), array)
