import copy
import json
import struct

import numpy as np
import pytest

from tools.rig_validation import validate_rig, validate_override, validate_tokens
from tools.rig_glb import read_glb, accessor_array, export_candidate, validate_serialized, source_geometry


def source_fixture(path, modify=None):
    points = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], dtype='<f4')
    faces = np.array([[0,1,2],[0,2,3]], dtype='<u2')
    uv = np.array([[0,0],[1,0],[1,1],[0,1]], dtype='<f4')
    binary = points.tobytes()+faces.tobytes()+uv.tobytes()+b'png!'
    doc = {'asset':{'version':'2.0'}, 'buffers':[{'byteLength':len(binary)}],
           'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':48}, {'buffer':0,'byteOffset':48,'byteLength':12}, {'buffer':0,'byteOffset':60,'byteLength':32}, {'buffer':0,'byteOffset':92,'byteLength':4}],
           'accessors':[{'bufferView':0,'componentType':5126,'count':4,'type':'VEC3'}, {'bufferView':1,'componentType':5123,'count':6,'type':'SCALAR'}, {'bufferView':2,'componentType':5126,'count':4,'type':'VEC2'}],
           'meshes':[{'primitives':[{'attributes':{'POSITION':0,'TEXCOORD_0':2},'indices':1,'material':0}]}],
           'materials':[{'pbrMetallicRoughness':{'baseColorTexture':{'index':0}}}], 'textures':[{'source':0}], 'images':[{'bufferView':3,'mimeType':'image/png'}],
           'nodes':[{'mesh':0}], 'scenes':[{'nodes':[0]}], 'scene':0}
    if modify: modify(doc)
    encoded=json.dumps(doc).encode();encoded+=b' '*((-len(encoded))%4)
    path.write_bytes(struct.pack('<III',0x46546c67,2,28+len(encoded)+len(binary))+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(binary),0x004e4942)+binary)
    return points, faces


def rig_fixture(points, faces):
    return dict(vertices=points.copy(),faces=faces.copy(),joints=np.array([[0,0,0],[0,.5,0]]),parents=np.array([-1,0]),weights=np.array([[.9,.1],[.7,.3],[.1,.9],[.2,.8]]))


def test_appended_skin_preserves_binary_uv_materials_and_serialized_pose(tmp_path):
    source=tmp_path/'mesh.glb';dest=tmp_path/'mesh_rig_candidate.glb'
    points,faces=source_fixture(source);before=source.read_bytes();rig=rig_fixture(points,faces)
    report=export_candidate(source,dest,rig)
    old,original=read_glb(source);new,binary=read_glb(dest)
    assert source.read_bytes()==before and binary[:len(original)]==original
    for key in ('materials','images','textures'): assert old[key]==new[key]
    primitive=new['meshes'][0]['primitives'][0]
    assert primitive['attributes']['POSITION']==0 and primitive['attributes']['TEXCOORD_0']==2 and primitive['indices']==1
    ids=accessor_array(new,binary,primitive['attributes']['JOINTS_0']);weights=accessor_array(new,binary,primitive['attributes']['WEIGHTS_0'])
    assert ids.shape==weights.shape==(4,4)
    assert np.all(ids[weights==0]==0)
    np.testing.assert_allclose(weights.sum(1),1)
    assert report['source_payload_prefix_exact'] and report['anatomically_qualified'] is False
    assert validate_serialized(dest)['rest_pose_max_error']<1e-6


@pytest.mark.parametrize('change,match',[
    (lambda r:r.update(parents=np.array([-1,1])), 'cycle'),
    (lambda r:r.update(parents=np.array([-1,3])), 'parent'),
    (lambda r:r.update(parents=np.array([-1,.5])), 'integer'),
    (lambda r:r.update(parents=np.array([-1,-1])), 'root'),
    (lambda r:r.update(weights=np.ones((3,2))*.5), 'weights'),
    (lambda r:r.update(weights=np.ones((4,2))*.4), 'normalized'),
    (lambda r:r['weights'].__setitem__((0,0),-.1), 'nonnegative'),
    (lambda r:r['joints'].__setitem__((1,1),10), 'bounds'),
    (lambda r:r['joints'].__setitem__((1,1),0), 'zero.length'),
    (lambda r:r['vertices'].__setitem__((0,0),.1), 'source vertices'),
    (lambda r:r['faces'].__setitem__((0,0),2), 'source faces'),
])
def test_reject_invalid_rigs(tmp_path,change,match):
    source=tmp_path/'mesh.glb';points,faces=source_fixture(source);rig=rig_fixture(points,faces);change(rig)
    with pytest.raises(ValueError,match=match): validate_rig(rig,points,faces)


def test_override_requires_exact_joints_and_parents(tmp_path):
    source=tmp_path/'mesh.glb';points,faces=source_fixture(source);rig=rig_fixture(points,faces)
    override={'joints':rig['joints'].tolist(),'parents':rig['parents'].tolist()}
    validate_override(rig,override)
    rig['joints'][1,1]+=1e-9
    with pytest.raises(ValueError,match='exact'): validate_override(rig,override)


def test_incomplete_tokens_rejected():
    tokens=np.array([1,2,3,4,5,6,7,8,9,10,11,100])
    result={'status':'inference_passed','tokens':12,'tokens_eos':100,'final_token':100,'skeleton_eos_present':True,'global_eos_present':True,'skin_tokens_valid':True,'skin_token_count':8,'counted_skeleton_bones':2,'skeleton_eos':3,'skin_vocab_start':4,'tokens_per_skin':4}
    validate_tokens(tokens,result,2)
    for key,value in [('status','failed'),('final_token',4),('skin_tokens_valid',False),('skin_token_count',7),('skeleton_eos_present',False),('tokens',11),('counted_skeleton_bones',3)]:
        broken={**result,key:value}
        with pytest.raises(ValueError):validate_tokens(tokens,broken,2)


@pytest.mark.parametrize('modify',[
    lambda d:d['nodes'][0].update(translation=[1,0,0]),
    lambda d:d.update(skins=[{}]),
    lambda d:d['meshes'][0]['primitives'].append(copy.deepcopy(d['meshes'][0]['primitives'][0])),
    lambda d:d['nodes'].append({'mesh':0}),
    lambda d:d['nodes'].append({'children':[0],'translation':[1,0,0]}),
    lambda d:d['meshes'][0]['primitives'][0].update(mode=1),
])
def test_source_rejects_unsupported_scene(tmp_path,modify):
    source=tmp_path/'mesh.glb';source_fixture(source,modify)
    with pytest.raises(ValueError):source_geometry(source)


def test_never_overwrite_or_follow_symlink(tmp_path):
    source=tmp_path/'mesh.glb';points,faces=source_fixture(source);rig=rig_fixture(points,faces);before=source.read_bytes()
    for target in (source,tmp_path/'existing.glb',tmp_path/'link.glb'):
        if target.name=='existing.glb':target.write_bytes(b'keep')
        if target.name=='link.glb':target.symlink_to(source)
        with pytest.raises(ValueError):export_candidate(source,target,rig)
    assert source.read_bytes()==before


def test_serialized_validation_rejects_corrupt_bind(tmp_path):
    source=tmp_path/'mesh.glb';points,faces=source_fixture(source);dest=tmp_path/'candidate.glb';export_candidate(source,dest,rig_fixture(points,faces))
    document,binary=read_glb(dest);acc=document['accessors'][document['skins'][0]['inverseBindMatrices']];view=document['bufferViews'][acc['bufferView']]
    raw=bytearray(dest.read_bytes());size=struct.unpack_from('<I',raw,12)[0];offset=28+size+view['byteOffset'];struct.pack_into('<f',raw,offset,2);dest.write_bytes(raw)
    with pytest.raises(ValueError,match='bind|rest'):validate_serialized(dest)


def test_tokens_independently_checked_against_model_vocabulary():
    tokens=np.array([1,2,3,4,5,6,7,8,9,10,11,100])
    result={'status':'inference_passed','tokens':12,'tokens_eos':100,'final_token':100,'skeleton_eos_present':True,'global_eos_present':True,'skin_tokens_valid':True,'skin_token_count':8,'counted_skeleton_bones':2,'skeleton_eos':3,'skin_vocab_start':4,'tokens_per_skin':4}
    for index,value in [(2,2),(3,3),(7,100)]:
        broken=tokens.copy();broken[index]=value
        with pytest.raises(ValueError):validate_tokens(broken,result,2)
    for field in ('skeleton_eos','skin_vocab_start','tokens_per_skin'):
        broken=result.copy();del broken[field]
        with pytest.raises(ValueError):validate_tokens(tokens,broken,2)


def test_export_rejects_symlink_parent_and_source(tmp_path):
    source=tmp_path/'mesh.glb';points,faces=source_fixture(source);rig=rig_fixture(points,faces)
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir();alias=tmp_path/'alias';alias.symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError):export_candidate(source,alias/'candidate.glb',rig)
    assert not (elsewhere/'candidate.glb').exists()
    alias_source=tmp_path/'alias-source.glb';alias_source.symlink_to(source)
    with pytest.raises(ValueError):export_candidate(alias_source,tmp_path/'candidate.glb',rig)


def test_five_joint_out_of_order_tree_and_top_four_report(tmp_path):
    source=tmp_path/'mesh.glb';points,faces=source_fixture(source);rig=rig_fixture(points,faces)
    rig.update(joints=np.array([[0,.3,0],[.1,.5,0],[0,0,0],[-.1,.2,0],[-.2,.4,0]]),parents=np.array([2,0,-1,2,3]),weights=np.tile([.05,.15,.5,.1,.2],(4,1)))
    report=export_candidate(source,tmp_path/'candidate.glb',rig)
    assert report['rest_pose_max_error']<1e-6
    assert report['top4_discarded_weight_max']==pytest.approx(.05)


def test_nonfinite_joint_and_weight_values_rejected(tmp_path):
    source=tmp_path/'mesh.glb';points,faces=source_fixture(source)
    for array,value in [('joints',np.nan),('weights',np.inf)]:
        rig=rig_fixture(points,faces);rig[array][0,0]=value
        with pytest.raises(ValueError):validate_rig(rig,points,faces)
