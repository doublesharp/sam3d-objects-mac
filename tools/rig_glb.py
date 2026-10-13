"""Append a learned skin to a restricted static GLB, preserving its binary payload."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from tools.rig_validation import validate_rig

_COMPONENTS = {5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}
_WIDTHS = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}


def read_glb(path):
    raw = Path(path).read_bytes()
    if len(raw) < 28 or struct.unpack_from('<III',raw) != (0x46546c67,2,len(raw)):
        raise ValueError('invalid GLB header')
    chunks=[]; offset=12
    while offset < len(raw):
        if offset+8 > len(raw): raise ValueError('truncated GLB chunk')
        size,kind=struct.unpack_from('<II',raw,offset);offset+=8
        if size % 4 or offset+size > len(raw): raise ValueError('invalid GLB chunk bounds')
        chunks.append((kind,raw[offset:offset+size]));offset+=size
    if len(chunks)!=2 or chunks[0][0]!=0x4e4f534a or chunks[1][0]!=0x004e4942:
        raise ValueError('expected one JSON and one embedded binary chunk')
    return json.loads(chunks[0][1]), chunks[1][1]


def accessor_array(document, binary, index):
    accessor=document['accessors'][index]
    if 'sparse' in accessor or accessor.get('normalized'):
        raise ValueError('sparse or normalized accessors are unsupported')
    view=document['bufferViews'][accessor['bufferView']]
    if view.get('buffer',0)!=0: raise ValueError('external buffer is unsupported')
    dtype=np.dtype(_COMPONENTS[accessor['componentType']]);width=_WIDTHS[accessor['type']]
    count=accessor['count'];stride=view.get('byteStride',width*dtype.itemsize)
    base=view.get('byteOffset',0);offset=base+accessor.get('byteOffset',0)
    end=offset+(count-1)*stride+width*dtype.itemsize if count else offset
    if count<0 or stride<width*dtype.itemsize or offset<base or end>base+view['byteLength'] or end>len(binary):
        raise ValueError('accessor exceeds binary buffer bounds')
    return np.ndarray((count,width),dtype=dtype,buffer=binary,offset=offset,strides=(stride,dtype.itemsize)).copy()


def _source(document, binary):
    buffers=document.get('buffers',[])
    if len(buffers)!=1 or 'uri' in buffers[0] or buffers[0]['byteLength']>len(binary):
        raise ValueError('source must use one embedded buffer')
    if document.get('skins') or any('skin' in n for n in document.get('nodes',[])):
        raise ValueError('source already has a skin')
    meshes=document.get('meshes',[])
    if len(meshes)!=1 or len(meshes[0].get('primitives',[]))!=1:
        raise ValueError('source must contain one mesh and one primitive')
    primitive=meshes[0]['primitives'][0]
    if primitive.get('mode',4)!=4 or 'extensions' in primitive or 'targets' in primitive:
        raise ValueError('source must contain uncompressed triangles without morph targets')
    if any(key.startswith(('JOINTS_','WEIGHTS_')) for key in primitive['attributes']):
        raise ValueError('source already has skin attributes')
    nodes=document.get('nodes',[])
    mesh_nodes=[i for i,node in enumerate(nodes) if 'mesh' in node]
    if len(mesh_nodes)!=1 or nodes[mesh_nodes[0]]['mesh']!=0:
        raise ValueError('source must have one uninstanced mesh node')
    mesh_node=mesh_nodes[0]
    # Restrict to a scene-root mesh; transformed ancestors change bind semantics.
    if any(mesh_node in node.get('children',[]) for node in nodes):
        raise ValueError('source mesh must be a scene root without ancestors')
    node=nodes[mesh_node]
    expected={'matrix':np.eye(4).flatten().tolist(),'translation':[0,0,0],'rotation':[0,0,0,1],'scale':[1,1,1]}
    for key,identity in expected.items():
        if key in node and not np.array_equal(node[key],identity):
            raise ValueError('source mesh node must have an identity transform')
    if not any(mesh_node in scene.get('nodes',[]) for scene in document.get('scenes',[])):
        raise ValueError('source mesh must belong to a scene')
    position=primitive['attributes']['POSITION']
    if document['accessors'][position]['componentType']!=5126 or document['accessors'][position]['type']!='VEC3':
        raise ValueError('source POSITION must contain float VEC3 values')
    vertices=accessor_array(document,binary,position)
    if 'indices' in primitive:
        index=primitive['indices'];accessor=document['accessors'][index]
        if accessor['componentType'] not in (5121,5123,5125) or accessor['type']!='SCALAR':
            raise ValueError('source indices must be unsigned scalars')
        flat=accessor_array(document,binary,index).ravel()
    else: flat=np.arange(len(vertices))
    if len(flat)%3:raise ValueError('source triangle index count is not divisible by three')
    return vertices,flat.reshape(-1,3),mesh_node


def source_geometry(path):
    document,binary=read_glb(path)
    vertices,faces,_=_source(document,binary)
    return vertices,faces


def _encode(document,binary):
    binary+=b'\0'*((-len(binary))%4)
    document['buffers'][0]['byteLength']=len(binary)
    encoded=json.dumps(document,separators=(',',':'),allow_nan=False).encode()
    encoded+=b' '*((-len(encoded))%4)
    return struct.pack('<III',0x46546c67,2,28+len(encoded)+len(binary))+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(binary),0x004e4942)+binary


def _serialized(document,binary):
    skin=document['skins'][0];nodes=document['nodes'];joint_nodes=skin['joints']
    parents={}
    for i,node in enumerate(nodes):
        for child in node.get('children',[]):
            if child in parents:raise ValueError('serialized node has multiple parents')
            parents[child]=i
    world=[]
    for index in joint_nodes:
        matrix=np.eye(4);seen=set();current=index
        while current is not None:
            if current in seen:raise ValueError('serialized parent cycle')
            seen.add(current);node=nodes[current]
            if any(key in node for key in ('matrix','scale','rotation')):raise ValueError('unsupported serialized joint transform')
            local=np.eye(4);local[:3,3]=node.get('translation',[0,0,0]);matrix=local@matrix;current=parents.get(current)
        world.append(matrix)
    inverse=accessor_array(document,binary,skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
    transforms=np.asarray(world)@inverse
    primitive=document['meshes'][0]['primitives'][0];attributes=primitive['attributes']
    points=accessor_array(document,binary,attributes['POSITION'])
    ids=accessor_array(document,binary,attributes['JOINTS_0']);weights=accessor_array(document,binary,attributes['WEIGHTS_0'])
    if ids.shape!=(len(points),4) or weights.shape!=ids.shape or np.any(ids>=len(joint_nodes)) or not np.isfinite(weights).all() or np.any(weights<0) or not np.allclose(weights.sum(1),1,atol=1e-6,rtol=0):
        raise ValueError('invalid serialized skin attributes')
    rest=np.sum(np.einsum('nkij,nj->nki',transforms[ids],np.c_[points,np.ones(len(points))])*weights[:,:,None],axis=1)[:,:3]
    bind_error=float(np.max(np.abs(transforms-np.eye(4))))
    rest_error=float(np.max(np.abs(rest-points)))
    tolerance=max(float(np.ptp(points,axis=0).max()),1)*2e-6
    if not np.isfinite(rest).all() or bind_error>tolerance or rest_error>tolerance:
        raise ValueError('serialized bind/rest pose does not preserve source positions')
    return {'bind_identity_max_error':bind_error,'rest_pose_max_error':rest_error,'weights_sum_max_error':float(np.max(np.abs(weights.sum(1)-1)))}


def validate_serialized(path):
    return _serialized(*read_glb(path))


def export_candidate(source, destination, rig):
    source=Path(source);destination=Path(destination)
    if any(path.is_symlink() for path in (source,destination,*source.parents,*destination.parents)) or destination.exists() or source.resolve()==destination.resolve():
        raise ValueError('refusing source overwrite, existing destination or symlink')
    original,source_binary=read_glb(source);document=copy.deepcopy(original)
    points,faces,mesh_node=_source(document,source_binary)
    report=validate_rig(rig,points,faces)
    joints=np.asarray(rig['joints'],dtype=float);parents=np.asarray(rig['parents'],dtype=int);dense=np.asarray(rig['weights'],dtype=float)
    selected_ids=np.argsort(-dense,axis=1,kind='stable')[:,:4]
    selected_weights=np.take_along_axis(dense,selected_ids,axis=1)
    discarded=1-selected_weights.sum(1)
    selected_weights/=selected_weights.sum(1,keepdims=True)
    ids=np.zeros((len(points),4),dtype='<u2');weights=np.zeros((len(points),4),dtype='<f4')
    ids[:,:selected_ids.shape[1]]=selected_ids;weights[:,:selected_weights.shape[1]]=selected_weights;ids[weights==0]=0
    binary=bytearray(source_binary)
    def append(array,kind,component,target=None):
        binary.extend(b'\0'*((-len(binary))%4));offset=len(binary);binary.extend(array.tobytes())
        view={'buffer':0,'byteOffset':offset,'byteLength':array.nbytes}
        if target:view['target']=target
        vi=len(document.setdefault('bufferViews',[]));document['bufferViews'].append(view)
        ai=len(document.setdefault('accessors',[]));document['accessors'].append({'bufferView':vi,'componentType':component,'count':len(array),'type':kind})
        return ai
    attrs=document['meshes'][0]['primitives'][0]['attributes']
    attrs['JOINTS_0']=append(ids,'VEC4',5123,34962);attrs['WEIGHTS_0']=append(weights,'VEC4',5126,34962)
    base=len(document['nodes'])
    for i,point in enumerate(joints):
        parent=parents[i];node={'name':f'SkinTokens_joint_{i}','translation':(point-joints[parent] if parent>=0 else point).tolist()}
        children=np.flatnonzero(parents==i)
        if len(children):node['children']=(base+children).tolist()
        document['nodes'].append(node)
    root=base+int(np.flatnonzero(parents==-1)[0])
    for scene in document['scenes']:
        if mesh_node in scene.get('nodes',[]):scene['nodes'].append(root)
    inverse=np.repeat(np.eye(4)[None],len(joints),axis=0);inverse[:,:3,3]=-joints
    accessor=append(inverse.transpose(0,2,1).copy().astype('<f4'),'MAT4',5126)
    document['skins']=[{'name':'SkinTokens experimental candidate','joints':list(range(base,base+len(joints))),'skeleton':root,'inverseBindMatrices':accessor}]
    document['nodes'][mesh_node]['skin']=0
    encoded=_encode(document,binary)
    report.update(_serialized(document,binary))
    report.update(source_payload_prefix_exact=bytes(binary[:len(source_binary)])==source_binary,source_binary_sha256=hashlib.sha256(source_binary).hexdigest(),top4_discarded_weight_max=float(discarded.max()),top4_discarded_weight_mean=float(discarded.mean()))
    # O_EXCL and O_NOFOLLOW prevent replacement after the initial path checks.
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)
    try:
        fd=os.open(destination,flags,0o644)
    except FileExistsError as error:raise ValueError('candidate destination already exists') from error
    with os.fdopen(fd,'wb') as stream:stream.write(encoded)
    return report
