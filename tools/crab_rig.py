"""Approximate automatic crab rig inferred from y-up reconstruction geometry.

This is a procedural first pass, not anatomical reconstruction. It estimates
up to six leg chains and two claws and exports an editable glTF skin and walk cycle.
Pass triangle faces to infer_rig for continuous mesh attachment weights.
"""
import json
import struct
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


def _connected_limbs(points, candidates, seed_cut, bottom_cut, faces=None):
    """Find whole distal limbs before assigning a chain, rather than z slices."""
    seeds = np.flatnonzero(candidates & (points[:, 1] < seed_cut))
    if len(seeds) < 6:
        return []
    if faces is None:
        tree = cKDTree(points[seeds])
        distances, _ = tree.query(points[seeds], k=min(5, len(seeds)))
        radius = max(float(np.median(distances[:, -1])) * 1.7, 1e-6)
        edges = tree.query_pairs(radius, output_type='ndarray')
    else:
        lookup = np.full(len(points), -1, dtype=int)
        lookup[seeds] = np.arange(len(seeds))
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = lookup[edges]
        edges = edges[(edges >= 0).all(axis=1)]
    graph = sparse.coo_matrix((np.ones(2*len(edges)),
                              (np.r_[edges[:, 0], edges[:, 1]],
                               np.r_[edges[:, 1], edges[:, 0]])),
                             shape=(len(seeds), len(seeds))).tocsr()
    _, labels = connected_components(graph)
    regions = []
    for label in np.unique(labels):
        region = seeds[labels == label]
        if len(region) >= 3 and points[region, 1].min() < bottom_cut:
            regions.append(region)
    # Keep the supported maximum of three chains per side. Empty chains remain
    # editable, but are never allowed to bisect a detected connected limb.
    regions = sorted(regions, key=len, reverse=True)[:3]
    return sorted(regions, key=lambda region: float(points[region, 2].mean()))


def _smooth_mesh_weights(points, faces, rig, limb_seeds, ycut, width, zfront):
    """Harmonic attachment weights on welded geometry, including UV seams."""
    joints = np.asarray(rig['joint_indices'])
    weights = np.asarray(rig['weights'])
    dense = np.zeros((len(points), len(rig['bones'])))
    np.add.at(dense, (np.arange(len(points))[:, None], joints), weights)
    pinned = (points[:, 1] >= ycut+.04*width) | ((np.abs(points[:, 0]) < width*.45) & (points[:, 2] < zfront-.10*width))
    dense[pinned] = 0
    dense[pinned, 0] = 1
    for region in limb_seeds:
        pinned[region] = True
    claw_seeds = (points[:, 1] < ycut-.06*width) & (points[:, 2] > zfront+.12*width)
    pinned[claw_seeds] = True
    for side, joint in ((-1, 13), (1, 14)):
        mask = claw_seeds & (points[:, 0]*side > 0)
        dense[mask] = 0
        dense[mask, joint] = 1
    # The cotangent energy respects triangle shape. Inverse edge lengths alone
    # let a nearly collinear triangle's middle vertex receive a nonlinear weight
    # and turn over during an otherwise mild bend. Clamp negative conductance to
    # retain nonnegative harmonic weights on obtuse reconstruction triangles.
    corners = points[faces]
    area2 = np.linalg.norm(np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0]), axis=1)
    edge_sets = []
    cotangents = []
    for corner, a, b in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        edge_sets.append(np.sort(faces[:, [a, b]], axis=1))
        cotangents.append(np.einsum('ij,ij->i', corners[:, a]-corners[:, corner], corners[:, b]-corners[:, corner])/np.maximum(area2, width**2*1e-12))
    edges, mapping = np.unique(np.vstack(edge_sets), axis=0, return_inverse=True)
    conductance = np.zeros(len(edges))
    np.add.at(conductance, mapping, np.concatenate(cotangents))
    conductance = np.maximum(conductance, 1e-8)
    graph = sparse.coo_matrix((np.r_[conductance, conductance],
                              (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                             shape=(len(points), len(points))).tocsr()
    _, components = connected_components(graph)
    for component in np.unique(components):
        selected = components == component
        if not pinned[selected].any():
            pinned[selected] = True
            dense[selected] = 0
            dense[selected, 0] = 1
    free = np.flatnonzero(~pinned)
    fixed = np.flatnonzero(pinned)
    if len(free):
        laplacian = sparse.diags(np.asarray(graph.sum(axis=1)).ravel())-graph
        dense[free] = spsolve(laplacian[free][:, free].tocsc(), -(laplacian[free][:, fixed] @ dense[fixed]))
    dense = np.maximum(dense, 0)
    indices = np.argsort(-dense, axis=1, kind='stable')[:, :4]
    weights = np.take_along_axis(dense, indices, axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    indices[weights == 0] = 0
    rig['joint_indices'] = indices.tolist()
    rig['weights'] = weights.tolist()


def infer_rig(points, faces=None):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected finite, nonempty [N,3] y-up points')
    if faces is not None:
        faces = np.asarray(faces, dtype=int)
        if faces.ndim != 2 or faces.shape[1] != 3 or (faces < 0).any() or (faces >= len(points)).any():
            raise ValueError('Expected valid [F,3] triangle indices')
        # Coincident UV-split vertices must share the same deformation.
        points, unmap = np.unique(points, axis=0, return_inverse=True)
        faces = unmap[faces]
    lo,hi=points.min(0),points.max(0)
    width=max(abs(lo[0]),abs(hi[0]),1e-3)
    ycut=lo[1]+.34*(hi[1]-lo[1])
    zfront=lo[2]+.73*(hi[2]-lo[2])
    bones=[{'name':'Body','parent':-1,'position':[0.,0.,0.],'kind':'body'}]
    indices=np.zeros((len(points),2),dtype=int)
    weights=np.zeros((len(points),2));weights[:,0]=1
    lower=points[:,1]<ycut
    claws=lower&(points[:,2]>zfront)&(np.abs(points[:,0])<width*.82)
    limb_seeds = []
    for side_index,side in enumerate((-1,1)):
        candidates=lower&(points[:,0]*side>width*.52)&~claws
        side_points=points[candidates]
        if len(side_points)<3:
            side_points=np.array([[side*width*.8,lo[1],z] for z in np.linspace(lo[2],zfront,3)])
        regions = _connected_limbs(points, candidates,
                                   lo[1]+.27*(hi[1]-lo[1]),
                                   lo[1]+.12*(hi[1]-lo[1]), faces)
        cuts=np.quantile(side_points[:,2],[1/3,2/3])
        bands=np.digitize(points[:,2],cuts)
        if regions:
            seed_points = np.concatenate(regions)
            seed_bands = np.concatenate([np.full(len(region), band) for band, region in enumerate(regions)])
            _, nearest = cKDTree(points[seed_points]).query(points)
            bands = seed_bands[nearest]
            limb_seeds.extend(regions)
        for band in range(3):
            selection=candidates&(bands==band)
            if regions and band >= len(regions):
                selection[:] = False
            cloud=points[selection]
            if len(cloud):
                z=float(np.median(cloud[:,2]))
                distal=cloud[cloud[:,1]<=np.quantile(cloud[:,1],.25)]
                foot=distal.mean(0)
            else:
                z=float(np.quantile(side_points[:,2],(band+.5)/3))
                foot=np.array([side*width*.85,lo[1],z])
            hip=np.array([side*width*.52,ycut,z])
            knee=hip*.45+foot*.55
            hip_index=len(bones)
            bones.append({'name':f'{"Left" if side<0 else "Right"}Leg{band+1}Hip','parent':0,'position':hip.tolist(),'kind':'hip','side':side,'phase':float(((side_index+band)%2)*np.pi)})
            knee_index=len(bones)
            bones.append({'name':f'{"Left" if side<0 else "Right"}Leg{band+1}Knee','parent':hip_index,'position':knee.tolist(),'kind':'knee','side':side,'phase':float(((side_index+band)%2)*np.pi)})
            if selection.any():
                axis=knee-hip
                t=np.clip((points[selection]-hip)@axis/max(float(axis@axis),1e-6),0,1)
                indices[selection]=[hip_index,knee_index]
                weights[selection,0]=1-t;weights[selection,1]=t
    for side in (-1,1):
        selection=claws&(points[:,0]*side>0)
        joint=len(bones)
        bones.append({'name':'LeftClaw' if side<0 else 'RightClaw','parent':0,'position':[float(side*width*.38),float(ycut),float(zfront)],'kind':'claw','side':side})
        indices[selection]=[joint,0];weights[selection]=[1,0]
    rig = {'bones':bones,'joint_indices':indices.tolist(),'weights':weights.tolist(),'duration':1.8,'method':'Connected distal limb segmentation; walk in place'}
    if faces is not None:
        _smooth_mesh_weights(points, faces, rig, limb_seeds, ycut, width, zfront)
        rig['method'] = 'Connected distal limb segmentation with harmonic mesh attachment weights; walk in place'
        rig['joint_indices'] = np.asarray(rig['joint_indices'])[unmap].tolist()
        rig['weights'] = np.asarray(rig['weights'])[unmap].tolist()
    return rig


def walk_pose(rig,t):
    phase=2*np.pi*t/rig['duration']
    rotations=np.zeros((len(rig['bones']),4));rotations[:,3]=1
    for i,bone in enumerate(rig['bones']):
        kind=bone['kind']
        if kind=='body':rotations[i]=Rotation.from_rotvec([0,.025*np.sin(phase),0]).as_quat()
        elif kind in ('hip','knee'):
            wave=np.sin(phase+bone['phase'])
            angle=bone['side']*(.19*wave if kind=='hip' else -.30*max(0.,wave))
            rotations[i]=Rotation.from_rotvec([0,0,angle]).as_quat()
        else:rotations[i]=Rotation.from_rotvec([.065*np.sin(phase),0,0]).as_quat()
    return {'rotations':rotations,'root_translation':np.array([0,.012*(1-np.cos(2*phase)),0])}


def rig_matrices(rig,t,animate=True):
    pose=walk_pose(rig,t) if animate else {'rotations':np.tile([0,0,0,1.],(len(rig['bones']),1)),'root_translation':np.zeros(3)}
    worlds=[];skins=[]
    for i,bone in enumerate(rig['bones']):
        pos=np.array(bone['position']);parent=bone['parent']
        local=np.eye(4);local[:3,:3]=Rotation.from_quat(pose['rotations'][i]).as_matrix()
        local[:3,3]=pos-(np.array(rig['bones'][parent]['position']) if parent>=0 else 0)
        if parent<0:local[:3,3]+=pose['root_translation']
        world=worlds[parent]@local if parent>=0 else local
        worlds.append(world)
        inv=np.eye(4);inv[:3,3]=-pos
        skins.append(world@inv)
    return skins


def read_glb(path):
    data=Path(path).read_bytes()
    magic,version,length=struct.unpack_from('<III',data)
    if magic!=0x46546C67 or version!=2 or length!=len(data):raise ValueError('Invalid GLB')
    document=None;binary=None;offset=12
    while offset<len(data):
        size,kind=struct.unpack_from('<II',data,offset);offset+=8
        chunk=data[offset:offset+size];offset+=size
        if kind==0x4E4F534A:document=json.loads(chunk)
        elif kind==0x004E4942:binary=bytearray(chunk)
    if document is None or binary is None:raise ValueError('GLB must contain JSON and binary chunks')
    return document,binary


def accessor_array(document,binary,index):
    a=document['accessors'][index];v=document['bufferViews'][a['bufferView']]
    types={5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}
    widths={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
    dtype=np.dtype(types[a['componentType']]);width=widths[a['type']]
    offset=v.get('byteOffset',0)+a.get('byteOffset',0)
    stride=v.get('byteStride',width*dtype.itemsize)
    return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,offset=offset,strides=(stride,dtype.itemsize)).copy()


def export_rigged_glb(source,destination,rig,rig_points):
    document,binary=read_glb(source)
    if len(document['buffers'])!=1:raise ValueError('Expected a single embedded GLB buffer')
    def append_accessor(array,kind,component=5126,target=None):
        array=np.ascontiguousarray(array)
        binary.extend(b'\0'*((-len(binary))%4));offset=len(binary);binary.extend(array.tobytes())
        view={'buffer':0,'byteOffset':offset,'byteLength':array.nbytes}
        if target:view['target']=target
        vi=len(document.setdefault('bufferViews',[]));document['bufferViews'].append(view)
        accessor={'bufferView':vi,'componentType':component,'count':len(array),'type':kind}
        if kind in ('SCALAR','VEC3') and component==5126:
            accessor['min']=array.reshape(len(array),-1).min(0).tolist();accessor['max']=array.reshape(len(array),-1).max(0).tolist()
        ai=len(document.setdefault('accessors',[]));document['accessors'].append(accessor);return ai
    nodes=document.setdefault('nodes',[]);base=len(nodes)
    for bone in rig['bones']:
        pos=np.asarray(bone['position']);parent=bone['parent']
        translation=pos-(np.asarray(rig['bones'][parent]['position']) if parent>=0 else 0)
        nodes.append({'name':bone['name'],'translation':translation.tolist()})
    for i,bone in enumerate(rig['bones']):
        if bone['parent']>=0:nodes[base+bone['parent']].setdefault('children',[]).append(base+i)
    for scene in document['scenes']:scene.setdefault('nodes',[]).append(base)
    inverse=[]
    for bone in rig['bones']:
        matrix=np.eye(4,dtype=np.float32);matrix[:3,3]=-np.asarray(bone['position'])
        inverse.append(matrix.T.ravel())
    inverse_accessor=append_accessor(np.asarray(inverse,dtype='<f4'),'MAT4')
    skin_index=len(document.setdefault('skins',[]))
    document['skins'].append({'name':'Crab automatic rig','inverseBindMatrices':inverse_accessor,'skeleton':base,'joints':list(range(base,base+len(rig['bones'])))})
    tree=cKDTree(rig_points)
    rig_indices=np.asarray(rig['joint_indices']);rig_weights=np.asarray(rig['weights'])
    for node in nodes[:base]:
        if 'mesh' not in node:continue
        for field,identity in [('matrix',np.eye(4).ravel()),('translation',[0,0,0]),('rotation',[0,0,0,1]),('scale',[1,1,1])]:
            if field in node and not np.allclose(node[field],identity):raise ValueError('Normalize mesh node transforms before rigging')
        node['skin']=skin_index
        for primitive in document['meshes'][node['mesh']]['primitives']:
            points=accessor_array(document,binary,primitive['attributes']['POSITION'])
            _,nearest=tree.query(points,workers=2)
            joint_array=np.zeros((len(points),4),dtype='<u2');joint_array[:,:rig_indices.shape[1]]=rig_indices[nearest]
            weight_array=np.zeros((len(points),4),dtype='<f4');weight_array[:,:rig_weights.shape[1]]=rig_weights[nearest]
            joint_array[weight_array==0]=0
            primitive['attributes']['JOINTS_0']=append_accessor(joint_array,'VEC4',5123,34962)
            primitive['attributes']['WEIGHTS_0']=append_accessor(weight_array,'VEC4',5126,34962)
    times=np.linspace(0,rig['duration'],61,dtype='<f4')
    time_accessor=append_accessor(times.reshape(-1,1),'SCALAR')
    poses=[walk_pose(rig,float(t)) for t in times]
    animation={'name':'Walk in place','samplers':[],'channels':[]}
    for i,bone in enumerate(rig['bones']):
        values=np.asarray([p['rotations'][i] for p in poses],dtype='<f4')
        output=append_accessor(values,'VEC4');sampler=len(animation['samplers'])
        animation['samplers'].append({'input':time_accessor,'output':output,'interpolation':'LINEAR'})
        animation['channels'].append({'sampler':sampler,'target':{'node':base+i,'path':'rotation'}})
    values=np.asarray([p['root_translation'] for p in poses],dtype='<f4')
    output=append_accessor(values,'VEC3');sampler=len(animation['samplers'])
    animation['samplers'].append({'input':time_accessor,'output':output,'interpolation':'LINEAR'})
    animation['channels'].append({'sampler':sampler,'target':{'node':base,'path':'translation'}})
    document.setdefault('animations',[]).append(animation)
    document['asset']['generator']='SAM 3D MPS port + approximate procedural crab rig'
    document['extras']={'rig_method':rig['method'],'bones':len(rig['bones'])}
    document['buffers'][0]['byteLength']=len(binary)
    text=json.dumps(document,separators=(',',':')).encode();text+=b' '*((-len(text))%4)
    binary.extend(b'\0'*((-len(binary))%4))
    length=12+8+len(text)+8+len(binary)
    Path(destination).write_bytes(struct.pack('<III',0x46546C67,2,length)+struct.pack('<II',len(text),0x4E4F534A)+text+struct.pack('<II',len(binary),0x004E4942)+binary)
