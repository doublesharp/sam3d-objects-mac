import numpy as np
from tools.crab_rig import infer_rig, rig_matrices, walk_pose


def test_rig_weights_and_rest_pose():
    points=np.array([[0,.25,0],[-.42,-.3,-.2],[.42,-.3,-.2],[-.4,-.32,0],[.4,-.32,0],[-.42,-.3,.14],[.42,-.3,.14],[-.2,-.25,.3],[.2,-.25,.3]],dtype=float)
    rig=infer_rig(points)
    weights=np.asarray(rig['weights']);joints=np.asarray(rig['joint_indices'])
    assert weights.shape==(len(points),2)
    np.testing.assert_allclose(weights.sum(1),1)
    assert np.all(weights>=0) and np.all(weights<=1)
    assert joints.max()<len(rig['bones']) and joints.min()>=0
    matrices=rig_matrices(rig,0,animate=False)
    for matrix in matrices:np.testing.assert_allclose(matrix,np.eye(4),atol=1e-7)
    assert len(rig['bones'])==15
    assert joints[0,0]==0


def test_walk_is_finite_cyclic_and_moves_limbs():
    points=np.array([[0,.3,0],[-.4,-.3,-.2],[.4,-.3,-.2],[-.4,-.3,0],[.4,-.3,0],[-.4,-.3,.12],[.4,-.3,.12]])
    rig=infer_rig(points)
    first=walk_pose(rig,0);last=walk_pose(rig,rig['duration'])
    np.testing.assert_allclose(first['rotations'],last['rotations'],atol=1e-7)
    np.testing.assert_allclose(first['root_translation'],last['root_translation'],atol=1e-7)
    mid=walk_pose(rig,rig['duration']/4)
    assert np.isfinite(mid['rotations']).all()
    np.testing.assert_allclose(np.linalg.norm(mid['rotations'],axis=1),1,atol=1e-7)
    assert not np.allclose(mid['rotations'],first['rotations'])
    assert all(np.isfinite(m).all() for m in rig_matrices(rig,.5))


def test_glb_export_skin_and_animation_round_trip(tmp_path):
    import trimesh
    from tools.crab_rig import export_rigged_glb, read_glb, accessor_array
    mesh=trimesh.creation.box()
    points=np.asarray(mesh.vertices)
    rig=infer_rig(points)
    rig['joint_indices'][0]=[1,2]
    rig['weights'][0]=[0,1]
    source=tmp_path/'source.glb';output=tmp_path/'rigged.glb'
    mesh.export(source)
    export_rigged_glb(source,output,rig,points)
    document,binary=read_glb(output)
    primitive=document['meshes'][0]['primitives'][0]
    joints=accessor_array(document,binary,primitive['attributes']['JOINTS_0'])
    weights=accessor_array(document,binary,primitive['attributes']['WEIGHTS_0'])
    np.testing.assert_allclose(weights.sum(1),1,atol=1e-7)
    assert np.all(joints[weights==0]==0)
    skin=document['skins'][0]
    assert len(skin['joints'])==15
    inverse=accessor_array(document,binary,skin['inverseBindMatrices'])
    for i,bone in enumerate(rig['bones']):
        bind=np.eye(4);bind[:3,3]=bone['position']
        np.testing.assert_allclose(bind@inverse[i].reshape(4,4).T,np.eye(4),atol=1e-7)
    animation=document['animations'][0]
    assert animation['name']=='Walk in place'
    assert len(animation['channels'])==16
    for sampler in animation['samplers']:
        times=accessor_array(document,binary,sampler['input'])[:,0]
        assert np.all(np.diff(times)>0)
        values=accessor_array(document,binary,sampler['output'])
        assert np.isfinite(values).all()
        np.testing.assert_allclose(values[0],values[-1],atol=1e-6)
