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


def test_connected_leg_is_not_split_by_longitudinal_quantiles():
    # A broad, continuous leg straddles the old z-band boundary. The rear
    # leg contributes enough samples to place the quantile inside this leg.
    points = [[0, .3, 0]]
    faces = []
    for side in (-1, 1):
        for center, half_width in ((-.26, .03), (-.04, .065)):
            start = len(points)
            for y in np.linspace(-.4, -.21, 12):
                for z in np.linspace(center-half_width, center+half_width, 9):
                    points.append([side*(.4+.15*(-y-.21)), y, z])
            for row in range(11):
                for col in range(8):
                    a = start+row*9+col
                    faces.extend([[a, a+1, a+9], [a+1, a+10, a+9]])
    points = np.asarray(points)
    faces = np.asarray(faces)
    rig = infer_rig(points)
    joints = np.asarray(rig['joint_indices'])
    weights = np.asarray(rig['weights'])
    edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    rest_lengths = np.linalg.norm(points[edges[:, 0]]-points[edges[:, 1]], axis=1)
    rest_normals = np.cross(points[faces[:, 1]]-points[faces[:, 0]], points[faces[:, 2]]-points[faces[:, 0]])
    homogeneous = np.c_[points, np.ones(len(points))]
    for time in np.linspace(0, rig['duration'], 61):
        transforms = np.asarray(rig_matrices(rig, time))
        deformed = np.einsum('nq,nqij,nj->ni', weights, transforms[joints], homogeneous)[:, :3]
        lengths = np.linalg.norm(deformed[edges[:, 0]]-deformed[edges[:, 1]], axis=1)
        assert (lengths/rest_lengths).max() < 1.5
        normals = np.cross(deformed[faces[:, 1]]-deformed[faces[:, 0]], deformed[faces[:, 2]]-deformed[faces[:, 0]])
        assert np.all(np.einsum('ij,ij->i', normals, rest_normals) > 0)


def test_repaired_crab_has_no_detached_or_flipped_patches_over_walk_cycle():
    from pathlib import Path
    import pytest
    from tools.crab_rig import read_glb, accessor_array
    source = Path(__file__).resolve().parents[1]/'outputs/crab-20261012/mesh-repaired.glb'
    if not source.exists():
        pytest.skip('Local generated crab qualification fixture is unavailable')
    document, binary = read_glb(source)
    primitive = document['meshes'][0]['primitives'][0]
    points = accessor_array(document, binary, primitive['attributes']['POSITION']).astype(float)
    faces = accessor_array(document, binary, primitive['indices']).reshape(-1, 3)
    rig = infer_rig(points, faces)
    joints = np.asarray(rig['joint_indices'])
    weights = np.asarray(rig['weights'])
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-7)
    assert (weights >= 0).all() and (joints >= 0).all() and (joints < len(rig['bones'])).all()
    edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(points[edges[:, 0]]-points[edges[:, 1]], axis=1)
    rest_normals = np.cross(points[faces[:, 1]]-points[faces[:, 0]], points[faces[:, 2]]-points[faces[:, 0]])
    homogeneous = np.c_[points, np.ones(len(points))]
    # UV seams use identical weights, and the bind pose preserves all positions.
    _, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    np.testing.assert_array_equal(joints, joints[first][inverse])
    np.testing.assert_array_equal(weights, weights[first][inverse])
    matrices = np.asarray(rig_matrices(rig, 0, animate=False))
    rest = np.einsum('nq,nqij,nj->ni', weights, matrices[joints], homogeneous)[:, :3]
    np.testing.assert_allclose(rest, points, atol=1e-7)
    for time in np.linspace(0, rig['duration'], 61):
        matrices = np.asarray(rig_matrices(rig, time))
        deformed = np.einsum('nq,nqij,nj->ni', weights, matrices[joints], homogeneous)[:, :3]
        moved_lengths = np.linalg.norm(deformed[edges[:, 0]]-deformed[edges[:, 1]], axis=1)
        assert (moved_lengths[lengths > 1e-6]/lengths[lengths > 1e-6]).max() < 2
        normals = np.cross(deformed[faces[:, 1]]-deformed[faces[:, 0]], deformed[faces[:, 2]]-deformed[faces[:, 0]])
        assert (np.einsum('ij,ij->i', normals, rest_normals) >= 0).all()


def test_topology_weights_export_preserves_uv_seams_and_source_payload(tmp_path):
    import trimesh
    from tools.crab_rig import export_rigged_glb, read_glb, accessor_array
    # Two coincident copies model an ordinary texture UV seam.
    mesh = trimesh.creation.box()
    points = np.vstack([mesh.vertices, mesh.vertices])
    faces = np.vstack([mesh.faces, mesh.faces+len(mesh.vertices)])
    rig = infer_rig(points, faces)
    joints = np.asarray(rig['joint_indices'])
    weights = np.asarray(rig['weights'])
    assert weights.shape == (len(points), 4)
    np.testing.assert_array_equal(weights[:8], weights[8:])
    np.testing.assert_array_equal(joints[:8], joints[8:])
    source = tmp_path/'source.glb'
    output = tmp_path/'rigged.glb'
    mesh.export(source)
    before, source_binary = read_glb(source)
    export_rigged_glb(source, output, rig, points)
    after, binary = read_glb(output)
    assert binary[:len(source_binary)] == source_binary
    assert before.get('materials') == after.get('materials')
    assert before.get('images') == after.get('images')
    primitive = after['meshes'][0]['primitives'][0]
    actual_weights = accessor_array(after, binary, primitive['attributes']['WEIGHTS_0'])
    actual_joints = accessor_array(after, binary, primitive['attributes']['JOINTS_0'])
    for matrix in rig_matrices(rig, 0, animate=False):
        np.testing.assert_allclose(matrix, np.eye(4), atol=1e-7)
    np.testing.assert_allclose(actual_weights.sum(axis=1), 1, atol=1e-7)
    assert actual_joints.max() < len(rig['bones'])
    assert np.all(actual_joints[actual_weights == 0] == 0)
