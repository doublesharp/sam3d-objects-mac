"""Bounded CPU surface repair, separate from appearance generation and rigging.

Healthy meshes bypass remeshing. Defective meshes are rebuilt from a fine
unsigned-distance shell, with small gaps closed before exterior flood fill.
Every candidate is checked against the source before it may be exported.
"""
from dataclasses import dataclass
import time
import numpy as np
import trimesh


class RepairRejected(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


@dataclass
class RepairResult:
    vertices: np.ndarray
    faces: np.ndarray
    report: dict


def _arrays(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Expected finite, nonempty [N,3] vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Expected valid nonempty [N,3] triangle indices')
    return vertices, faces


def _mesh(vertices, faces):
    # Geometry-only welding: UV seams must not count as open boundaries.
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    return trimesh.Trimesh(vertices, inverse[faces], process=False)


def topology(vertices, faces):
    mesh = _mesh(vertices, faces)
    _, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    return dict(vertices=len(mesh.vertices), triangles=len(mesh.faces),
                boundary_edges=int((counts == 1).sum()), nonmanifold_edges=int((counts > 2).sum()),
                watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                euler_number=int(mesh.euler_number), bounds=mesh.bounds.tolist())


def _scene(vertices, faces):
    import open3d as o3d
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.ascontiguousarray(vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.ascontiguousarray(faces), dtype=o3d.core.Dtype.UInt32))
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(mesh)
    return scene


def _distance(scene, points):
    import open3d as o3d
    output = []
    for start in range(0, len(points), 150_000):
        tensor = o3d.core.Tensor(np.ascontiguousarray(points[start:start+150_000]), dtype=o3d.core.Dtype.Float32)
        output.append(scene.compute_distance(tensor, nthreads=8).numpy())
    return np.concatenate(output)


def validate_repair(source_vertices, source_faces, vertices, faces,
                    max_p95=.035, max_error=.10, min_bounds=.97):
    source_vertices, source_faces = _arrays(source_vertices, source_faces)
    vertices, faces = _arrays(vertices, faces)
    after = topology(vertices, faces)
    report = {'after': after, 'status': 'rejected'}
    if not after['watertight'] or not after['winding_consistent'] or after['nonmanifold_edges']:
        raise RepairRejected('Repair rejected: surface is not closed and manifold', report)
    scale = max(float(np.linalg.norm(np.ptp(source_vertices, axis=0))), 1e-8)
    extents = np.maximum(np.ptp(source_vertices, axis=0), scale*1e-5)
    retention = np.ptp(vertices, axis=0)/extents
    source_dist = _distance(_scene(vertices, faces), source_vertices)/scale
    candidate_dist = _distance(_scene(source_vertices, source_faces), vertices)/scale
    source_mesh = _mesh(source_vertices, source_faces)
    candidate_mesh = _mesh(vertices, faces)
    source_volume = abs(float(source_mesh.volume))
    candidate_volume = abs(float(candidate_mesh.volume))
    source_samples=trimesh.sample.sample_surface(source_mesh,8192,seed=19)[0].astype(np.float32)
    candidate_samples=trimesh.sample.sample_surface(candidate_mesh,8192,seed=19)[0].astype(np.float32)
    sampled_source_dist=_distance(_scene(vertices,faces),source_samples)/scale
    sampled_candidate_dist=_distance(_scene(source_vertices,source_faces),candidate_samples)/scale
    report.update(surface_error_p95=float(max(np.quantile(sampled_source_dist,.95), np.quantile(sampled_candidate_dist,.95))),
                  source_error_max=float(source_dist.max()), candidate_error_max=float(candidate_dist.max()),
                  bounds_retention_min=float(retention.min()), bounds_expansion_max=float(retention.max()),
                  scale=scale)
    report.update(source_volume=source_volume, candidate_volume=candidate_volume)
    if source_mesh.is_winding_consistent and source_volume > scale**3*1e-5 and candidate_volume < source_volume*.7:
        raise RepairRejected('Repair rejected: shell volume lost the expected interior', report)
    if retention.min() < min_bounds or retention.max() > 1.12:
        raise RepairRejected('Repair rejected: shape bounds lost detail or expanded too far', report)
    if report['surface_error_p95'] > max_p95 or max(report['source_error_max'],report['candidate_error_max']) > max_error:
        raise RepairRejected('Repair rejected: surface shape changed too far', report)
    report['status'] = 'repaired'
    return report


def close_small_boundaries(mesh):
    """Cap only small simple loops; never guess a large missing surface."""
    vertices=np.asarray(mesh.vertices,dtype=np.float32)
    faces=np.asarray(mesh.faces,dtype=np.int64)
    edges=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    unique,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    boundary=unique[counts==1]
    adjacency={}
    for a,b in boundary:
        adjacency.setdefault(int(a),[]).append(int(b));adjacency.setdefault(int(b),[]).append(int(a))
    if not adjacency:return mesh,0
    scale=float(np.linalg.norm(np.ptp(vertices,axis=0)))
    unvisited={tuple(edge) for edge in boundary.tolist()};new_vertices=[];caps=[];closed=0
    directions={tuple(sorted(edge)):tuple(edge) for edge in edges.tolist() if tuple(sorted(edge)) in unvisited}
    while unvisited:
        start,current=next(iter(unvisited));loop=[start];previous=start
        unvisited.discard(tuple(sorted((start,current))))
        while current!=start and len(loop)<=128:
            loop.append(current)
            choices=[v for v in adjacency.get(current,[]) if v!=previous]
            if len(choices)!=1:break
            following=choices[0];edge=tuple(sorted((current,following)))
            if edge not in unvisited:break
            unvisited.discard(edge);previous,current=current,following
        if current!=start or len(loop)>128:continue
        cloud=vertices[loop];center=cloud.mean(0)
        area=np.linalg.norm(np.cross(cloud-center,np.roll(cloud,-1,axis=0)-center),axis=1).sum()*.5
        if area>scale*scale*.005 or np.linalg.norm(np.ptp(cloud,axis=0))>scale*.15:continue
        middle=len(vertices)+len(new_vertices);new_vertices.append(center)
        for a,b in zip(loop,np.roll(loop,-1)):
            forward=directions[tuple(sorted((int(a),int(b))))]==(int(a),int(b))
            caps.append((int(b),int(a),middle) if forward else (int(a),int(b),middle))
        closed+=1
    if not caps:return mesh,0
    return trimesh.Trimesh(np.vstack([vertices,new_vertices]),np.vstack([faces,caps]),process=False),closed


def extract_closed_surface(solid, pitch, origin):
    from skimage.measure import marching_cubes
    # An exact half value creates zero-area contacts in ambiguous binary cells.
    # A tiny inward offset resolves them without a measurable shape change.
    output, faces, _, _ = marching_cubes(solid.astype(np.float32), level=.501,
                                        spacing=(pitch,)*3, allow_degenerate=False)
    return output + origin, faces


def choose_repair_settings(vertices):
    points=np.unique(np.asarray(vertices,dtype=np.float32),axis=0)
    axes=np.linalg.eigh(np.cov(points.T))[1]
    extent=np.ptp((points-points.mean(0))@axes,axis=0)
    thin_ratio=float(extent.min()/max(extent.max(),1e-8))
    return (224,2) if thin_ratio<.5 else (192,4)


def optional_repair_setting(value):
    return None if value is None or str(value).lower()=="auto" else int(value)


def repair_surface(vertices, faces, resolution=None, closing_radius=None, force=False, target_faces=160_000):
    from scipy import ndimage
    started = time.perf_counter()
    vertices, faces = _arrays(vertices, faces)
    before = topology(vertices, faces)
    if not force and before['watertight'] and before['winding_consistent'] and not before['nonmanifold_edges']:
        return RepairResult(vertices.copy(), faces.copy(),
                            {'status':'unchanged','before':before,'after':before,'seconds':time.perf_counter()-started})
    auto_resolution, auto_radius=choose_repair_settings(vertices)
    resolution=auto_resolution if resolution is None else resolution
    closing_radius=auto_radius if closing_radius is None else closing_radius
    if not 32 <= int(resolution) <= 320 or not 0 <= int(closing_radius) <= 6:
        raise ValueError('Repair resolution must be 32..320 and closing radius 0..6 voxels')
    resolution, closing_radius = int(resolution), int(closing_radius)
    mesh, closed_loops = close_small_boundaries(_mesh(vertices, faces))
    pitch = float(np.ptp(vertices, axis=0).max()) / resolution
    # Any remaining large boundary is outside the bounded repair contract.
    # A flood fill cannot infer an interior through it, even if source volume is zero.
    edges=np.sort(np.asarray(mesh.edges),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    boundary=unique[counts==1]
    if len(boundary):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        ids,inverse=np.unique(boundary,return_inverse=True);pairs=inverse.reshape(-1,2)
        graph=coo_matrix((np.ones(len(pairs)*2),
                         (np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(ids),len(ids)))
        components,labels=connected_components(graph,directed=False)
        scale=float(np.linalg.norm(np.ptp(vertices,axis=0)))
        for component in range(components):
            cloud=np.asarray(mesh.vertices)[ids[labels==component]]
            if np.linalg.norm(np.ptp(cloud,axis=0))>min(scale*.15,(2*closing_radius+2)*pitch):
                raise RepairRejected('Repair rejected: unresolved large opening cannot establish an interior',
                                     {'status':'rejected','before':before,'closed_boundary_loops':closed_loops})
    pitch = float(np.ptp(vertices, axis=0).max()) / resolution
    if pitch <= 1e-8:
        raise ValueError('Repair input has no spatial extent')
    padding = closing_radius+4
    origin = vertices.min(0)-pitch*padding
    shape = np.ceil((vertices.max(0)-origin)/pitch).astype(int)+padding+1
    axes = [origin[i]+np.arange(shape[i],dtype=np.float32)*pitch for i in range(3)]
    scene = _scene(np.asarray(mesh.vertices,dtype=np.float32),np.asarray(mesh.faces,dtype=np.int64))
    yz = np.stack(np.meshgrid(axes[1],axes[2],indexing='ij'),axis=-1).reshape(-1,2)
    shell = np.empty(tuple(shape),dtype=bool)
    for i,x in enumerate(axes[0]):
        points = np.empty((len(yz),3),dtype=np.float32);points[:,0]=x;points[:,1:]=yz
        shell[i] = (_distance(scene,points)<=pitch*.75).reshape(shape[1:])
    sampled_at = time.perf_counter()
    if closing_radius:
        xyz = np.ogrid[tuple(slice(-closing_radius,closing_radius+1) for _ in range(3))]
        ball = sum(axis*axis for axis in xyz)<=closing_radius*closing_radius
        shell = ndimage.binary_closing(shell,structure=ball)
    solid = ndimage.binary_fill_holes(shell)
    interior_voxels=int((solid & ~shell).sum())
    if interior_voxels < max(1,int(solid.sum()*.05)):
        raise RepairRejected('Repair rejected: shell cannot establish a sufficient enclosed interior',
                             {'status':'rejected','before':before,'interior_voxels':interior_voxels})
    # Retain appendages even if an input surface represented them separately.
    labels,count = ndimage.label(solid)
    if count>1:
        sizes=np.bincount(labels.ravel());keep=sizes>=max(8,int(sizes[1:].max()*.0001));keep[0]=False
        solid=keep[labels]
    if not solid.any():
        raise RepairRejected('Repair rejected: no occupied surface',{'status':'rejected','before':before})
    output,triangles = extract_closed_surface(solid,pitch,origin)
    result = trimesh.Trimesh(output,triangles,process=True)
    trimesh.smoothing.filter_taubin(result,lamb=.45,nu=.5,iterations=5)
    result.fix_normals(multibody=True)
    if target_faces and len(result.faces)>target_faces:
        import open3d as o3d
        geometry=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(result.vertices),o3d.utility.Vector3iVector(result.faces.astype(np.int32)))
        geometry=geometry.simplify_quadric_decimation(int(target_faces),boundary_weight=20)
        result=trimesh.Trimesh(np.asarray(geometry.vertices),np.asarray(geometry.triangles),process=False)
        result.fix_normals(multibody=True)
    output=np.asarray(result.vertices,dtype=np.float32);triangles=np.asarray(result.faces,dtype=np.int64)
    try:
        report=validate_repair(vertices,faces,output,triangles)
    except RepairRejected as exc:
        exc.report.update(before=before,resolution=resolution,closing_radius=closing_radius,seconds=time.perf_counter()-started)
        raise
    report.update(before=before,resolution=resolution,closing_radius=closing_radius,pitch=pitch,closed_boundary_loops=closed_loops,
                  grid_shape=shape.tolist(),interior_voxels=interior_voxels,distance_query_seconds=sampled_at-started,
                  seconds=time.perf_counter()-started)
    return RepairResult(output,triangles,report)
