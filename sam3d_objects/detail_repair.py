"""Rebuild a body core while preserving exterior detail on a finer grid."""
import time
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure
from .surface_repair import (_arrays, _mesh, _scene, _distance, topology,
                             geometry_report, RepairResult, RepairRejected)


def _ball(radius):
    radius=int(radius)
    axes=np.ogrid[tuple(slice(-radius,radius+1) for _ in range(3))]
    return sum(axis*axis for axis in axes)<=radius*radius


def _largest_volume(field):
    labels,count=ndimage.label(field)
    if not count:return np.zeros_like(field)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    return labels==int(sizes.argmax())


def body_core_from_shell(shell, closing_radius=6, opening_radius=3, erosion=1):
    shell=np.asarray(shell,dtype=bool)
    if shell.ndim!=3 or not shell.any():raise ValueError('Cannot establish a body core')
    # A one-cell shell can have diagonal leaks at rounded boundary openings.
    # Thicken only the coarse core seed, then undo that offset by erosion.
    seeded=ndimage.binary_dilation(shell,structure=_ball(1))
    closed=ndimage.binary_closing(seeded,structure=_ball(closing_radius))
    filled=ndimage.binary_fill_holes(closed)
    opened=ndimage.binary_opening(filled,structure=_ball(opening_radius))
    opened=ndimage.binary_erosion(opened,iterations=int(erosion)+1)
    core=_largest_volume(opened)
    if not core.any():raise ValueError('Cannot establish a body core')
    return core


def compose_body_and_detail(detail_shell, core):
    detail_shell=np.asarray(detail_shell,dtype=bool);core=np.asarray(core,dtype=bool)
    if detail_shell.shape!=core.shape or not core.any():raise ValueError('Invalid body core or grid shape')
    # Closing is deliberately confined to the body core, preserving spike spacing.
    return ndimage.binary_fill_holes(detail_shell | core)


def _grid(vertices,resolution,padding=10):
    pitch=float(np.ptp(vertices,axis=0).max())/int(resolution)
    origin=vertices.min(0)-pitch*padding
    shape=np.ceil((vertices.max(0)-origin)/pitch).astype(int)+padding+1
    return pitch,origin,shape


def _sample_shell(scene,pitch,origin,shape,progress=None):
    axes=[origin[i]+np.arange(shape[i],dtype=np.float32)*pitch for i in range(3)]
    yz=np.stack(np.meshgrid(axes[1],axes[2],indexing='ij'),axis=-1).reshape(-1,2)
    shell=np.empty(tuple(shape),dtype=bool)
    for i,x in enumerate(axes[0]):
        points=np.empty((len(yz),3),dtype=np.float32);points[:,0]=x;points[:,1:]=yz
        shell[i]=(_distance(scene,points)<=pitch*.75).reshape(shape[1:])
        if progress and (i%24==0 or i+1==len(axes[0])):progress(f'Detail field {i+1}/{len(axes[0])}')
    return shell


def _sample_grid(field,points,pitch,origin,order=0):
    coordinates=((np.asarray(points)-origin)/pitch).T
    return ndimage.map_coordinates(field.astype(np.float32,copy=False),coordinates,
                                    order=order,mode='constant',cval=0.)


def _upsample_core(core,coarse_pitch,coarse_origin,pitch,origin,shape):
    ys=origin[1]+np.arange(shape[1],dtype=np.float32)*pitch
    zs=origin[2]+np.arange(shape[2],dtype=np.float32)*pitch
    yz=np.stack(np.meshgrid(ys,zs,indexing='ij'),axis=-1).reshape(-1,2)
    signed=(ndimage.distance_transform_edt(core)-ndimage.distance_transform_edt(~core))*coarse_pitch
    result=np.empty(tuple(shape),dtype=bool)
    for i in range(shape[0]):
        points=np.empty((len(yz),3),dtype=np.float32);points[:,0]=origin[0]+i*pitch;points[:,1:]=yz
        result[i]=(_sample_grid(signed,points,coarse_pitch,coarse_origin,order=1)>0).reshape(shape[1:])
    return result



def _source_visibility(vertices, faces, points, pitch):
    """Distributed source-only visibility; the candidate never defines this mask."""
    import open3d as o3d
    points=np.asarray(points,dtype=np.float32)
    scene=_scene(vertices,faces)
    travel=max(float(np.linalg.norm(np.ptp(vertices,axis=0)))*3, pitch*10)
    visible=np.zeros(len(points),dtype=bool)
    # The 26 axis/diagonal directions are a bounded exterior proxy. Always retain
    # convex-hull extrema separately, including sharp tips hit ambiguously by rays.
    for direction in np.ndindex(3,3,3):
        direction=np.asarray(direction,dtype=np.float32)-1
        if not direction.any():continue
        direction/=np.linalg.norm(direction)
        rays=np.empty((len(points),6),dtype=np.float32)
        rays[:,:3]=points-direction*travel;rays[:,3:]=direction
        hits=scene.cast_rays(o3d.core.Tensor(rays),nthreads=8)['t_hit'].numpy()
        visible |= np.abs(hits-travel)<=pitch*2
    from scipy.spatial import ConvexHull
    try:
        hull=ConvexHull(points)
        visible[hull.vertices]=True
    except Exception:
        # Degenerate point sets must not silently lose protection.
        visible[:]=True
    return visible


def _simplify_detail(mesh, target_faces):
    import open3d as o3d
    if target_faces and len(mesh.faces)>target_faces:
        geometry=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                          o3d.utility.Vector3iVector(mesh.faces))
        reduced=geometry.simplify_quadric_decimation(int(target_faces),boundary_weight=20.)
        mesh=trimesh.Trimesh(np.asarray(reduced.vertices),np.asarray(reduced.triangles),process=False)
        mesh.fix_normals(multibody=True)
    # Decimation can cross close detail sheets. Clean only the bounded candidate,
    # never the original million-face reconstruction, and check the result below.
    check=geometry_report(mesh.vertices,mesh.faces)
    if not check['strict_watertight'] or not check['orientable']:
        import pymeshfix
        fix=pymeshfix.MeshFix(mesh.vertices,mesh.faces)
        fix.repair(joincomp=False,remove_smallest_components=False)
        mesh=trimesh.Trimesh(np.asarray(fix.points),np.asarray(fix.faces),process=False)
        mesh.fix_normals(multibody=True)
    return mesh

def _validate_detail(source_vertices,source_faces,vertices,faces,core,coarse_pitch,
                     coarse_origin,pitch):
    core=np.asarray(core,dtype=bool)
    core_euler=int(measure.euler_number(core,connectivity=3))
    if not core.any() or core_euler!=1:
        raise RepairRejected('Protected detail repair cannot establish a solid body core',
                             {'status':'rejected','core_euler':core_euler})
    after=topology(vertices,faces);geometry=geometry_report(vertices,faces)
    report={'status':'rejected','after':after,'geometry':geometry,'core_euler':core_euler}
    if not geometry['strict_watertight'] or not geometry['orientable'] or not after['winding_consistent']:
        raise RepairRejected('Protected detail repair left intersections or an invalid surface',report)
    scale=float(np.linalg.norm(np.ptp(source_vertices,axis=0)))
    retention=np.ptp(vertices,axis=0)/np.maximum(np.ptp(source_vertices,axis=0),scale*1e-5)
    outside_distance=ndimage.distance_transform_edt(~core)*coarse_pitch
    distances=_sample_grid(outside_distance,source_vertices,coarse_pitch,coarse_origin,order=1)
    coordinates=(source_vertices-coarse_origin)/coarse_pitch
    outside_grid=((coordinates<0)|(coordinates>np.asarray(core.shape)-1)).any(axis=1)
    detail=(distances>coarse_pitch*2)|outside_grid
    visible=_source_visibility(source_vertices,source_faces,source_vertices,pitch)
    protected=detail & visible
    report.update(source_visible_vertices=int(visible.sum()),hidden_detail_vertices=int((detail & ~visible).sum()),
                  visibility_directions=26,visibility_limit='Distributed proxy; not exhaustive visibility proof')
    candidate_scene=_scene(vertices,faces)
    if not protected.any():raise RepairRejected('No protected exterior detail could be established',report)
    protected_error=_distance(candidate_scene,source_vertices[protected])
    protected_p95=float(np.quantile(protected_error,.95));protected_max=float(protected_error.max())
    candidate_samples=trimesh.sample.sample_surface(_mesh(vertices,faces),8192,seed=19)[0].astype(np.float32)
    candidate_error=_distance(_scene(source_vertices,source_faces),candidate_samples)/scale
    # Body cavity walls inside the core are intentionally replaced. Exterior detail
    # must remain close, while the new visible surface must fit the source envelope.
    report.update(protected_vertices=int(protected.sum()),protected_error_p95=protected_p95,
                  protected_error_max=protected_max,protected_tolerance=pitch*3,
                  candidate_surface_error_p95=float(np.quantile(candidate_error,.95)),
                  bounds_retention_min=float(retention.min()),bounds_expansion_max=float(retention.max()),scale=scale)
    if retention.min()<.97 or retention.max()>1.12:
        raise RepairRejected('Protected detail repair changed the silhouette bounds',report)
    if protected_p95>pitch*3 or protected_max>max(pitch*8,scale*.025):
        raise RepairRejected('Protected detail repair lost exterior spike detail',report)
    if report['candidate_surface_error_p95']>.035:
        raise RepairRejected('Protected detail repair changed the visible surface too far',report)
    # Test well-inside core samples against the final surface, after simplification.
    inner=ndimage.binary_erosion(core,iterations=2)
    ids=np.argwhere(inner)
    if not len(ids):raise RepairRejected('Body core has no reliable interior',report)
    ids=ids[np.linspace(0,len(ids)-1,min(2048,len(ids)),dtype=int)]
    points=(coarse_origin+ids*coarse_pitch).astype(np.float32)
    import open3d as o3d
    inside=candidate_scene.compute_occupancy(o3d.core.Tensor(points),nsamples=3).numpy()
    report['core_containment_fraction']=float((inside>.5).mean())
    if report['core_containment_fraction']<.995:
        raise RepairRejected('Protected detail repair did not seal the body core',report)
    report['status']='repaired'
    return report



def repair_with_protected_detail(vertices,faces,resolution=256,core_resolution=96,
                                 closing_radius=8,opening_radius=8,erosion=3,
                                 target_faces=160000,progress=None,cache_path=None):
    """Explicit body/detail profile; rejects candidates that lose visible features.

    Distributed visibility excludes old internal cavity walls from detail checks.
    It does not infer anatomy or qualify every possible object category.
    """
    started=time.perf_counter();vertices,faces=_arrays(vertices,faces)
    if not 64<=int(resolution)<=320 or not 32<=int(core_resolution)<=128:
        raise ValueError('Fine resolution must be 64..320 and core resolution 32..128')
    if not 1<=int(closing_radius)<=12 or not 0<=int(opening_radius)<=12 or not 0<=int(erosion)<=6:
        raise ValueError('Invalid body core radius or erosion')
    if target_faces is not None and int(target_faces)<4:
        raise ValueError('Face budget must be at least four triangles')
    if np.ptp(vertices,axis=0).max()<=1e-8:
        raise ValueError('Repair input has no spatial extent')
    before=topology(vertices,faces);scene=_scene(vertices,faces)
    coarse_pitch,coarse_origin,coarse_shape=_grid(vertices,core_resolution,closing_radius+5)
    shell=_sample_shell(scene,coarse_pitch,coarse_origin,coarse_shape)
    core=body_core_from_shell(shell,closing_radius,opening_radius,erosion)
    core_seconds=time.perf_counter()-started
    pitch,origin,shape=_grid(vertices,resolution)
    fine_shell=_sample_shell(scene,pitch,origin,shape,progress)
    upsampled=_upsample_core(core,coarse_pitch,coarse_origin,pitch,origin,shape)
    solid=_largest_volume(compose_body_and_detail(fine_shell,upsampled))
    if cache_path:
        np.savez_compressed(cache_path,solid=solid,core=core,fine_shell=fine_shell,pitch=pitch,origin=origin,
                            coarse_pitch=coarse_pitch,coarse_origin=coarse_origin)
    field_seconds=time.perf_counter()-started
    field=ndimage.gaussian_filter(solid.astype(np.float32),sigma=.65)
    v,f,_,_=measure.marching_cubes(field,level=.501,spacing=(pitch,)*3,allow_degenerate=False)
    v+=origin
    mesh=trimesh.Trimesh(v,f,process=True);mesh.fix_normals(multibody=True)
    mesh=_simplify_detail(mesh,target_faces)
    if target_faces and len(mesh.faces)>target_faces:
        raise RepairRejected('Protected detail repair exceeds the face budget',
                             {'status':'rejected','triangles':len(mesh.faces),'target_faces':int(target_faces)})
    v=np.asarray(mesh.vertices,dtype=np.float32);f=np.asarray(mesh.faces,dtype=np.int64)
    report=_validate_detail(vertices,faces,v,f,core,coarse_pitch,coarse_origin,pitch)
    report.update(before=before,profile='protected-detail',resolution=int(resolution),
                  core_resolution=int(core_resolution),closing_radius=int(closing_radius),
                  opening_radius=int(opening_radius),erosion=int(erosion),pitch=pitch,
                  smoothing_sigma=.65,target_faces=target_faces,
                  core_seconds=core_seconds,field_seconds=field_seconds,
                  core_voxels=int(core.sum()),seconds=time.perf_counter()-started)
    return RepairResult(v,f,report)
