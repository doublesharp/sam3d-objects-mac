"""Bounded, exact CPU helpers shared by both repair profiles."""
import numpy as np
from scipy import ndimage


def spherical_closing(field, radius):
    radius = int(radius)
    field = np.asarray(field, dtype=bool)
    if radius < 4:
        axes = np.ogrid[tuple(slice(-radius, radius+1) for _ in range(3))]
        return ndimage.binary_closing(field, structure=sum(a*a for a in axes) <= radius*radius)
    if not field.any():
        return np.zeros_like(field)
    # Dilation is clipped to the original domain, exactly as binary_closing.
    dilated = ndimage.distance_transform_edt(~field) <= radius
    # False exterior is essential: erosion must remove the finite-grid border.
    padded = np.pad(dilated, 1, constant_values=False)
    return (ndimage.distance_transform_edt(padded) > radius)[1:-1, 1:-1, 1:-1]


def triangle_band(axes, vertices, faces, threshold):
    """Conservative union of expanded triangle AABBs on the actual query axes."""
    shape = tuple(len(a) for a in axes)
    delta = np.zeros(tuple(n+1 for n in shape), dtype=np.int32)
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.float32)
    for start in range(0, len(faces), 32768):
        triangles = vertices[faces[start:start+32768]].astype(np.float64)
        low = triangles.min(axis=1)-threshold
        high = triangles.max(axis=1)+threshold
        lower=[]; upper=[]
        for axis, coords in enumerate(axes):
            lower.append(np.clip(np.searchsorted(coords, low[:,axis], side='left')-1,0,shape[axis]))
            upper.append(np.clip(np.searchsorted(coords, high[:,axis], side='right')+1,0,shape[axis]))
        valid = np.logical_and.reduce([lo < hi for lo,hi in zip(lower,upper)])
        for corner in np.ndindex(2,2,2):
            indices=tuple((upper[i] if c else lower[i])[valid] for i,c in enumerate(corner))
            np.add.at(delta, indices, -1 if sum(corner)%2 else 1)
    for axis in range(3):
        np.cumsum(delta, axis=axis, dtype=np.int32, out=delta)
    return delta[:-1,:-1,:-1] > 0


def sample_shell(scene, pitch, origin, shape, distance, vertices=None, faces=None, progress=None):
    axes=[origin[i]+np.arange(shape[i],dtype=np.float32)*pitch for i in range(3)]
    # Open3D receives float32 points even if origin promoted an axis to float64.
    axes=[axis.astype(np.float32) for axis in axes]
    band = None if vertices is None or faces is None else triangle_band(axes,vertices,faces,pitch*.75)
    yz=np.stack(np.meshgrid(axes[1],axes[2],indexing='ij'),axis=-1).reshape(-1,2)
    shell=np.zeros(tuple(shape),dtype=bool)
    for i,x in enumerate(axes[0]):
        ids = np.arange(len(yz)) if band is None else np.flatnonzero(band[i].ravel())
        for start in range(0,len(ids),262144):
            selected=ids[start:start+262144]
            points=np.empty((len(selected),3),dtype=np.float32)
            points[:,0]=x;points[:,1:]=yz[selected]
            shell[i].ravel()[selected]=distance(scene,points)<=pitch*.75
        if progress and (i%24==0 or i+1==len(axes[0])):
            progress(f'Detail field {i+1}/{len(axes[0])}')
    return shell
