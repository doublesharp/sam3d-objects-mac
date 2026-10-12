import os
os.environ.setdefault('SPARSE_BACKEND', 'native')
os.environ.setdefault('LIDRA_SKIP_INIT', 'true')

from types import SimpleNamespace
import numpy as np
import pytest
import torch
from sam3d_objects.pipeline.inference_pipeline_pointmap import (
    InferencePipelinePointMap, camera_to_pytorch3d_camera,
)


def test_half_pointmap_uses_float32_camera_geometry():
    if not torch.backends.mps.is_available():
        pytest.skip('Requires MPS')
    device = torch.device('mps')
    points = torch.arange(12, device=device, dtype=torch.float16).reshape(2, 2, 3)
    intrinsics = torch.eye(3, device=device)
    stub = SimpleNamespace(
        device=device, dtype=torch.float16,
        image_to_float=lambda image: image.astype(np.float32) / 255,
        depth_model=lambda image: {'pointmaps': points, 'intrinsics': intrinsics},
        _clip_pointmap=lambda p, mask: p,
    )
    image = np.full((2, 2, 4), 255, dtype=np.uint8)
    result = InferencePipelinePointMap.compute_pointmap(stub, image)
    expected = points.float() @ camera_to_pytorch3d_camera(device).rotation[0]
    assert result['pointmap'].dtype == torch.float32
    assert result['pointmap'].device.type == 'mps'
    torch.testing.assert_close(result['pointmap'].permute(1, 2, 0), expected)
