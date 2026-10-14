import numpy as np
import pickle
import pytest
import torch

from sam3d_objects.model.io import load_model_from_checkpoint


@pytest.mark.parametrize('mapped', [True, False])
@pytest.mark.parametrize('modern', [True, False])
def test_checkpoint_load_preserves_weights_and_inference(tmp_path, mapped, modern):
    torch.manual_seed(5)
    original = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / 'model.ckpt'
    torch.save({'state_dict': original.state_dict(), 'array': np.arange(3)}, checkpoint,
               _use_new_zipfile_serialization=modern)
    actual = torch.nn.Linear(3, 2)
    load_model_from_checkpoint(actual, str(checkpoint), device='cpu', mmap=mapped,
                               freeze=True, eval=True)
    inputs = torch.tensor([[1., 2., 3.], [-2., 1., 4.]])
    torch.testing.assert_close(actual(inputs), original(inputs), rtol=0, atol=0)
    assert not actual.training
    assert all(not parameter.requires_grad for parameter in actual.parameters())


def test_mapped_checkpoint_supports_prefix_filter_and_key_remapping(tmp_path):
    from sam3d_objects.model.io import filter_and_remove_prefix_state_dict_fn
    original = torch.nn.Linear(3, 2)
    state = {'prefix.'+name: value for name, value in original.state_dict().items()}
    state['unneeded.large_tensor'] = torch.arange(12)
    checkpoint = tmp_path / 'prefix.ckpt'
    torch.save({'state_dict': state}, checkpoint)
    actual = torch.nn.Linear(3, 2)
    load_model_from_checkpoint(actual, str(checkpoint), device='cpu', mmap=True,
                               state_dict_fn=filter_and_remove_prefix_state_dict_fn('prefix.'))
    for name, value in original.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)


def test_corrupt_checkpoint_is_not_silently_retried(tmp_path):
    checkpoint = tmp_path / 'bad.ckpt'
    checkpoint.write_bytes(b'broken checkpoint')
    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        load_model_from_checkpoint(torch.nn.Linear(3, 2), str(checkpoint), mmap=True)
