import os
from pathlib import Path
import subprocess
import sys


def test_headless_inference_loads_images_without_notebook_ui_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = r"""
import importlib.abc
import sys
from pathlib import Path
class RejectUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gradio', 'seaborn'}:
            raise AssertionError('Optional UI imported: ' + fullname)
sys.meta_path.insert(0, RejectUI())
root, image_path = map(Path, sys.argv[1:])
sys.path[:0] = [str(root), str(root / 'notebook')]
from inference import Inference, load_image, interactive_visualizer
from PIL import Image
Image.new('RGB', (2, 3), (4, 5, 6)).save(image_path)
image = load_image(image_path)
assert image.shape == (3, 2, 3)
assert image[0, 0].tolist() == [4, 5, 6]
assert callable(Inference)
try:
    interactive_visualizer('unused.ply')
except AssertionError as error:
    assert 'gradio' in str(error)
else:
    raise AssertionError('Visualizer should request its optional dependency')
"""
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'SPARSE_BACKEND': 'native', 'ATTN_BACKEND': 'sdpa',
           'LIDRA_SKIP_INIT': 'true', 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-c', code, str(root), str(tmp_path / 'image.png')],
                            cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
