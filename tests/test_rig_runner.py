"""Exercise the actual run.sh process boundaries with CPU exporter/worker stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from test_learned_rig import source_fixture
from test_automatic_rig import STUB


PROJECT=Path(__file__).resolve().parents[1]


def runner_fixture(tmp_path):
    root=tmp_path/'project';root.mkdir();shutil.copy(PROJECT/'run.sh',root/'run.sh')
    tools=root/'tools';tools.mkdir()
    for name in ('auto_rig.py','rig_validation.py','rig_glb.py'):shutil.copy(PROJECT/'tools'/name,tools/name)
    checkpoint=root/'checkpoints/hf/checkpoints';checkpoint.mkdir(parents=True)
    for name in ('pipeline.yaml','ss_generator.ckpt','slat_generator.ckpt','ss_decoder.ckpt','slat_decoder_gs.ckpt','slat_decoder_mesh.ckpt'):(checkpoint/name).write_bytes(b'stub')
    python=root/'.venv/bin/python3';python.parent.mkdir(parents=True);python.write_text('#!/bin/sh\nexec '+str(sys.executable)+' "$@"\n');python.chmod(0o755)
    commands=root/'commands';commands.mkdir()
    for name,body in [('sysctl','echo 4096'),('vm_stat',"echo 'Pages free: 10000000.'")]:
        command=commands/name;command.write_text('#!/bin/sh\n'+body+'\n');command.chmod(0o755)
    source_fixture(root/'fixture.glb')
    (root/'ply2glb.py').write_text("import sys,shutil\nfrom pathlib import Path\np=Path(sys.argv[1])\nif p.name=='bad':sys.exit(9)\nshutil.copy(Path(__file__).parent/'fixture.glb',p/'mesh.glb')\n")
    (root/'cli.py').write_text("import os\nfrom pathlib import Path\np=Path(__file__).parent\nPath(os.environ['SAM3D_MANIFEST']).write_text(''.join(str(p/n)+'\\t0\\t0\\n' for n in ('good','bad','last')))\n")
    for name in ('good','bad','last'):(root/name).mkdir()
    model=root/'model';model.mkdir();(model/'run_local.py').write_text(STUB)
    env={**os.environ,'PATH':str(commands)+':'+os.environ['PATH'],'SAM3D_RIG_MODEL_DIR':str(model),'SAM3D_RIG_PYTHON':str(sys.executable),'SAM3D_TEST_PROJECT':str(root),'PYTHONDONTWRITEBYTECODE':'1'}
    env.pop('STUB_ACTION',None)
    return root,env


def test_glb_entry_runs_rig_after_static_export(tmp_path):
    root,env=runner_fixture(tmp_path)
    completed=subprocess.run(['bash',str(root/'run.sh'),'glb',str(root/'good')],env=env,capture_output=True,text=True,timeout=10)
    assert completed.returncode==0,completed.stderr
    assert (root/'good/mesh.glb').is_file()
    assert (root/'good/mesh_rig_candidate.glb').is_file()
    assert json.loads((root/'good/rig_status.json').read_text())['status']=='candidate_generated'


def test_full_batch_preserves_export_failures_and_continues_rig(tmp_path):
    root,env=runner_fixture(tmp_path)
    completed=subprocess.run(['bash',str(root/'run.sh'),'full'],env=env,capture_output=True,text=True,timeout=10)
    assert completed.returncode==0,completed.stderr
    assert '2 GLB export(s) and 1 failure(s)' in completed.stdout
    assert not (root/'bad/rig_status.json').exists()
    for name in ('good','last'):
        assert (root/name/'mesh_rig_candidate.glb').is_file()
        assert json.loads((root/name/'rig_status.json').read_text())['status']=='candidate_generated'


def test_full_batch_rig_failure_keeps_successful_static_status(tmp_path):
    root,env=runner_fixture(tmp_path);env['STUB_ACTION']='exit'
    completed=subprocess.run(['bash',str(root/'run.sh'),'full'],env=env,capture_output=True,text=True,timeout=10)
    assert completed.returncode==0 and '2 GLB export(s) and 1 failure(s)' in completed.stdout
    for name in ('good','last'):
        assert (root/name/'mesh.glb').is_file() and not (root/name/'mesh_rig_candidate.glb').exists()
        assert json.loads((root/name/'rig_status.json').read_text())['status']=='failed'
