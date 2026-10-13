import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from tools.auto_rig import run_rig
from test_learned_rig import source_fixture


STUB = r'''
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ['SAM3D_TEST_PROJECT'])
from tools.rig_glb import source_geometry
p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');p.add_argument('--device');p.add_argument('--dtype');p.add_argument('--threads');p.add_argument('--beams');p.add_argument('--max-time');p.add_argument('--full-quality',action='store_true');p.add_argument('--gltf-to-blender',action='store_true');p.add_argument('--skeleton');a=p.parse_args()
out=Path(a.output);out.mkdir(exist_ok=True)
if os.getenv('STUB_ACTION')=='sleep':time.sleep(20)
if os.getenv('STUB_ACTION')=='exit':sys.exit(7)
points,faces=source_geometry(a.input)
joints=np.array([[0,0,0],[0,.5,0]]);parents=np.array([-1,0]);weights=np.tile([.4,.6],(len(points),1))
if a.skeleton:
 sk=json.loads(Path(a.skeleton).read_text());joints=np.asarray(sk['joints']);parents=np.asarray(sk['parents'])
 if os.getenv('STUB_ACTION')=='bad_override':joints[1,1]+=.01
result={'status':'inference_passed','tokens':12,'tokens_eos':100,'final_token':100,'skeleton_eos_present':True,'global_eos_present':True,'skin_tokens_valid':True,'skin_token_count':8,'counted_skeleton_bones':2,'skeleton_eos':3,'skin_vocab_start':4,'tokens_per_skin':4}
if os.getenv('STUB_ACTION')=='failed_metadata':result['status']='failed'
if os.getenv('STUB_ACTION')=='incomplete':result['skin_token_count']=7
result['env']={key:os.environ.get(key) for key in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_MPS_LOW_WATERMARK_RATIO','OMP_NUM_THREADS')}
result['args']=vars(a)
(out/'result.json').write_text(json.dumps(result))
np.savez(out/'rig.npz',vertices=points,faces=faces,joints=joints,parents=parents,weights=weights,tokens=np.array([1,2,3,4,5,6,7,8,9,10,11,100]))
'''


@pytest.fixture
def configured(tmp_path,monkeypatch):
    folder=tmp_path/'export';folder.mkdir();source_fixture(folder/'mesh.glb')
    model=tmp_path/'model';model.mkdir();(model/'run_local.py').write_text(STUB)
    monkeypatch.setenv('SAM3D_TEST_PROJECT',str(Path(__file__).resolve().parents[1]))
    return folder,model


def test_cpu_worker_produces_separate_unqualified_candidate(configured,monkeypatch):
    folder,model=configured;source=folder/'mesh.glb';before=source.read_bytes()
    monkeypatch.setenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO','0.0')
    status=run_rig(folder,model_dir=model,python=sys.executable,device='cpu')
    assert status['status']=='candidate_generated' and status['structurally_valid'] and status['anatomically_qualified'] is False
    assert (folder/'mesh_rig_candidate.glb').is_file() and source.read_bytes()==before
    result=json.loads((folder/'rig_candidate_data'/'result.json').read_text())
    assert result['env']=={'PYTORCH_MPS_HIGH_WATERMARK_RATIO':'0.9','PYTORCH_MPS_LOW_WATERMARK_RATIO':'0.8','OMP_NUM_THREADS':'2'}
    assert result['args']['threads']=='2' and result['args']['beams']=='10'
    assert result['args']['full_quality'] and result['args']['gltf_to_blender']
    assert os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']=='0.0'
    assert json.loads((folder/'rig_status.json').read_text())['status']=='candidate_generated'


def test_off_and_missing_runtime_preserve_static(configured):
    folder,model=configured;before=(folder/'mesh.glb').read_bytes()
    assert run_rig(folder,mode='off')['status']=='off'
    status=run_rig(folder,model_dir=model,python=folder/'missing')
    assert status['status']=='skipped_missing_runtime' and 'SAM3D_RIG_PYTHON' in status['error']
    assert not (folder/'mesh_rig_candidate.glb').exists() and (folder/'mesh.glb').read_bytes()==before


@pytest.mark.parametrize('action',['exit','failed_metadata','incomplete','bad_override'])
def test_worker_failures_cannot_promote_candidate(configured,monkeypatch,action):
    folder,model=configured;monkeypatch.setenv('STUB_ACTION',action)
    override=folder/'skeleton.json';override.write_text(json.dumps({'joints':[[0,0,0],[0,.5,0]],'parents':[-1,0]}))
    status=run_rig(folder,model_dir=model,python=sys.executable,device='cpu',skeleton=override)
    assert status['status']=='failed' and status['anatomically_qualified'] is False
    assert not (folder/'mesh_rig_candidate.glb').exists() and (folder/'mesh.glb').is_file()


def test_timeout_terminates_worker_and_keeps_static(configured,monkeypatch):
    folder,model=configured;monkeypatch.setenv('STUB_ACTION','sleep');start=time.monotonic()
    status=run_rig(folder,model_dir=model,python=sys.executable,device='cpu',timeout=.2,max_time=.1)
    assert status['status']=='failed' and 'timeout' in status['error'].lower()
    assert time.monotonic()-start<3 and not (folder/'mesh_rig_candidate.glb').exists()


def test_exact_override_passed_to_worker_and_retained(configured):
    folder,model=configured;override=folder/'skeleton.json';override.write_text(json.dumps({'joints':[[.012,0,0],[.011,.4321,0]],'parents':[-1,0]}))
    status=run_rig(folder,model_dir=model,python=sys.executable,device='cpu',skeleton=override)
    assert status['status']=='candidate_generated' and status['exact_override_retained']
    with np.load(folder/'rig_candidate_data'/'rig.npz') as rig:
        np.testing.assert_array_equal(rig['joints'],json.loads(override.read_text())['joints'])


def test_existing_candidate_and_symlink_source_are_rejected(configured):
    folder,model=configured;candidate=folder/'mesh_rig_candidate.glb';candidate.write_bytes(b'keep')
    assert run_rig(folder,model_dir=model,python=sys.executable)['status']=='failed'
    assert candidate.read_bytes()==b'keep'
    candidate.unlink();source=folder/'mesh.glb';source.rename(folder/'real.glb');source.symlink_to(folder/'real.glb')
    assert run_rig(folder,model_dir=model,python=sys.executable)['status']=='failed'


def test_invalid_override_rejected_before_worker(configured):
    folder,model=configured;override=folder/'skeleton.json';override.write_text(json.dumps({'joints':[[0,0,0],[0,.5,0]],'parents':[-1,1]}))
    assert run_rig(folder,model_dir=model,python=sys.executable,skeleton=override)['status']=='failed'
    assert not (folder/'rig_candidate_data').exists()


def test_configurable_beams_and_generation_bound(configured):
    folder,model=configured
    status=run_rig(folder,model_dir=model,python=sys.executable,device='cpu',beams=3,timeout=10,max_time=5)
    assert status['status']=='candidate_generated'
    result=json.loads((folder/'rig_candidate_data'/'result.json').read_text())
    assert result['args']['beams']=='3' and float(result['args']['max_time'])==5


@pytest.mark.parametrize('beams',[0,-1,1.5])
def test_invalid_beams_rejected_before_worker(configured,beams):
    folder,model=configured
    assert run_rig(folder,model_dir=model,python=sys.executable,beams=beams)['status']=='failed'
    assert not (folder/'rig_candidate_data').exists()


def test_concurrent_worker_lock_rejects_second_process(configured):
    import fcntl
    folder,model=configured
    with (model.parent/'rig-worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=run_rig(folder,model_dir=model,python=sys.executable)
    assert status['status']=='failed' and 'another learned rig worker' in status['error']
    assert not (folder/'rig_candidate_data').exists()


def test_status_symlink_does_not_launch_worker(configured):
    folder,model=configured;target=folder/'important.json';target.write_text('keep');(folder/'rig_status.json').symlink_to(target)
    status=run_rig(folder,model_dir=model,python=sys.executable)
    assert status['status']=='failed' and target.read_text()=='keep'
    assert not (folder/'rig_candidate_data').exists()


def test_refused_symlink_folder_writes_no_status(configured):
    folder,model=configured;alias=folder.parent/'alias';alias.symlink_to(folder,target_is_directory=True)
    status=run_rig(alias,model_dir=model,python=sys.executable)
    assert status['status']=='failed' and not (folder/'rig_status.json').exists()


@pytest.mark.parametrize('joints',[[[0,0,0],[0,0,0]],[[0,0,0],[0,10,0]]])
def test_invalid_override_bones_rejected_before_worker(configured,joints):
    folder,model=configured;override=folder/'skeleton.json';override.write_text(json.dumps({'joints':joints,'parents':[-1,0]}))
    assert run_rig(folder,model_dir=model,python=sys.executable,skeleton=override)['status']=='failed'
    assert not (folder/'rig_candidate_data').exists()


@pytest.mark.parametrize('settings',[{'timeout':241},{'max_time':121},{'timeout':10,'max_time':20},{'timeout':float('inf')},{'max_time':0}])
def test_resource_time_ceilings_rejected_before_worker(configured,settings):
    folder,model=configured
    status=run_rig(folder,model_dir=model,python=sys.executable,**settings)
    assert status['status']=='failed'
    assert not (folder/'rig_candidate_data').exists()
