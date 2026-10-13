"""Optional learned rig stage, launched after the static exporter has exited."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

# Support both `python tools/auto_rig.py` and importing the tools module.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _status_file(folder, status):
    if not folder.is_dir() or any(path.is_symlink() for path in (folder,*folder.parents)):
        raise ValueError('refusing status write through missing or symlink output folder')
    path = folder/'rig_status.json'
    if path.is_symlink():
        raise ValueError('refusing symlink rig_status.json')
    fd, temporary = tempfile.mkstemp(prefix='.rig-status-', dir=folder)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(status, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _worker(command, cwd, environment, log_path, timeout):
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the worker process group, including any children, then reap it.
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
            raise ValueError(f'rig worker timeout after {timeout:g} seconds')
    if returncode:
        raise ValueError(f'rig worker exited with status {returncode}; see {log_path.name}')


def run_rig(folder, *, mode='auto', model_dir=None, python=None, skeleton=None,
            device='mps', timeout=240, max_time=120, beams=10):
    """Return/write rig_status.json; ordinary rig failure preserves static export.

    All outputs are separate. A candidate passes CPU structure and bind checks,
    but remains anatomically unqualified. Existing candidates require a new
    output folder, so unattended reruns cannot replace a reviewed asset.
    """
    started=time.monotonic();folder=Path(folder).absolute()
    project=Path(__file__).resolve().parents[1]
    model_dir=Path(model_dir or os.environ.get('SAM3D_RIG_MODEL_DIR',project/'.cache/rigging/SkinTokens')).absolute()
    python=Path(python or os.environ.get('SAM3D_RIG_PYTHON',project/'.cache/rigging/runtime/bin/python')).absolute()
    skeleton=skeleton or os.environ.get('SAM3D_RIG_SKELETON')
    source=folder/'mesh.glb';candidate=folder/'mesh_rig_candidate.glb';data=folder/'rig_candidate_data'
    status={'schema_version':1, 'mode':mode, 'source':str(source), 'candidate':None,
            'status':'failed', 'structurally_valid':False, 'anatomically_qualified':False,
            'anatomical_review':'pending', 'experimental':True}
    lock_fd=None
    try:
        if not folder.is_dir() or any(path.is_symlink() for path in (folder,*folder.parents)): raise ValueError('output folder must be an existing nonsymlink directory')
        if (folder/'rig_status.json').is_symlink():raise ValueError('refusing symlink rig_status.json')
        if mode not in ('auto','off'): raise ValueError('rig mode must be auto or off')
        if mode=='off':
            status['status']='off'
        else:
            if source.is_symlink() or not source.is_file(): raise ValueError('static mesh.glb must be a regular nonsymlink file')
            if candidate.exists() or candidate.is_symlink() or data.exists() or data.is_symlink():
                raise ValueError('rig candidate or diagnostics already exist; use a new output folder')
            if not (model_dir/'run_local.py').is_file() or not python.is_file() or not os.access(python,os.X_OK):
                status.update(status='skipped_missing_runtime', error='Optional SkinTokens runtime is unavailable. Configure SAM3D_RIG_MODEL_DIR with run_local.py and SAM3D_RIG_PYTHON with the prepared runtime Python.')
            else:
                if not all(math.isfinite(value) and value>0 for value in (timeout,max_time)):
                    raise ValueError('rig timeout and generation max-time must be finite and positive')
                if max_time>120 or timeout>240 or timeout<max_time:
                    raise ValueError('rig generation max-time must be <=120 seconds; timeout must be <=240 seconds and >=max-time')
                if isinstance(beams,bool) or not isinstance(beams,int) or beams<1:
                    raise ValueError('rig beams must be a positive integer')
                if device not in ('cpu','mps'): raise ValueError('rig device must be cpu or mps')
                import numpy as np
                from tools.rig_glb import source_geometry, export_candidate
                from tools.rig_validation import validate_skeleton, validate_rig, validate_override, validate_tokens
                points,faces=source_geometry(source)
                supplied=None
                if skeleton:
                    skeleton=Path(skeleton).absolute()
                    supplied=json.loads(skeleton.read_text())
                    joints,parents=validate_skeleton(supplied['joints'],supplied['parents'])
                    check_weights=np.zeros((len(points),len(joints)))
                    check_weights[:,int(np.flatnonzero(parents==-1)[0])]=1
                    validate_rig(dict(vertices=points,faces=faces,joints=joints,parents=parents,weights=check_weights),points,faces)
                # One worker per configured model installation across batch callers.
                lock_path=model_dir.parent/'rig-worker.lock'
                lock_fd=os.open(lock_path,os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
                try:fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:raise ValueError('another learned rig worker is active for this model installation')
                source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
                data.mkdir()
                environment=os.environ.copy()
                environment.update(PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.9',
                                   PYTORCH_MPS_LOW_WATERMARK_RATIO='0.8',
                                   PYTORCH_ENABLE_MPS_FALLBACK='1',OMP_NUM_THREADS='2',
                                   MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                                   VECLIB_MAXIMUM_THREADS='2')
                command=[str(python),str(model_dir/'run_local.py'),'--input',str(source),
                         '--output',str(data),'--device',device,'--dtype','float32',
                         '--threads','2','--beams',str(beams),'--full-quality','--gltf-to-blender',
                         '--max-time',str(max_time)]
                if skeleton:command.extend(['--skeleton',str(skeleton)])
                status.update(worker_command=command, worker_timeout_seconds=timeout,
                              worker_memory_fraction=.45, diagnostics=str(data))
                _worker(command,model_dir,environment,data/'worker.log',timeout)
                if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:
                    raise ValueError('static source changed while rig worker was running')
                result=json.loads((data/'result.json').read_text())
                if result.get('status')!='inference_passed':
                    raise ValueError('model failed: '+str(result.get('error','missing inference_passed result status')))
                with np.load(data/'rig.npz',allow_pickle=False) as payload:
                    rig={key:payload[key] for key in ('vertices','faces','joints','parents','weights','tokens')}
                validate_tokens(rig['tokens'],result,len(rig['joints']))
                validate_rig(rig,points,faces)
                if supplied:validate_override(rig,supplied)
                metrics=export_candidate(source,candidate,rig)
                status.update(status='candidate_generated',candidate=str(candidate),structurally_valid=True,
                              exact_override_retained=bool(supplied),validation=metrics,
                              source_sha256=source_hash,
                              anatomical_review='pending: inspect joint placement and deformation before use')
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
    finally:
        if lock_fd is not None:os.close(lock_fd)
    status['elapsed_seconds']=time.monotonic()-started
    try:_status_file(folder,status)
    except Exception as error:status['status_write_error']=f'{type(error).__name__}: {error}'
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_folder')
    parser.add_argument('--mode',choices=('auto','off'),default=os.environ.get('SAM3D_RIG_MODE','auto'))
    parser.add_argument('--skeleton',default=os.environ.get('SAM3D_RIG_SKELETON'))
    parser.add_argument('--device',choices=('mps','cpu'),default='mps')
    parser.add_argument('--timeout',type=float,default=os.environ.get('SAM3D_RIG_TIMEOUT_SECONDS','240'))
    parser.add_argument('--max-time',type=float,default=os.environ.get('SAM3D_RIG_MAX_TIME_SECONDS','120'))
    parser.add_argument('--beams',type=int,default=os.environ.get('SAM3D_RIG_BEAMS','10'))
    args=parser.parse_args()
    status=run_rig(args.output_folder,mode=args.mode,skeleton=args.skeleton,device=args.device,
                   timeout=args.timeout,max_time=args.max_time,beams=args.beams)
    print(json.dumps(status,indent=2))
    # The completed static export is still successful; callers inspect status.
    return 0


if __name__=='__main__':sys.exit(main())
