import json, os, sys
from pathlib import Path
import pytest

def test_scheduler_real_children_overlap_and_bound(tmp_path):
    import batch_scheduler as scheduler
    events = tmp_path / 'events'
    code = "import sys,time,json; p,kind,i=sys.argv[1:]; f=open(p,'a'); f.write(json.dumps([kind,i,'start',time.monotonic()])+'\\n'); f.flush(); time.sleep(.12 if kind=='g' else .28); f.write(json.dumps([kind,i,'end',time.monotonic()])+'\\n'); f.close()"
    jobs = [{'name': str(i), 'generate': [sys.executable,'-c',code,str(events),'g',str(i)], 'export': [sys.executable,'-c',code,str(events),'e',str(i)]} for i in range(4)]
    records = scheduler.run_batch(jobs, overlap=True)
    assert all(r['generation'] == 0 and r['export'] == 0 for r in records)
    active = {'g':0,'e':0}; overlapped=False
    for kind,i,event,t in sorted((json.loads(line) for line in events.read_text().splitlines()),key=lambda r:r[3]):
        active[kind] += 1 if event=='start' else -1
        assert 0 <= active[kind] <= 1
        overlapped |= active['g']==active['e']==1
    assert overlapped
    data = [json.loads(line) for line in events.read_text().splitlines()]
    times = {(k,i,e):t for k,i,e,t in data}
    assert times['g','3','start'] >= times['e','1','end']

@pytest.mark.parametrize('overlap',[False,True])
def test_failures_continue_and_preserve_environment(tmp_path,overlap):
    import batch_scheduler as scheduler
    result = tmp_path/'✓ space.json'
    jobs=[{'name':'failed generation','generate':[sys.executable,'-c','raise SystemExit(7)'],'export':[sys.executable,'-c','raise SystemExit(99)']}, {'name':'export fails','generate':[sys.executable,'-c','pass'],'export':[sys.executable,'-c','raise SystemExit(8)']}, {'name':'good','generate':[sys.executable,'-c','pass'],'export':[sys.executable,'-c',"import os,json,sys; json.dump(dict(os.environ),open(sys.argv[1],'w'))",str(result)]}]
    records=scheduler.run_batch(jobs,overlap=overlap,env={**os.environ,'SAM3D_SOURCE_ONLY':'1','SAM3D_REPAIR_PROFILE':'final','OMP_NUM_THREADS':'2','SAM3D_EXPORT_THREADS':'2'})
    assert [(r['generation'],r['export']) for r in records]==[(7,None),(0,8),(0,0)]
    env=json.loads(result.read_text()); assert env['SAM3D_SOURCE_ONLY']=='1'; assert env.get('SAM3D_EXPORT_DEVICE') == ('cpu' if overlap else os.environ.get('SAM3D_EXPORT_DEVICE')); assert env['OMP_NUM_THREADS']=='2'

@pytest.mark.parametrize('env,total,available,want', [({},64,32,True),({},63,40,False),({},64,31,False),({'SAM3D_PIPELINED_BATCH':'0'},128,80,False),({'SAM3D_TEXTURE_MODE':'opt'},128,80,False),({'SAM3D_PIPELINED_BATCH':'garbage'},128,80,False)])
def test_eligibility(env,total,available,want):
    import batch_scheduler as scheduler
    assert scheduler.overlap_eligible(env,memory=(total*2**30,available*2**30)) is want

def test_low_memory_drains_export_before_next_generation(tmp_path):
    import batch_scheduler as scheduler
    events=tmp_path/'events'
    code="import sys,time; f=open(sys.argv[1],'a'); f.write(sys.argv[2]+' start\\n'); f.flush(); time.sleep(.06); f.write(sys.argv[2]+' end\\n'); f.close()"
    jobs=[{'name':str(i),'generate':[sys.executable,'-c',code,str(events),f'g{i}'],'export':[sys.executable,'-c',code,str(events),f'e{i}']} for i in range(2)]
    scheduler.run_batch(jobs,overlap=True,memory_check=lambda:False)
    assert events.read_text().splitlines()==['g0 start','g0 end','e0 start','e0 end','g1 start','g1 end','e1 start','e1 end']


def test_export_device_setting_is_validated(monkeypatch):
    import ply2glb
    monkeypatch.setenv('SAM3D_EXPORT_DEVICE','cpu')
    assert ply2glb.choose_export_device()=='cpu'
    monkeypatch.setenv('SAM3D_EXPORT_DEVICE','typo')
    with pytest.raises(ValueError,match='SAM3D_EXPORT_DEVICE'):
        ply2glb.choose_export_device()


def test_shell_skips_completed_pipeline_exports(tmp_path):
    import subprocess
    root=Path(__file__).resolve().parents[1]
    runner=tmp_path/'run.sh'; runner.write_text((root/'run.sh').read_text())
    checkpoint=tmp_path/'checkpoints/hf/checkpoints'; checkpoint.mkdir(parents=True)
    for name in ['pipeline.yaml','ss_generator.ckpt','slat_generator.ckpt','ss_decoder.ckpt','slat_decoder_gs.ckpt','slat_decoder_mesh.ckpt']:
        (checkpoint/name).write_text('stub')
    py=tmp_path/'.venv/bin/python3'; py.parent.mkdir(parents=True); py.symlink_to(sys.executable)
    (tmp_path/'cli.py').write_text("import os,json\np=os.path.join(os.path.dirname(__file__),'output space ✓');os.makedirs(p,exist_ok=True)\nopen(os.environ['SAM3D_MANIFEST'],'a').write(p+'\\t0\\t0\\tcomplete\\tauto\\tquality\\n')\njson.dump({'export':0,'run_token':os.environ['SAM3D_MANIFEST']},open(p+'/.sam3d-batch-export.json','w'))\n")
    (tmp_path/'ply2glb.py').write_text("raise SystemExit('DUPLICATE EXPORT')")
    result=subprocess.run(['bash',str(runner)],capture_output=True,text=True)
    assert result.returncode==0, result.stdout+result.stderr
    assert 'DUPLICATE EXPORT' not in result.stderr

def test_explicit_accelerator_export_falls_back_serial():
    import batch_scheduler as scheduler
    for name in ['SAM3D_EXPORT_DEVICE','SAM3D_MESH_DECODE_DEVICE']:
        assert not scheduler.overlap_eligible({name:'mps'},memory=(128*2**30,80*2**30))


def test_interrupt_reaps_both_children(tmp_path):
    import subprocess, signal, time
    import batch_scheduler
    pidfile=tmp_path/'pids'
    child="import os,sys,time; open(sys.argv[1],'a').write(str(os.getpid())+'\\n'); time.sleep(float(sys.argv[2]))"
    script="import batch_scheduler,sys; child=sys.argv[2]; jobs=[dict(name=str(i),generate=[sys.executable,'-c',child,sys.argv[1],'.08'],export=[sys.executable,'-c',child,sys.argv[1],'20']) for i in range(2)]; jobs[1]['generate'][-1]='20'; batch_scheduler.run_batch(jobs,overlap=True)"
    parent=subprocess.Popen([sys.executable,'-c',script,str(pidfile),child],env={**os.environ,'PYTHONPATH':str(Path(batch_scheduler.__file__).parent)},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline and (not pidfile.exists() or len(pidfile.read_text().splitlines())<3): time.sleep(.02)
        assert len(pidfile.read_text().splitlines())==3
        parent.send_signal(signal.SIGINT); parent.wait(timeout=8)
        for pid in map(int,pidfile.read_text().splitlines()):
            with pytest.raises(ProcessLookupError): os.kill(pid,0)
    finally:
        if parent.poll() is None: parent.kill(); parent.wait()

def test_cli_scheduler_preserves_manifest_and_export_options(tmp_path, monkeypatch):
    import cli
    worker=tmp_path/'worker'
    worker.write_text('#!'+sys.executable+'\n'+'''import sys,os,json
args=sys.argv[1:]
if '--batch-worker' in args:
    folder=args[args.index('--output-dir')+1]; os.makedirs(folder,exist_ok=True)
    open(folder+'/slat.pt','w').write('kept latent')
    open(os.environ['SAM3D_MANIFEST'],'a').write(folder+'\\t7\\t25\\tcomplete\\tauto\\tquality\\n')
else:
    folder=args[-1]; json.dump(dict(os.environ),open(folder+'/export-env.json','w'))
''')
    worker.chmod(0o755)
    manifest=tmp_path/'manifest'; manifest.write_text('')
    monkeypatch.setenv('SAM3D_MANIFEST',str(manifest))
    monkeypatch.setenv('SAM3D_SOURCE_ONLY','1')
    monkeypatch.setenv('SAM3D_REPAIR_PROFILE','final')
    monkeypatch.setenv('SAM3D_TEXTURE_PROFILE','preview')
    monkeypatch.setenv('SAM3D_GAME_METHOD','fast')
    monkeypatch.setenv('SAM3D_GAME_TARGET_VERTICES','6000')
    monkeypatch.setattr(cli.sys,'executable',str(worker))
    assert cli._run_pipelined_folder_batch([dict(paths=['a space ✓.png'],name='a space ✓')],str(tmp_path/'out'),8)
    directory=Path(manifest.read_text().split('\t')[0])
    env=json.loads((directory/'export-env.json').read_text())
    for key,value in [('SAM3D_SOURCE_ONLY','1'),('SAM3D_REPAIR_PROFILE','final'),('SAM3D_TEXTURE_PROFILE','preview'),('SAM3D_GAME_METHOD','fast'),('SAM3D_GAME_TARGET_VERTICES','6000'),('SAM3D_PROGRESS_DONE','7'),('SAM3D_PROGRESS_TOTAL','25'),('SAM3D_MESH_DECODE_DEVICE','cpu')]:
        assert env[key]==value
    marker=json.loads((directory/'.sam3d-batch-export.json').read_text()); assert marker['run_token']==str(manifest); assert marker['export']==0
    assert (directory/'slat.pt').read_text()=='kept latent'


def test_unavailable_executable_is_reported_and_continues():
    import batch_scheduler as scheduler
    records=scheduler.run_batch([dict(name='missing',generate=['/no/such/executable'],export=['/no/such/export']),dict(name='export missing',generate=[sys.executable,'-c','pass'],export=['/no/such/export'])],overlap=True)
    assert [(r['generation'],r['export']) for r in records]==[(127,None),(0,127)]

@pytest.mark.parametrize('exitcode',[0,9])
def test_shell_never_retries_run_marker(tmp_path,exitcode):
    import subprocess
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'run.sh').write_text((root/'run.sh').read_text())
    checkpoint=tmp_path/'checkpoints/hf/checkpoints'; checkpoint.mkdir(parents=True)
    for name in ['pipeline.yaml','ss_generator.ckpt','slat_generator.ckpt','ss_decoder.ckpt','slat_decoder_gs.ckpt','slat_decoder_mesh.ckpt']: (checkpoint/name).write_text('stub')
    py=tmp_path/'.venv/bin/python3'; py.parent.mkdir(parents=True); py.symlink_to(sys.executable)
    (tmp_path/'cli.py').write_text("import os,json\np=os.path.join(os.path.dirname(__file__),'output');os.makedirs(p,exist_ok=True)\nopen(os.environ['SAM3D_MANIFEST'],'a').write(p+'\\t0\\t0\\tcomplete\\tauto\\tquality\\n')\njson.dump({'export':"+str(exitcode)+",'run_token':os.environ['SAM3D_MANIFEST']},open(p+'/.sam3d-batch-export.json','w'))\n")
    (tmp_path/'ply2glb.py').write_text("raise SystemExit('DUPLICATE EXPORT')")
    result=subprocess.run(['bash',str(tmp_path/'run.sh')],capture_output=True,text=True)
    assert result.returncode==(0 if exitcode==0 else 1)
    assert 'DUPLICATE EXPORT' not in result.stderr


def test_cpu_export_environment_overrides_mesh_accelerator():
    import batch_scheduler as scheduler
    env=scheduler.cpu_export_env({'SAM3D_MESH_DECODE_DEVICE':'mps','SAM3D_SOURCE_ONLY':'0'})
    assert env['SAM3D_MESH_DECODE_DEVICE']=='cpu'
    assert env['SAM3D_SOURCE_ONLY']=='0'
    assert env['SAM3D_EXPORT_THREADS']=='2'
    assert env['SAM3D_EXPORT_INTEROP_THREADS']=='1'
    assert env['SAM3D_GAUSSIAN_KD_WORKERS']=='2'


def test_invalid_resource_budget_rejected(monkeypatch):
    import ply2glb
    for raw in ['0','-1','bad']:
        monkeypatch.setenv('SAM3D_EXPORT_THREADS',raw)
        with pytest.raises(ValueError,match='positive integer'): ply2glb.configure_export_threads()

def test_interrupt_during_spawn_handshake_reaps_unassigned_child(monkeypatch):
    import batch_scheduler as scheduler
    import signal, subprocess
    real_popen=subprocess.Popen
    children=[]
    def interrupt_after_spawn(*args,**kwargs):
        child=real_popen(*args,**kwargs)
        children.append(child)
        os.kill(os.getpid(),signal.SIGINT)
        return child
    monkeypatch.setattr(scheduler.subprocess,'Popen',interrupt_after_spawn)
    try:
        with pytest.raises(KeyboardInterrupt):
            scheduler.run_batch([dict(name='spawn race',generate=[sys.executable,'-c','import time; time.sleep(20)'],export=[sys.executable,'-c','pass'])],overlap=True)
        assert len(children)==1
        assert children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None: child.kill()
            child.wait()

def test_exec_children_do_not_inherit_blocked_sigint(tmp_path):
    import batch_scheduler as scheduler
    code="import signal,sys; assert signal.SIGINT not in signal.pthread_sigmask(signal.SIG_BLOCK, []); open(sys.argv[1],'w').write('unblocked')"
    records=scheduler.run_batch([dict(name='signals',generate=[sys.executable,'-c',code,str(tmp_path/'generator')],export=[sys.executable,'-c',code,str(tmp_path/'exporter')])],overlap=True)
    assert records[0]['generation']==records[0]['export']==0
    assert (tmp_path/'generator').read_text()==(tmp_path/'exporter').read_text()=='unblocked'


@pytest.mark.parametrize('raw', ['0', 'false', 'no', 'off', ' FALSE '])
def test_repair_disabled_is_ineligible(raw):
    import batch_scheduler as scheduler
    assert not scheduler.overlap_eligible({'SAM3D_SURFACE_REPAIR': raw}, memory=(128*2**30,80*2**30))

@pytest.mark.parametrize('setting,value', [('SAM3D_SURFACE_REPAIR','0'), ('SAM3D_TEXTURE_MODE','opt'), ('SAM3D_TEXTURE_MODE','vertex')])
def test_cli_actual_serial_dispatch_preserves_options(tmp_path, monkeypatch, setting, value):
    import cli, batch_scheduler
    worker=tmp_path/'serial-worker'
    worker.write_text('#!'+sys.executable+'\n'+"import sys,os,json\nargs=sys.argv[1:]; folder=args[args.index('--output-dir')+1]; os.makedirs(folder,exist_ok=True); json.dump({'args':args,'env':dict(os.environ)},open(folder+'/worker.json','w'))\n")
    worker.chmod(0o755)
    monkeypatch.setattr(cli.sys,'executable',str(worker))
    monkeypatch.setattr(batch_scheduler,'host_memory',lambda:(128*2**30,80*2**30))
    monkeypatch.setenv('SAM3D_FULL_BATCH','1')
    monkeypatch.setenv(setting,value)
    monkeypatch.setenv('SAM3D_GAME_TARGET_VERTICES','6000')
    monkeypatch.setenv('SAM3D_GAME_METHOD','fast')
    monkeypatch.setenv('SAM3D_SOURCE_ONLY','0')
    monkeypatch.setenv('SAM3D_EXPORT_DEVICE','auto')
    monkeypatch.setenv('SAM3D_MESH_DECODE_DEVICE','auto')
    assert cli._run_folder_batch([dict(paths=['a space.png'],name='a')],str(tmp_path/'out'),9)
    records=list((tmp_path/'out').glob('*/worker.json'))
    assert len(records)==1
    record=json.loads(records[0].read_text())
    assert '--batch-worker' in record['args']
    assert record['args'][record['args'].index('--steps')+1]=='9'
    for key,want in [(setting,value),('SAM3D_GAME_TARGET_VERTICES','6000'),('SAM3D_GAME_METHOD','fast'),('SAM3D_SOURCE_ONLY','0'),('SAM3D_EXPORT_DEVICE','auto'),('SAM3D_MESH_DECODE_DEVICE','auto')]:
        assert record['env'][key]==want
    assert not list((tmp_path/'out').glob('*/.sam3d-batch-export.json'))


@pytest.mark.parametrize('raw', ['1','true','yes','on','','garbage'])
def test_repair_enabled_matches_exporter_boolean_semantics(raw, monkeypatch):
    import batch_scheduler as scheduler
    import ply2glb
    monkeypatch.setenv('SAM3D_SURFACE_REPAIR',raw)
    assert ply2glb.bool_env('SAM3D_SURFACE_REPAIR',True)
    assert scheduler.overlap_eligible({'SAM3D_SURFACE_REPAIR':raw},memory=(128*2**30,80*2**30))


def test_overlap_honors_explicit_cpu_export_budgets():
    import batch_scheduler as scheduler
    original = {"SAM3D_EXPORT_THREADS": "8", "SAM3D_EXPORT_INTEROP_THREADS": "2",
                "SAM3D_GAUSSIAN_KD_WORKERS": "4", "OMP_NUM_THREADS": "1"}
    result = scheduler.cpu_export_env(original)
    for name in ["SAM3D_EXPORT_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        assert result[name] == "8"
    assert result["SAM3D_EXPORT_INTEROP_THREADS"] == "2"
    assert result["SAM3D_GAUSSIAN_KD_WORKERS"] == "4"
    assert original["OMP_NUM_THREADS"] == "1"
    assert "OPENBLAS_NUM_THREADS" not in original
