"""Bounded fresh-process generation and CPU export for full folder runs."""
import os
import signal
import subprocess
import time
from collections import deque


def host_memory():
    """Return physical and available bytes; unknown memory disables overlap."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.total, memory.available
    except (ImportError, OSError, AttributeError):
        return 0, 0


def overlap_eligible(env=None, memory=None):
    env = os.environ if env is None else env
    if env.get('SAM3D_PIPELINED_BATCH', 'auto').strip().lower() not in ('auto', '1', 'true', 'yes'):
        return False
    # Match ply2glb.bool_env: legacy fill_holes cleanup can select an accelerator.
    if env.get('SAM3D_SURFACE_REPAIR', '1').strip().lower() in ('0', 'false', 'no', 'off'):
        return False
    if env.get('SAM3D_TEXTURE_MODE', 'gaussian').strip().lower() != 'gaussian':
        return False
    if any(env.get(name, 'auto').strip().lower() not in ('auto', 'cpu')
           for name in ('SAM3D_EXPORT_DEVICE', 'SAM3D_MESH_DECODE_DEVICE')):
        return False
    total, available = host_memory() if memory is None else memory
    return total >= 64 * 2**30 and available >= 32 * 2**30


def cpu_export_env(env):
    result = dict(env)
    result.update(SAM3D_EXPORT_DEVICE='cpu', SAM3D_MESH_DECODE_DEVICE='cpu',
                  SAM3D_EXPORT_THREADS='2', SAM3D_EXPORT_INTEROP_THREADS='1',
                  SAM3D_GAUSSIAN_KD_WORKERS='2', OMP_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    return result


def _stop(process):
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait()


def run_batch(jobs, *, overlap=False, env=None, on_result=None, memory_check=None, export_env_factory=None):
    """Run command-list jobs with <=1 generator, <=1 exporter, <=1 pending output.

    Each job has name, generate/export argv, optional generation_env/export_env.
    Results retain generation and export exit codes (None means export skipped).
    Child output inherits the terminal. Callback receives each final result once.
    A memory check can suspend new overlapping generation, never serial progress.
    """
    env = dict(os.environ if env is None else env)
    records = [dict(name=job['name'], generation=None, export=None) for job in jobs]
    pending = deque()
    generator = exporter = None
    generation_index = export_index = None
    next_index = 0
    owned_children = set()

    def finish(index):
        if on_result:
            on_result(records[index])

    def launch(command, child_env):
        # Register ownership before delivering Ctrl-C, including Popen's exec handshake.
        previous_handler = signal.getsignal(signal.SIGINT)
        deferred_interrupt = []
        def defer_interrupt(signum, frame):
            deferred_interrupt.append((signum, frame))
        if previous_handler != signal.SIG_IGN:
            signal.signal(signal.SIGINT, defer_interrupt)
        try:
            process = subprocess.Popen(command, env=child_env, start_new_session=True)
            owned_children.add(process)
            return process, None
        except OSError as error:
            return None, str(error)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            if deferred_interrupt:
                signum, frame = deferred_interrupt[0]
                if callable(previous_handler):
                    previous_handler(signum, frame)
                elif previous_handler == signal.SIG_DFL:
                    os.kill(os.getpid(), signum)

    try:
        while next_index < len(jobs) or generator or exporter or pending:
            if exporter is not None and exporter.poll() is not None:
                records[export_index]['export'] = exporter.wait()
                owned_children.discard(exporter)
                exporter = None
                finish(export_index)
            if generator is not None and generator.poll() is not None:
                status = generator.wait()
                owned_children.discard(generator)
                records[generation_index]['generation'] = status
                generator = None
                if status == 0:
                    pending.append(generation_index)
                else:
                    finish(generation_index)
            if exporter is None and pending:
                export_index = pending.popleft()
                child_env = dict(env)
                if export_env_factory:
                    child_env.update(export_env_factory(jobs[export_index]))
                child_env.update(jobs[export_index].get('export_env', {}))
                # CPU selection and bounded resources cannot be overridden by a job.
                if overlap:
                    child_env = cpu_export_env(child_env)
                exporter, error = launch(jobs[export_index]['export'], child_env)
                if error:
                    records[export_index].update(export=127, error=error)
                    finish(export_index)
            can_overlap = overlap and (memory_check is None or memory_check())
            if generator is None and not pending and next_index < len(jobs) and (exporter is None or can_overlap):
                generation_index = next_index
                next_index += 1
                child_env = dict(env)
                child_env.update(jobs[generation_index].get('generation_env', {}))
                generator, error = launch(jobs[generation_index]['generate'], child_env)
                if error:
                    records[generation_index].update(generation=127, error=error)
                    finish(generation_index)
            if generator or exporter:
                time.sleep(.02)
    finally:
        for child in owned_children:
            _stop(child)
    return records
