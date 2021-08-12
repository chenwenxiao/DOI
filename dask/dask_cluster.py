import logging
import os
import re
import signal
import socket
import subprocess
import sys
from contextlib import contextmanager
from threading import Thread

import click

os.environ['MLSTORAGE_SERVER_URI'] = 'http://mlserver.ipwx.me:7897'


def timed_wait_proc(proc, timeout):
    try:
        return proc.wait(timeout)
    except subprocess.TimeoutExpired:
        return None


@contextmanager
def exec_proc(args, on_stdout=None, on_stderr=None, stderr_to_stdout=False,
              buffer_size=16*1024, ctrl_c_timeout=3, kill_timeout=60, **kwargs):
    """
    Execute an external program within a context.

    Args:
        args: Arguments of the program.
        on_stdout ((bytes) -> None): Callback for capturing stdout.
        on_stderr ((bytes) -> None): Callback for capturing stderr.
        stderr_to_stdout (bool): Whether or not to redirect stderr to
            stdout?  If specified, `on_stderr` will be ignored.
            (default :obj:`False`)
        buffer_size (int): Size of buffers for reading from stdout and stderr.
        ctrl_c_timeout (int): Seconds to wait for the program to
            respond to CTRL+C signal. (default 3)
        kill_timeout (int): Seconds to wait for the program to terminate after
            being killed. (default 60)
        **kwargs: Other named arguments passed to :func:`subprocess.Popen`.

    Yields:
        subprocess.Popen: The process object.
    """
    # check the arguments
    if stderr_to_stdout:
        kwargs['stderr'] = subprocess.STDOUT
        on_stderr = None
    if on_stdout is not None:
        kwargs['stdout'] = subprocess.PIPE
    if on_stderr is not None:
        kwargs['stderr'] = subprocess.PIPE

    # output reader
    def reader_func(fd, action):
        while not giveup_waiting[0]:
            buf = os.read(fd, buffer_size)
            if not buf:
                break
            action(buf)

    def make_reader_thread(fd, action):
        th = Thread(target=reader_func, args=(fd, action))
        th.daemon = True
        th.start()
        return th

    # internal flags
    giveup_waiting = [False]

    # launch the process
    stdout_thread = None  # type: Thread
    stderr_thread = None  # type: Thread
    proc = subprocess.Popen(args, **kwargs)

    try:
        if on_stdout is not None:
            stdout_thread = make_reader_thread(proc.stdout.fileno(), on_stdout)
        if on_stderr is not None:
            stderr_thread = make_reader_thread(proc.stderr.fileno(), on_stderr)

        try:
            yield proc
        except KeyboardInterrupt:  # pragma: no cover
            if proc.poll() is None:
                # Wait for a while to ensure the program has properly dealt
                # with the interruption signal.  This will help to capture
                # the final output of the program.
                # TODO: use signal.signal instead for better treatment
                _ = timed_wait_proc(proc, 1)

    finally:
        if proc.poll() is None:
            # First, try to interrupt the process with Ctrl+C signal
            ctrl_c_signal = (signal.SIGINT if sys.platform != 'win32'
                             else signal.CTRL_C_EVENT)
            os.kill(proc.pid, ctrl_c_signal)
            if timed_wait_proc(proc, ctrl_c_timeout) is None:
                # If the Ctrl+C signal does not work, terminate it.
                proc.kill()
            # Finally, wait for at most 60 seconds
            if timed_wait_proc(proc, kill_timeout) is None:  # pragma: no cover
                giveup_waiting[0] = True

        # Close the pipes such that the reader threads will ensure to exit,
        # if we decide to give up waiting.
        def close_pipes():
            for f in (proc.stdout, proc.stderr, proc.stdin):
                if f is not None:
                    f.close()

        if giveup_waiting[0]:  # pragma: no cover
            close_pipes()

        # Wait for the reader threads to exit
        for th in (stdout_thread, stderr_thread):
            if th is not None:
                th.join()

        # Ensure all the pipes are closed.
        if not giveup_waiting[0]:
            close_pipes()


class InterruptHandler(object):

    def __init__(self, fn, sig=signal.SIGINT):
        self.fn = fn
        self.sig = sig
        self._released = False

    def __enter__(self):
        self._interrupted = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self._interrupted = True
            self.fn()

        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self._released:
            return False
        signal.signal(self.sig, self.original_handler)
        self._released = True


def interruption_handler():
    raise KeyboardInterrupt()


def run_scheduler():
    hostname = socket.gethostname()
    return exec_proc([
        'dask-scheduler',
        '--port=7891',
        '--bokeh',
        '--bokeh-port=7892',
        '--local-directory=/mnt/mfs/var/dask-cwx17/scheduler-' + hostname
    ])


def run_worker(gpu):
    hostname = socket.gethostname()
    worker_name = 'worker-{}-gpu-{}'.format(hostname, gpu)
    scheduler = hostname + ':7891'
    args = [
        'dask-worker',
        scheduler,
        '--nprocs=1',
        '--nthreads=1',
        '--name=' + worker_name,
        '--local-directory=/mnt/mfs/var/dask-cwx17/' + worker_name
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return exec_proc(args, env=env)


@click.command()
@click.option('-g', '--gpu', required=True,
              help='Specify the GPU devices.')
def main(gpu):
    """Start a Dask cluster for executing GPU tasks."""
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s]: %(message)s'
    )

    # parse the GPU resources
    gpu_list = []
    for g in gpu.split(','):
        g = g.strip()
        if g:
            m = re.match(r'^(\d+)-(\d+)$', g)
            if not m:
                gpu_list.append(int(g))
            else:
                for i in range(int(m.group(1)), int(m.group(2)) + 1):
                    gpu_list.append(int(i))

    gpu_list = sorted(set(gpu_list))
    if not gpu_list:
        raise ValueError('Empty GPU devices.')

    logging.info('GPU devices: {}'.format(','.join(map(str, gpu_list))))

    # start the dask scheduler
    @contextmanager
    def recursively_start_workers(i):
        if i >= len(gpu_list):
            yield []
        else:
            gpu = gpu_list[i]
            with run_worker(gpu) as proc:
                logging.info('Worker on GPU {} started.'.format(gpu))
                with recursively_start_workers(i + 1) as procs:
                    yield [proc] + procs
            logging.info('Worker on GPU {} stopped.'.format(gpu))

    with InterruptHandler(interruption_handler, signal.SIGINT), \
            InterruptHandler(interruption_handler, signal.SIGTERM), \
            run_scheduler() as scheduler_proc:
        logging.info('Scheduler started.')
        with recursively_start_workers(0) as procs:
            for proc in procs:
                proc.wait()
        scheduler_proc.wait()

    logging.info('Scheduler stopped.')


if __name__ == '__main__':
    main()
