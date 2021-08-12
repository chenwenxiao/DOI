#!/usr/bin/env python
import itertools
import os
import signal
import socket
import subprocess
import sys
import uuid
from contextlib import contextmanager
from functools import partial
from threading import Thread

import click
from dask.distributed import Client, as_completed


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


def run_proc(args, work_dir=None, log_file=None):
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    python_path = env.get('PYTHONPATH', '')
    python_path = (os.path.split(os.path.abspath(__file__))[0] +
                   os.pathsep + python_path)
    env['PYTHONPATH'] = python_path

    header = '>>> Start process: {}\n    at: {}'.format(args, work_dir)
    print(header)

    @contextmanager
    def log_writer():
        if log_file is not None:
            with open(log_file, 'ab') as f:
                fd = f.fileno()
                os.write(fd, header.encode('utf-8') + b'\n\n')
                yield partial(os.write, fd)
        else:
            yield (lambda s: None)

    with log_writer() as writer, \
            exec_proc(args, on_stdout=writer, stderr_to_stdout=True,
                      cwd=work_dir, env=env) as proc:
        exit_code = proc.wait()
        footer = '>>> Exit code is {}'.format(exit_code)
        print(footer)
        if writer is not None:
            writer(b'\n\n' + footer.format(exit_code).encode('utf-8') + b'\n')

        if exit_code != 0:
            raise RuntimeError('Exit code != 0: {}'.format(exit_code))
        return exit_code


@click.command()
@click.option('-s', '--scheduler', default=socket.gethostname() + ':7891',
              help='Specify the dask scheduler.')
@click.option('-l', '--log-file', default=None, required=False,
              help='Save console log to this file.')
@click.option('-r', '--retries', type=int, default=0, required=False,
              help='Maximum number of retries on error.')
@click.option('-N', '--instance-num', type=int, default=1, required=False,
              help='Start this number of instances.')
@click.option('-w', '--work-dir', type=str, required=False, default='.',
              help='Work directory for the task.  Defaults to current dir.')
@click.option('--no-wait', is_flag=True, default=False, required=False,
              help='If specified, submit the task to scheduler and exit '
                   'without waiting for the program to finish.')
@click.option('template_args', '--template-arg', multiple=True, default=None,
              required=False)
@click.option('template_values', '--template-val', multiple=True, default=None,
              required=False)
@click.argument('args', nargs=-1)
def main(scheduler, log_file, retries, instance_num, work_dir, no_wait,
         template_args, template_values, args):
    work_dir = os.path.abspath(work_dir)
    if log_file is not None:
        log_file = os.path.abspath(log_file)

    # parse with template args
    template_keys = []
    template_value_list = []

    for s in template_values or ():
        key, val = s.split('=')
        val_list = [i for i in val.split(',')]
        template_keys.append(key)
        template_value_list.append(val_list)

    def iter_template_dict():
        for values in itertools.product(*template_value_list):
            yield {k: v for k, v in zip(template_keys, values)}

    # do submit the tasks
    client = Client(scheduler)
    task_map = {}

    def submit(args):
        for i in range(instance_num):
            next_task_id = len(task_map)
            key = 'run_proc:{}'.format(uuid.uuid4())
            local_log_file = log_file
            if local_log_file is not None:
                local_log_file = list(os.path.splitext(log_file))
                local_log_file[0] = '{}.{}'.format(local_log_file[0], next_task_id)
                local_log_file = ''.join(local_log_file)

            f = client.submit(
                run_proc, args=args, work_dir=work_dir, log_file=local_log_file,
                key=key, retries=retries
            )
            task_map[f] = next_task_id
            print('> submitted task #{}: {}: {}'.format(next_task_id, key, args))
            tasks.append(f)

    tasks = []

    if template_args:
        for template_dict in iter_template_dict():
            the_args = list(args)
            for a in template_args:
                the_args.append(a.format(**template_dict))
            submit(the_args)
    else:
        submit(args)

    # wait for the tasks to finish
    if not no_wait:
        for f in as_completed(tasks, with_results=False):
            try:
                _ = f.result()
            except RuntimeError as ex:
                print('$ task #{}: error: {}'.format(task_map[f], ex))
            else:
                print('$ task #{}: done'.format(task_map[f]))


if __name__ == '__main__':
    main()
