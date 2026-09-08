"""Killable execution: Docker for generated programs; subprocess for CPU tests."""
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from .common import boxed

# Resource limits are also applied inside the Docker container.
PRELUDE = """import resource
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
"""

PROCESS_PRELUDE = """import os, errno, resource, pyseccomp
resource.setrlimit(resource.RLIMIT_AS, (2147483648, 2147483648))
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
os.setgroups([])
os.setgid(65534)
os.setuid(65534)
_filter = pyseccomp.SyscallFilter(defaction=pyseccomp.ALLOW)
for _call in ('socket', 'socketpair', 'connect', 'fork', 'vfork', 'clone', 'clone3',
              'execve', 'execveat', 'ptrace', 'mount', 'umount2', 'unshare', 'setns'):
    _filter.add_rule(pyseccomp.ERRNO(errno.EPERM), _call)
_filter.load()
"""


def execute(code, timeout=15, backend="docker", image="symplan-executor:v1"):
    start = time.monotonic()
    name = "symplan-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="symplan-exec-") as tmp:
        if backend == "docker":
            command = ["docker", "run", "--rm", "-i", "--name", name,
                       "--network=none", "--read-only", "--cap-drop=ALL",
                       "--security-opt=no-new-privileges", "--pids-limit=64",
                       "--memory=1g", "--cpus=1", "--user=65534:65534",
                       "--tmpfs=/tmp:rw,noexec,nosuid,size=64m", "--workdir=/tmp",
                       image, "python", "-I", "-"]
        elif backend in {"local-test", "process"}:
            # Test-only: process isolation is not a security boundary.
            command = [sys.executable, "-I", "-"]
            if backend == "process" and (sys.platform != "linux" or os.geteuid() != 0):
                raise RuntimeError("Process executor requires Linux root to drop to nobody; otherwise use Docker")
        else:
            raise ValueError(backend)
        env = {k: v for k, v in os.environ.items() if k in
               {"PATH", "DOCKER_HOST", "DOCKER_CONTEXT", "HOME", "SYSTEMROOT"}}
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        with open(Path(tmp) / "stdout", "w+b") as stdout, open(Path(tmp) / "stderr", "w+b") as stderr:
            # Limit the host-side output files too, including Docker CLI output.
            bounded_command = [sys.executable, "-c", PRELUDE +
                               "import os, sys; os.execvp(sys.argv[1], sys.argv[1:])", *command]
            proc = subprocess.Popen(bounded_command, stdin=subprocess.PIPE, stdout=stdout,
                                    stderr=stderr, cwd=tmp, env=env, start_new_session=True)
            status = "error"
            try:
                proc.communicate((PRELUDE + (PROCESS_PRELUDE if backend == "process" else "") + code).encode(), timeout=timeout)
                status = "success" if proc.returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                status = "timeout"
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
            finally:
                if backend == "docker":
                    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
            stdout.seek(0)
            stderr.seek(0)
            output = stdout.read(65536).decode(errors="replace")
            error = stderr.read(65536).decode(errors="replace")
            if backend == "docker" and (proc.returncode in {125, 126, 127} or error.startswith("docker:")):
                raise RuntimeError("Docker infrastructure failure: " + error)
    answer = boxed(output) if status == "success" else None
    if answer is not None and answer.lower() in {"none", "nan", "null", "undefined", "invalid"}:
        answer = None
    if status == "success" and answer is None:
        status = "unparseable"
        error = "Program completed but did not print a nonempty valid boxed answer."
    return {"status": status, "answer": answer, "stdout": output,
            "diagnosis": error[-4000:] or ("Execution time limit exceeded." if status == "timeout" else ""),
            "seconds": time.monotonic() - start}
