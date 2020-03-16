"""system 및 os 단에서 파일 및 명령등에 접근하는 모듈"""

import os
import sys
import platform
from typing import Optional, List


# find absolute current working directory
CWD = os.path.dirname(sys.argv[0])
if platform.system() == "Windows":
    CWD = CWD.replace("/", "\\")
CWD = os.path.realpath(CWD)


def find_executable(executable: str, path: Optional[List] = None) -> List:
    """Tries to find 'executable' in the directories listed in 'path'.

    A string listing directories separated by 'os.pathsep'; defaults to
    os.environ['PATH'].  Returns the complete filename or None if not found.
    """
    if path is None:
        path = os.environ["PATH"]

    paths = path.split(os.pathsep)

    # remove duplicating paths
    if sys.platform == "win32":
        paths = set(path.lower() for path in paths)

    executables = []

    if os.path.isfile(executable):
        executables.append(executable)

    for path in paths:
        filename = os.path.join(path, executable)
        if os.path.isfile(filename):
            # the file exists, we have a shot at spawn working
            executables.append(filename)

    return executables
