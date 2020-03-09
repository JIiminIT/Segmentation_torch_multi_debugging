"""Tinycat commandline application interface

Raises:
    ImportError: raises when app has not found
"""


import importlib
import subprocess
import sys
import tinycat as cat
from tinycat.app import apps


def main():
    print(cat.__doc__)
    print("Supported apps: %s" % apps)
    print("Version: %s" % cat.__version__)

    if len(sys.argv) > 2:
        cmd = "%s %s" % (sys.argv[1], " ".join(sys.argv[2:]))
        print("calling %s" % cmd)

        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
