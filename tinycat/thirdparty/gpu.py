"""NVIDIA GPU 정보 확인 및 호환성 보완 유틸리티 """
# GPUtil - GPU utilization
#
# A Python module for programmically getting the GPU utilization from NVIDA gpus using nvidia-smi
#
# Author: Anders Krogh Mortensen (anderskm)
# Date:   16 January 2017
# Web:    https://github.com/anderskm/gputil
#
# LICENSE
#
# MIT License
#
# Copyright (c) 2017 anderskm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Implemented for Python3 and Windows
# iam@nyanye.com

import os
import time
import platform
from subprocess import Popen, PIPE
import numpy as np
from tinycat.system import CWD, find_executable


CUDA_DLLNAMES = ["nvcuda", "nvapi64", "nvfatbinaryLoader"]


class GPU:
    """Simple Datastructure for GPU"""

    def __init__(
        self,
        _id,
        uuid,
        load,
        memory_total,
        memory_used,
        memory_free,
        driver,
        gpu_name,
        serial,
        display_mode,
        display_active,
    ):
        self.id = _id
        self.uuid = uuid
        self.load = load
        self.memory_util = float(memory_used) / float(memory_total)
        self.memory_total = memory_total
        self.memory_used = memory_used
        self.memory_free = memory_free
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active

    def describe(self):
        """prints out current gpu informations """
        print("-" * 80)
        for line in self.get_summary():
            print(line)
        print("-" * 80)

    def get_summary(self):
        summary = [
            "GPU_ID: %s" % self.id,
            "UUID: %s" % self.uuid,
            "LOAD: %s" % self.load,
            "MEMORY_UTIL: %s" % self.memory_util,
            "MEMORY_TOTAL: %s" % self.memory_total,
            "MEMORY_USED: %s" % self.memory_used,
            "MEMORY_FREE: %s" % self.memory_free,
            "DRIVER: %s" % self.driver,
            "NAME: %s" % self.name,
            "SERIAL: %s" % self.serial,
            "DISPLAY_MODE: %s" % self.display_mode,
            "DISPLAY_ACTIVE: %s" % self.display_active,
        ]
        return summary


def safe_float_cast(str_number):
    try:
        number = float(str_number)
    except ValueError:
        number = float("nan")
    return number


def get_gpus():
    if platform.system() == "Windows":
        nvidia_smi = (
            "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"
            % os.environ["systemdrive"]
        )
    else:
        nvidia_smi = "nvidia-smi"

    command = [
        nvidia_smi,
        "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode",
        "--format=csv,noheader,nounits",
    ]

    # Get ID, processing and memory utilization for all gpus
    p = Popen(command, stdout=PIPE)
    output = p.stdout.read().decode("UTF-8")
    # Parse output
    # Split on line break
    lines = output.split(os.linesep)
    num_devices = len(lines) - 1
    device_ids = np.empty(num_devices, dtype=int)
    gpu_util = np.empty(num_devices, dtype=float)
    mem_total = np.empty(num_devices, dtype=float)
    mem_used = np.empty(num_devices, dtype=float)
    mem_free = np.empty(num_devices, dtype=float)
    driver = []
    gpus = []
    for g in range(num_devices):
        line = lines[g]
        vals = line.split(", ")
        for i in range(11):
            if i == 0:
                device_ids[g] = int(vals[i])
            elif i == 1:
                uuid = vals[i]
            elif i == 2:
                gpu_util[g] = safe_float_cast(vals[i]) / 100
            elif i == 3:
                mem_total[g] = safe_float_cast(vals[i])
            elif i == 4:
                mem_used[g] = safe_float_cast(vals[i])
            elif i == 5:
                mem_free[g] = safe_float_cast(vals[i])
            elif i == 6:
                driver = vals[i]
            elif i == 7:
                gpu_name = vals[i]
            elif i == 8:
                serial = vals[i]
            elif i == 9:
                display_active = vals[i]
            elif i == 10:
                display_mode = vals[i]
        gpus.append(
            GPU(
                device_ids[g],
                uuid,
                gpu_util[g],
                mem_total[g],
                mem_used[g],
                mem_free[g],
                driver,
                gpu_name,
                serial,
                display_mode,
                display_active,
            )
        )
    return gpus  # (device_ids, gpu_util, memUtil)


def get_available(
    order="first",
    limit=1,
    max_load=0.5,
    max_memory=0.5,
    include_nan=False,
    exclude_id=[],
    exclude_uuid=[],
):
    # Get device IDs, load and memory usage
    gpus = get_gpus()

    # Determine, which gpus are available
    gpu_availability = np.array(
        get_availability(
            gpus,
            max_load=max_load,
            max_memory=max_memory,
            include_nan=include_nan,
            exclude_id=exclude_id,
            exclude_uuid=exclude_uuid,
        )
    )
    available_gpu_index = np.where(gpu_availability == 1)[0]
    # Discard unavailable gpus
    gpus = [gpus[g] for g in available_gpu_index]

    # Sort available gpus according to the order argument
    if order == "first":
        gpus.sort(key=lambda x: np.Inf if np.isnan(x.id) else x.id, reverse=False)
    elif order == "last":
        gpus.sort(key=lambda x: -np.Inf if np.isnan(x.id) else x.id, reverse=True)
    elif order == "random":
        gpus = [gpus[g] for g in np.random.permutation(range(len(gpus)))]
    elif order == "load":
        gpus.sort(key=lambda x: np.Inf if np.isnan(x.load) else x.load, reverse=False)
    elif order == "memory":
        gpus.sort(
            key=lambda x: np.Inf if np.isnan(x.memory_util) else x.memory_util,
            reverse=False,
        )

    # Extract the number of desired gpus, but limited to the total number of available gpus
    gpus = gpus[0 : min(limit, len(gpus))]

    # Extract the device IDs from the gpus and return them
    device_ids = [gpu.id for gpu in gpus]

    return device_ids


def get_availability(
    gpus,
    max_load=0.5,
    max_memory=0.5,
    include_nan=False,
    exclude_id=[],
    exclude_uuid=[],
):
    # makes it work on single gpu string
    if not isinstance(gpus, list):
        gpus = [gpus]

    # Determine, which gpus are available
    gpu_availability = [
        True
        if (gpu.load < max_load or (include_nan and np.isnan(gpu.load)))
        and (
            gpu.memory_util < max_memory or (include_nan and np.isnan(gpu.memory_util))
        )
        and ((gpu.id not in exclude_id) and (gpu.uuid not in exclude_uuid))
        else False
        for gpu in gpus
    ]

    if len(gpus) == 1:
        return gpu_availability[0]

    return gpu_availability


def get_first_available(
    order="first",
    max_load=0.5,
    max_memory=0.5,
    attempts=1,
    interval=900,
    verbose=False,
    include_nan=False,
    exclude_id=[],
    exclude_uuid=[],
):
    for i in range(attempts):
        if verbose:
            print(
                "Attempting ("
                + str(i + 1)
                + "/"
                + str(attempts)
                + ") to locate available GPU."
            )
        # Get first available GPU
        available = get_available(
            order=order,
            limit=1,
            max_load=max_load,
            max_memory=max_memory,
            include_nan=include_nan,
            exclude_id=exclude_id,
            exclude_uuid=exclude_uuid,
        )
        # If an available GPU was found, break for loop.
        if available:
            if verbose:
                print("GPU " + str(available) + " located!")
            break
        # If this is not the last attempt, sleep for 'interval' seconds
        if i != attempts - 1:
            time.sleep(interval)

    # Check if an GPU was found, or if the attempts simply ran out. Throw error, if no GPU was found
    if not (available):
        raise RuntimeError(
            "Could not find an available GPU after "
            + str(attempts)
            + " attempts with "
            + str(interval)
            + " seconds interval."
        )

    # Return found GPU
    return available


def show_utilization(all=False, attr_list=None, use_old_code=False):
    gpus = get_gpus()
    if all:
        if use_old_code:
            print(
                " ID | Name | Serial | UUID || GPU util. | Memory util. || Memory total | Memory used | Memory free || Display mode | Display active |"
            )
            print(
                "------------------------------------------------------------------------------------------------------------------------------"
            )
            for gpu in gpus:
                print(
                    " {0:2d} | {1:s}  | {2:s} | {3:s} || {4:3.0f}% | {5:3.0f}% || {6:.0f}MB | {7:.0f}MB | {8:.0f}MB || {9:s} | {10:s}".format(
                        gpu.id,
                        gpu.name,
                        gpu.serial,
                        gpu.uuid,
                        gpu.load * 100,
                        gpu.memory_util * 100,
                        gpu.memory_total,
                        gpu.memory_used,
                        gpu.memory_free,
                        gpu.display_mode,
                        gpu.display_active,
                    )
                )
        else:
            attr_list = [
                [
                    {"attr": "id", "name": "ID"},
                    {"attr": "name", "name": "Name"},
                    {"attr": "serial", "name": "Serial"},
                    {"attr": "uuid", "name": "UUID"},
                ],
                [
                    {
                        "attr": "load",
                        "name": "GPU util.",
                        "suffix": "%",
                        "transform": lambda x: x * 100,
                        "precision": 0,
                    },
                    {
                        "attr": "memory_util",
                        "name": "Memory util.",
                        "suffix": "%",
                        "transform": lambda x: x * 100,
                        "precision": 0,
                    },
                ],
                [
                    {
                        "attr": "memory_total",
                        "name": "Memory total",
                        "suffix": "MB",
                        "precision": 0,
                    },
                    {
                        "attr": "memory_used",
                        "name": "Memory used",
                        "suffix": "MB",
                        "precision": 0,
                    },
                    {
                        "attr": "memory_free",
                        "name": "Memory free",
                        "suffix": "MB",
                        "precision": 0,
                    },
                ],
                [
                    {"attr": "display_mode", "name": "Display mode"},
                    {"attr": "display_active", "name": "Display active"},
                ],
            ]

    else:
        if use_old_code:
            print(" ID  GPU  MEM")
            print("--------------")
            for gpu in gpus:
                print(
                    " {0:2d} {1:3.0f}% {2:3.0f}%".format(
                        gpu.id, gpu.load * 100, gpu.memory_util * 100
                    )
                )
        else:
            attr_list = [
                [
                    {"attr": "id", "name": "ID"},
                    {
                        "attr": "load",
                        "name": "GPU",
                        "suffix": "%",
                        "transform": lambda x: x * 100,
                        "precision": 0,
                    },
                    {
                        "attr": "memory_util",
                        "name": "MEM",
                        "suffix": "%",
                        "transform": lambda x: x * 100,
                        "precision": 0,
                    },
                ]
            ]

    if not use_old_code:
        if attr_list is not None:
            header_string = ""
            gpustrings = [""] * len(gpus)
            for attr_group in attr_list:
                for attr_dict in attr_group:
                    header_string = header_string + "| " + attr_dict["name"] + " "
                    header_width = len(attr_dict["name"])
                    min_width = len(attr_dict["name"])

                    attr_precision = (
                        "." + str(attr_dict["precision"])
                        if ("precision" in attr_dict.keys())
                        else ""
                    )
                    attr_suffix = (
                        str(attr_dict["suffix"])
                        if ("suffix" in attr_dict.keys())
                        else ""
                    )
                    attr_transform = (
                        attr_dict["transform"]
                        if ("transform" in attr_dict.keys())
                        else lambda x: x
                    )
                    for gpu in gpus:
                        attr = getattr(gpu, attr_dict["attr"])

                        attr = attr_transform(attr)

                        if isinstance(attr, float):
                            attr_str = ("{0:" + attr_precision + "f}").format(attr)
                        elif isinstance(attr, np.int64):
                            attr_str = ("{0:d}").format(attr)
                        elif isinstance(attr, str):
                            attr_str = attr
                        elif isinstance(attr, np.int32):
                            attr_str = str(attr)
                        else:
                            attr_str = str(attr)

                        attr_str += attr_suffix

                        min_width = np.maximum(min_width, len(attr_str))

                    header_string += " " * np.maximum(0, min_width - header_width)

                    min_width_str = str(min_width - len(attr_suffix))

                    for gpu_idx, gpu in enumerate(gpus):
                        attr = getattr(gpu, attr_dict["attr"])

                        attr = attr_transform(attr)

                        if isinstance(attr, float):
                            attr_str = (
                                "{0:" + min_width_str + attr_precision + "f}"
                            ).format(attr)
                        elif isinstance(attr, np.int64):
                            attr_str = ("{0:" + min_width_str + "d}").format(attr)
                        elif isinstance(attr, str):
                            attr_str = ("{0:" + min_width_str + "s}").format(attr)
                        elif isinstance(attr, np.int32):
                            attr_str = str(attr)
                        else:
                            attr_str = str(attr)

                        attr_str += attr_suffix

                        gpustrings[gpu_idx] += "| " + attr_str + " "

                header_string = header_string + "|"
                for gpu_idx, gpu in enumerate(gpus):
                    gpustrings[gpu_idx] += "|"

            header_spacing_string = "-" * len(header_string)
            print(header_string)
            print(header_spacing_string)
            for gpustring in gpustrings:
                print(gpustring)


def describe():
    """prints quick summary informations about available gpus"""
    for gpu in get_gpus():
        gpu.describe()


def initialize_nvcuda(logger=None, warning_msg=None, cwd=CWD) -> None:
    """Initialize graphic driver components according to
    user operating system and gpu environment

    사용자의 PC에 GPU 및 NVIDIA Driver가 설치되어 있는지 여부에 따라
    배포된 Tensorflow 기반 Engine의 구성요소가 변경되어야 하기에 필요한 단락
    """
    for dllname in CUDA_DLLNAMES:
        dllpaths = find_executable(f"{dllname}.dll")
        if not dllpaths:
            try:
                # os.rename equivalent function
                # for cross-platform overwriting of the destination.
                os.replace(f"{cwd}\\{dllname}_bak.dll", f"{cwd}\\{dllname}.dll")
            except FileNotFoundError:
                if logger:
                    logger.warning(warning_msg)

        else:
            for dllpath in dllpaths:
                if "system32" in dllpath:
                    try:
                        os.replace(f"{cwd}\\{dllname}.dll", f"{cwd}\\{dllname}_bak.dll")
                    except FileNotFoundError:
                        pass
