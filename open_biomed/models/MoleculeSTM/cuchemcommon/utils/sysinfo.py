#!/opt/conda/envs/rapids/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter

import psutil
import pynvml as nv


def get_machine_config():
    """Get machine config for CPU and GPU(s)"""

    # CPU config
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    cpufreq = psutil.cpu_freq()
    cpufreq_max = cpufreq.max  # Mhz
    cpufreq_min = cpufreq.min
    cpufreq_cur = cpufreq.current

    svmem = psutil.virtual_memory()
    mem_total = svmem.total / (1024.0 ** 3)  # GB
    mem_avail = svmem.available / (1024.0 ** 3)

    # GPU config
    nv.nvmlInit()
    driver_version = nv.nvmlSystemGetDriverVersion()
    deviceCount = nv.nvmlDeviceGetCount()
    gpu_devices, gpu_mems = [], []
    for i in range(deviceCount):
        handle = nv.nvmlDeviceGetHandleByIndex(i)
        gpu_devices.append(nv.nvmlDeviceGetName(handle).decode("utf-8"))
        gpu_mem = nv.nvmlDeviceGetMemoryInfo(handle).total / (1024.0 ** 3)
        gpu_mems.append(gpu_mem)

    return {'cpu': {'physical_cores': physical_cores, 'logical_cores': logical_cores,
                    'min_freq_MHz': cpufreq_min, 'max_freq_MHz': cpufreq_max, 'cur_freq_MHz': cpufreq_cur,
                    'total_mem_GB': mem_total, 'avail_mem_GB': mem_avail},
            'gpu': {'devices': gpu_devices, 'mem_GB': gpu_mems}}


def print_machine_config(config):
    """Printable version of config"""
    cpu_cores = config['cpu']['physical_cores']
    cpu_freq = int(round(config['cpu']['max_freq_MHz'], 0))
    ram = int(round(config['cpu']['total_mem_GB'], 0))
    cpu_config_message = f'{cpu_freq} MHz CPU with {cpu_cores} cores, {ram} GB RAM'

    gpu_devices = Counter([(x, int(round(y, 0))) for x, y in zip(config['gpu']['devices'], config['gpu']['mem_GB'])])
    gpu_config_message = ''
    for (handle, mem), count in gpu_devices.items():
        gpu_config_message += f'{count} x {handle} GPU(s)'

    return ', '.join([cpu_config_message, gpu_config_message])
