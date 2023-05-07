import subprocess
import time
import re

def get_gpu_stats(sampling_period, num_samples):
    # Get the number of GPUs on the system
    output = subprocess.check_output(['nvidia-smi', '--list-gpus'])
    gpu_count = len(output.decode().strip().split('\n'))
    
    gpu_usage_dict = {}
    gpu_mem_dict = {}
    
    # Check GPU usage and memory every sampling_period seconds for num_samples iterations
    for i in range(num_samples):
        gpu_usage_list = []
        gpu_mem_list = []
        for j in range(gpu_count):
            # Run nvidia-smi command to get GPU usage and memory for the current GPU
            output = subprocess.check_output(['nvidia-smi', '--id={}'.format(j), '--format=csv,noheader,nounits', '--query-gpu=utilization.gpu,memory.used,memory.total'])
            output_lines = output.decode().strip().split('\n')
            gpu_usage_list.append(int(output_lines[0].rstrip('%')))
            mem_used = int(re.sub('[^0-9]', '', output_lines[1]))
            mem_total = int(re.sub('[^0-9]', '', output_lines[2]))
            gpu_mem_list.append(mem_total - mem_used)
        gpu_usage_dict[f"cuda:{j}"] = gpu_usage_list
        gpu_mem_dict[f"cuda:{j}"] = gpu_mem_list
        time.sleep(sampling_period)
    
    return gpu_usage_dict, gpu_mem_dict

def find_idle_gpu(sampling_period, num_samples, usage_threshold, mem_threshold):
    gpu_usage, gpu_mem = get_gpu_stats(sampling_period, num_samples)
    for gpu_id, usage_list in gpu_usage.items():
        if max(usage_list) < usage_threshold and min(gpu_mem[gpu_id]) > mem_threshold:
            return gpu_id
    return None
