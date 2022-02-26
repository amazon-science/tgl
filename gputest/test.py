import torch as th
import numpy as np
import time

# The number of GPUs we'll copy data to simultaneously.
num_gpus = 8
# The size of the array that we copy to GPUs.
arr_size = 1000000

arr = th.randn(arr_size, 10)
arr = arr.pin_memory()
devs = []
gpu_input_arrs = []
gpu_output_arrs = []
pinned_input_arrs = []
pinned_output_arrs = []
input_streams = []
output_streams = []
for i in range(num_gpus):
    dev = th.device('cuda:' + str(i))
    devs.append(dev)
    input_streams.append(th.cuda.Stream(device=dev))
    output_streams.append(th.cuda.Stream(device=dev))
    gpu_input_arrs.append(th.zeros(arr_size, 10, device=dev))
    gpu_output_arrs.append(th.randn(arr_size, 10, device=dev))
    pinned_input_arrs.append(th.randn(arr_size, 10, pin_memory=True))
    pinned_output_arrs.append(th.zeros(arr_size, 10, pin_memory=True))
cpu_dev = th.device('cpu')
th.cuda.synchronize()

# Measure the throughput of copying data from CPU pinned memory to GPUs.
for i in range(10):
    start = time.time()
    res = []
    for d, _ in enumerate(devs):
        with th.cuda.stream(input_streams[d]):
            gpu_input_arrs[d].copy_(pinned_input_arrs[d], non_blocking=True)
    th.cuda.synchronize()
    seconds = time.time() - start
    print('to {} GPUs: {:.5f}s, {:.3f}'.format(len(devs), seconds, np.prod(arr.shape) * 4 / seconds / 1000000000 * len(devs)))

# # Measure the throughput of copying data from GPU to CPU pinned memory.
# for i in range(10):
#     start = time.time()
#     res = []
#     for d, _ in enumerate(devs):
#         with th.cuda.stream(output_streams[d]):
#             pinned_output_arrs[d].copy_(gpu_output_arrs[d], non_blocking=True)
#     th.cuda.synchronize()
#     seconds = time.time() - start
#     print('from {} GPUs: {:.3f}'.format(len(devs), np.prod(arr.shape) * 4 / seconds / 1000000000 * len(devs)))

# # Measure the throughput of copying data in both directions simultaneously.
# for i in range(10):
#     start = time.time()
#     res = []
#     for d, _ in enumerate(devs):
#         with th.cuda.stream(input_streams[d]):
#             gpu_input_arrs[d].copy_(pinned_input_arrs[d], non_blocking=True)
#     for d, _ in enumerate(devs):
#         with th.cuda.stream(output_streams[d]):
#             pinned_output_arrs[d].copy_(gpu_output_arrs[d], non_blocking=True)
#     th.cuda.synchronize()
#     seconds = time.time() - start
#     print('from+to {} GPUs: {:.3f}'.format(len(devs), np.prod(arr.shape) * 4 * 2 / seconds / 1000000000 * len(devs)))