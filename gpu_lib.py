import math
import numpy as np
import multiprocessing
from numba import cuda, float32
import imageio
from queue import Queue as SimpleQueue
import sys

def reconstruct_multiple_svd(filenames, n_processes, max_items_on_gpu=3, min_items_on_gpu=1, verbose=True):
    # cannot fork process since we need a fresh CUDA context
    ctx = multiprocessing.get_context('spawn')

    manager = multiprocessing.Manager()
    queue = manager.Queue()

    processes = []
    for _ in range(n_processes):
        process = ctx.Process(target=run_process, args=[queue, max_items_on_gpu, min_items_on_gpu, verbose])
        process.start()
        processes.append(process)

    for filename in filenames:
        queue.put(filename)

    for _ in range(n_processes):
        queue.put(None)

    for process in processes:
        process.join()


def run_process(queue, max_items_on_gpu, min_items_on_gpu, verbose):
    name = multiprocessing.current_process().name
    print(f'[{name}] START')

    # create new CUDA stream for this process
    stream = cuda.stream()

    event_finished = cuda.event(timing=False)

    # number of items pushed to the CUDA stream
    n_scheduled = 0

    # contains tuples (original, reco) of items to be verified
    to_check_queue = SimpleQueue()

    while True:
        filename = queue.get()
        if filename is None:
            print(f'[{name}] no more items in queue')
            break

        # apply SVD to get U, S, V^T
        img = imageio.imread(filename)
        img = img - img.min() / img.max() - img.min() # normalize data
        u, s, vt = np.linalg.svd(img, full_matrices=False)

        #  calculate dimensions for kernel
        threads_per_block = 16
        grid_y_max = max(u.shape[0], vt.shape[0])
        grid_x_max = max(u.shape[1], vt.shape[1])
        n_blocks_x = math.ceil(grid_x_max / threads_per_block)
        n_blocks_y = math.ceil(grid_y_max / threads_per_block)

        with cuda.pinned(u, s, vt):
            # add "host to device" to stream
            reco_d = cuda.device_array((u.shape[0], vt.shape[1]), dtype=np.float32, stream=stream)
            u_d = cuda.to_device(u, stream=stream)
            s_d = cuda.to_device(s, stream=stream)
            vt_d = cuda.to_device(vt, stream=stream)

            # add kernel to stream
            _reconstruct_on_kernel[
                (n_blocks_x, n_blocks_y), (threads_per_block, threads_per_block), stream
            ](u_d, s_d, vt_d, reco_d)

            # add "device to host" to stream
            reco_h = reco_d.copy_to_host(stream=stream)

            to_check_queue.put((img, reco_h))

            # add event to get notified once this item has been reconstructed
            event_finished.record(stream=stream)
            event_finished.wait(stream=stream)

            n_scheduled += 1
            if n_scheduled == max_items_on_gpu:
                # waiting until we fall under min_items_on_gpu to add more items
                while n_scheduled > min_items_on_gpu:
                    event_finished.synchronize()
                    n_scheduled -= 1

                    # verify latest result
                    original, reco = to_check_queue.get()
                    np.testing.assert_array_almost_equal(original, reco, decimal=3)
                    if verbose:
                        print(f'[{name}] Checked reconstruction')

    # wait until all done
    while n_scheduled > 0:
        event_finished.synchronize()
        n_scheduled -= 1

        # verify latest result
        original, reco = to_check_queue.get()
        np.testing.assert_array_almost_equal(original, reco, decimal=3)
        if verbose:
            print(f'[{name}] Checked reconstruction')

    print(f'[{name}] DONE')
    sys.stdout.flush()


@cuda.jit
def _reconstruct_on_kernel(u, s, vt, reco):
    # each thread calculates the sum of products for a specific index (x, y)

    x, y = cuda.grid(2)
    local_x = cuda.threadIdx.x
    local_y = cuda.threadIdx.y
    threads_per_block = 16 # cuda.blockDim.x but must be constant
    blocks_per_grid = cuda.gridDim.x

    shrd_u = cuda.shared.array(shape=(threads_per_block, threads_per_block), dtype=float32)
    shrd_s = cuda.shared.array(shape=(threads_per_block,), dtype=float32)
    shrd_vt = cuda.shared.array(shape=(threads_per_block, threads_per_block), dtype=float32)
    shrd_u_pre = cuda.shared.array(shape=(threads_per_block, threads_per_block), dtype=float32)
    shrd_s_pre = cuda.shared.array(shape=(threads_per_block,), dtype=float32)
    shrd_vt_pre = cuda.shared.array(shape=(threads_per_block, threads_per_block), dtype=float32)

    shrd_u_pre[local_y, local_x] = 0
    if y < u.shape[0] and local_x < u.shape[1]:
        shrd_u_pre[local_y, local_x] = u[y, local_x]

    if local_y == 0:
        shrd_s_pre[local_x] = 0
        if local_x < s.shape[0]:
            # only first row in block loads shrd_s since it's a one-dimensional array
            shrd_s_pre[local_x] = s[local_x]

    shrd_vt_pre[local_y, local_x] = 0
    if x < vt.shape[1] and local_y < vt.shape[0]:
        shrd_vt_pre[local_y, local_x] = vt[local_y, x]

    sum_of_products = float32(0.)
    for block in range(blocks_per_grid):
        # calculate sum of products per block
        # the block is only moved to get the other values in the matrices u, s and vt
        # the index for which the sum of product is calculated remains the same

        # swap shared memory blocks
        shrd_u, shrd_u_pre = shrd_u_pre, shrd_u
        shrd_s, shrd_s_pre = shrd_s_pre, shrd_s
        shrd_vt, shrd_vt_pre = shrd_vt_pre, shrd_vt

        next_block = block + 1

        # wait until all threads have computed their sum of products before we move
        # to the next block as the shared values will be overridden with the next
        # iteration
        cuda.syncthreads()

        if next_block < blocks_per_grid:
            # load next block
            shrd_u_pre[local_y, local_x] = 0
            if y < u.shape[0] and (next_block * threads_per_block + local_x) < u.shape[1]:
                shrd_u_pre[local_y, local_x] = u[y, next_block * threads_per_block + local_x]

            if local_y == 0:
                shrd_s_pre[local_x] = 0
                if (next_block * threads_per_block + local_x) < s.shape[0]:
                    # only first row in block loads shrd_s since it's a one-dimensional array
                    shrd_s_pre[local_x] = s[next_block * threads_per_block + local_x]

            shrd_vt_pre[local_y, local_x] = 0
            if x < vt.shape[1] and (next_block * threads_per_block + local_y) < vt.shape[0]:
                shrd_vt_pre[local_y, local_x] = vt[next_block * threads_per_block + local_y, x]

        # start calculating the sum of products for index (y, x) and the current block
        for i in range(threads_per_block):
            # no checking of boundaries necessary since the warp executes the
            # statement anyway and threads out of bound would have to wait anyway

            # read values from shred_u in transposed order
            sum_of_products += shrd_u[local_y, i] * shrd_s[i] * shrd_vt[i, local_x]

    if y < reco.shape[0] and x < reco.shape[1]:
        reco[y, x] = sum_of_products
