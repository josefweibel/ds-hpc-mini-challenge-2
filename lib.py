import numpy as np
from multiprocessing import current_process, Process, Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager


def run_process(tasks, m, n, k, u_sm_name, s_sm_name, vt_sm_name, reco_sm_name):
    name = current_process().name
    print(f'[{name}] START')

    u_sm = SharedMemory(u_sm_name)
    u = np.ndarray(shape=(m, k), dtype=np.float64, buffer=u_sm.buf)
    s_sm = SharedMemory(s_sm_name)
    s = np.ndarray(shape=(k,), dtype=np.float64, buffer=s_sm.buf)
    vt_sm = SharedMemory(vt_sm_name)
    vt = np.ndarray(shape=(k, n), dtype=np.float64, buffer=vt_sm.buf)
    reco_sm = SharedMemory(reco_sm_name)
    reco = np.ndarray(shape=(m, n), dtype=np.float64, buffer=reco_sm.buf)

    while True:
        task = tasks.get()
        if task is None:
            print(f'[{name}] DONE')
            break

        i, j = task
        reco[i, j] = np.sum(u[i, :] * s * vt[:, j])


def reconstruct_svd_processes(u,s,vt,k):
    """SVD reconstruction for k components using processes

    Inputs:
    u: (m,n) numpy array
    s: (n) numpy array (diagonal matrix)
    vt: (n,n) numpy array
    k: number of reconstructed singular components

    Ouput:
    (m,n) numpy array U_mk * S_k * V^T_nk for k reconstructed components
    """

    tasks = Queue()

    with SharedMemoryManager() as smm:
        u_sm = smm.SharedMemory(np.dtype(np.float64).itemsize * u.shape[0] * k)
        shared_u = np.ndarray(shape=(u.shape[0], k), dtype=np.float64, buffer=u_sm.buf)
        np.copyto(shared_u, u[:, 0:k])

        s_sm = smm.SharedMemory(np.dtype(np.float64).itemsize * k)
        shared_s = np.ndarray(shape=(k,), dtype=np.float64, buffer=s_sm.buf)
        np.copyto(shared_s, s[0:k])

        vt_sm = smm.SharedMemory(np.dtype(np.float64).itemsize * k * vt.shape[1])
        shared_vt = np.ndarray(shape=(k, vt.shape[1]), dtype=np.float64, buffer=vt_sm.buf)
        np.copyto(shared_vt, vt[0:k, :])

        reco_sm = smm.SharedMemory(np.dtype(np.float64).itemsize * u.shape[0] * vt.shape[0])
        shared_reco = np.ndarray(shape=(u.shape[0], vt.shape[0]), dtype=np.float64, buffer=reco_sm.buf)

        processes = []
        n_proc = 4
        for i in range(n_proc):
            process = Process(target=run_process, args=[
                tasks, u.shape[0], vt.shape[0], k, u_sm.name, s_sm.name, vt_sm.name, reco_sm.name
            ])
            process.start()
            processes.append(process)

        for m in range(u.shape[0]):
            for n in range(vt.shape[0]):
                tasks.put((m, n))

        for i in range(n_proc):
            tasks.put(None)

        for process in processes:
            process.join()

        reco = np.ndarray(shared_reco.shape, dtype=np.float64)
        np.copyto(reco, shared_reco)

    return reco
