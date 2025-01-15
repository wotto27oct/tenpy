from tenpy.models.spins import SpinChain
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification import PurificationTEBD, PurificationApplyMPO

import time


def imag_tebd(model_params, beta_max=4., dt=0.05, order=2, bc="finite"):
    M = SpinChain(model_params)
    psi = PurificationMPS.from_infiniteT(M.lat.mps_sites(), bc=bc)
    options = {
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-8
        },
        'order': order,
        'dt': dt,
        'N_steps': 1
    }
    beta = 0.
    eng = PurificationTEBD(psi, M, options)
    Zs = [2**model_params["L"]]
    betas = [0.]
    while beta < beta_max:
        beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
        betas.append(beta)
        eng.run_imaginary(dt)  # cool down by dt
        Zs.append(psi.norm * 2**model_params["L"])
    return {'beta': betas, 'Z': Zs}

import numpy as np
from scipy.sparse import kron, identity, csr_matrix


def heisenberg_hamiltonian(L, J=1.0):
    """
    Open Boundary Condition の Heisenberg ハミルトニアンを生成。
    
    Args:
        L (int): スピンの数。
        J (float): 交換相互作用の強さ。
    
    Returns:
        csr_matrix: L個のスピンからなる Heisenberg モデルのハミルトニアン (スパース行列形式)。
    """
    # スピン演算子
    Sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    Sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    Sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    I = np.eye(2, dtype=complex)

    # ハミルトニアン初期化 (スパース行列形式)
    H = csr_matrix((2**L, 2**L), dtype=complex)

    # 隣接するスピンペアに対する相互作用を加える
    for i in range(L - 1):  # OBCなので L-1 まで
        # 左右の位置に演算子を挿入
        Sx_i = kron(identity(2**i), kron(Sx, identity(2**(L-i-1))))
        Sx_j = kron(identity(2**(i+1)), kron(Sx, identity(2**(L-i-2))))
        
        Sy_i = kron(identity(2**i), kron(Sy, identity(2**(L-i-1))))
        Sy_j = kron(identity(2**(i+1)), kron(Sy, identity(2**(L-i-2))))
        
        Sz_i = kron(identity(2**i), kron(Sz, identity(2**(L-i-1))))
        Sz_j = kron(identity(2**(i+1)), kron(Sz, identity(2**(L-i-2))))
        
        # ハミルトニアンに相互作用を加える
        H += J * (Sx_i @ Sx_j + Sy_i @ Sy_j + Sz_i @ Sz_j)

    return H


def imag_apply_mpo(model_params, beta_max=4., dt=0.05):
    H = heisenberg_hamiltonian(model_params['L'], model_params['Jz'])

    start = time.time()
    print("diagonalizing...")
    eigenvalues = np.linalg.eigvalsh(H.toarray())
    end = time.time()
    print(f"elapsed time for diagonalization: {end - start}[s]")

    beta = 0.
    betas = [0.]
    Zs = [2**L]
    while beta < beta_max:
        beta += 2. * dt  # factor of 2:  |psi> ~= exp^{- dt H}, but rho = |psi><psi|
        betas.append(beta)
        Z = np.sum(np.exp(-beta * eigenvalues))
        Zs.append(Z)

    return {'beta': betas, 'Z': Zs}


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    L = 10
    chi = 100
    dt = 0.05

    L_diag_upper_bound = 10

    model_params = {
        'L': L,
        'S': 0.5,
        'conserve': 'Sz',
        'Jz': 1.0,
        'Jy': 1.0,
        'Jx': 1.0,
        'hx': 0.0,
        'hy': 0.0,
        'hz': 0.0,
        'muJ': 0.0,
        'bc_MPS': 'finite',
    }

    if L <= L_diag_upper_bound:
        data_diag = imag_apply_mpo(model_params, dt=dt) # full diagonalization

    data_tebd = imag_tebd(model_params, dt=dt) # TEBD using MPS

    import numpy as np
    import matplotlib.pyplot as plt

    if L <= L_diag_upper_bound:
        plt.plot(data_diag['beta'], np.log(data_diag['Z']), label='diag')
    plt.plot(data_tebd['beta'], np.log(data_tebd['Z']), label='TEBD', linestyle="dashdot")
    plt.legend()
    plt.title(f'L={L}, chi={chi}, dt={dt}')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'log(Z)')
    plt.show()