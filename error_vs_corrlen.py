import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

import models as myModels
import auxillary as aux

n_sites = 12
point_num = 20 # number of random hamiltonians
trial_num = 5 # number of trials for each random hamiltonian
iter_num = 1500
key = jax.random.PRNGKey(11111)
key, *subkeys = jax.random.split(key, 200) # 200 should be larger than poin_num
data_list = []

graph  = nk.graph.Chain(n_sites, pbc=True)
#print("graph generated!")

hilbert = nk.hilbert.Spin(s=0.5, N=n_sites)
#print("hilbert space generated!")

s0, sx, sy, sz = [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]
s = [s0, sx, sy, sz]

for idx0 in range(point_num):
    "this loop loops over different random hamiltonians"
    
    print("system ", idx0)
    #idx0 = 34
    
    subkeys[idx0], *subsubkeys = jax.random.split(subkeys[idx0], 200) # 200 should be larger than trial_num
    "generate enough subsubkeys for later parameters initialization"

    print("key for random hamiltonian: ", subsubkeys[0])
    J = jax.random.uniform(key=subsubkeys[0], shape=(4,4))-0.5
    J = (J+J.T)/2.0
    bond = 0
    for idx1 in range(4):
        for idx2 in range(4):
            bond += J[idx1, idx2]*np.kron(s[idx1], s[idx2])

    bond_ops = [bond, ]
    site_ops = [-1*np.array(sx)]
    hamiltonian = nk.operator.LocalOperator(hilbert, dtype=complex)
    for i in range(n_sites):
        j = (i+1)%n_sites
        hamiltonian += nk.operator.LocalOperator(hilbert, bond, [i, j])
    #print("Hamiltonian generated!")

    exact_results = aux.Exact(hamiltonian)
    exact_gs_energy = exact_results.gs_energy
    corrlen = exact_results.CorrLen()
    long_corr = exact_results.Long_Range_Corr()
    print("\n")
    print('exact ground-state energy: ',exact_gs_energy)
    print("corrlen: ", corrlen)
    print("long range correlation: ", long_corr)
    print("\n")

    model = myModels.mySymmRBM(alpha=1, filter_len=n_sites)
    #model = myModels.myRBM(alpha = 2)
    #model = nk.models.RBM(alpha = 2)
    #symmetries = graph.translations()
    #model = nk.models.RBMSymm(symmetries=symmetries, alpha=1, dtype=complex)
    #print("model generated!")

    #sampler = nk.sampler.MetropolisExchange(hilbert=hilbert, n_chains=20, graph=graph)
    sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=20)
    #sampler = nk.sampler.MetropolisHamiltonian(hilbert=hilbert, hamiltonian=hamiltonian, n_chains=20)

    #op = nk.optimizer.Sgd(learning_rate=0.001)
    #op = nk.optimizer.Adam(learning_rate=0.01)
    #op = nk.optimizer.AdaGrad(learning_rate=0.01)
    #op = nk.optimizer.Momentum(learning_rate=0.01)
    #sr = nk.optimizer.SR(diag_shift=0.1)

    trial_record_list = []
    for idx1 in range(trial_num):
        "this loop tries different variational state initialization seeds"

        idx_bias = 0
        vs = nk.variational.MCState(
            sampler, model, n_samples=500, sampler_seed=int(subsubkeys[idx1+1+idx_bias][0]), seed=int(subsubkeys[idx1+1+idx_bias][1]))

        training = aux.Training(variational_state=vs, hamiltonian=hamiltonian)

        print("trial ", idx1)
        print("sampler seed: ", int(subsubkeys[idx1+1+idx_bias][0]), "seed for parameters: ", int(subsubkeys[idx1+1+idx_bias][1]))
        start = time.time()
        #training.Run0(iter_num=iter_num, lr=0.002)
        training.Run1(iter_num=iter_num, lr_list=[[0, 0.002]], sample_num_list=[[0, 500]])
        end = time.time()
        print('The RBM calculation took',end-start,'seconds')
        #approx_gs_energy = np.real(vs.expect(hamiltonian).__dict__["mean"])
        approx_gs_energy = np.average(training.log[iter_num-20:])
        print("vstate expectation energy:", approx_gs_energy.round(6))
        error = np.abs((approx_gs_energy-exact_gs_energy)/exact_gs_energy)[0]
        trial_record_list.append([idx1+idx_bias, error])
        print("error: ", error)
        print("#######################################")

    print("error list: ", trial_record_list)
    min_error = min(trial_record_list[i][1] for i in range(len(trial_record_list)))
    data_list.append([corrlen, error, min_error])
    np.save("data_1.npy", np.array(data_list))
    print("**************************************************************************************************************************")

data_list=np.array(data_list)
np.save("data_12.npy", data_list)


