import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import json

import models as myModels
import auxiliary as aux

n_sites = 12
point_num = 1 # number of random hamiltonians
trial_num = 1 # number of trials for each random hamiltonian
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
    idx0 = 10
    
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
    

    #sampler = nk.sampler.MetropolisExchange(hilbert=hilbert, n_chains_per_rank=20, graph=graph)
    sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains_per_rank=20)
    #sampler = nk.sampler.MetropolisHamiltonian(hilbert=hilbert, hamiltonian=hamiltonian, n_chains_per_rank=20)

    op = nk.optimizer.Sgd(learning_rate=0.002)
    #op = nk.optimizer.Adam(learning_rate=0.01)
    #op = nk.optimizer.AdaGrad(learning_rate=0.01)
    #op = nk.optimizer.Momentum(learning_rate=0.01)
    sr = nk.optimizer.SR(diag_shift=0.01, iterative=True)

    trial_record_list = []
    for idx1 in range(trial_num):
        "this loop tries different variational state initialization seeds"

        idx_bias = 0
        vs = nk.vqs.MCState(
            sampler, model, n_samples=500, sampler_seed=int(subsubkeys[idx1+1+idx_bias][0]), seed=int(subsubkeys[idx1+1+idx_bias][1]))

        "mean field initialization"
        mean_field = aux.MeanField_Init(J, variational_state=vs, n_sites=n_sites)
        vs.parameters = mean_field.Param_Init(noise=0.02, noise_key=subsubkeys[idx_bias])
        print("\n", "mean field energy:", mean_field.mean_field_energy)

        # training = aux.Training(variational_state=vs, hamiltonian=hamiltonian) # to do self-defined training (without SR), uncomment this line
        gs = nk.VMC(hamiltonian=hamiltonian, optimizer=op, variational_state=vs, preconditoner=sr, sr=sr)

        print("trial ", idx1)
        print("sampler seed: ", int(subsubkeys[idx1+1+idx_bias][0]), "seed for parameters: ", int(subsubkeys[idx1+1+idx_bias][1]))

        start = time.time()
        # training.Run1(iter_num=iter_num, lr_list=[[0, 0.002]], sample_num_list=[[0, 500]]) # to do self-defined training (without SR), uncomment this line
        gs.run(iter_num, out='SymmRBM_12')
        end = time.time()
        print('The RBM calculation took',end-start,'seconds')

        data = json.load(open("SymmRBM_12.log"))
        energy_list = data["Energy"]["Mean"]["real"]
        approx_gs_energy = np.average(energy_list[iter_num-20:])
        # approx_gs_energy = np.average(training.log[iter_num-20:]) # to do self-defined training (without SR), uncomment this line
        print("vstate expectation energy:", approx_gs_energy.round(6))
        error = np.abs((approx_gs_energy-exact_gs_energy)/exact_gs_energy)[0]
        trial_record_list.append([idx1+idx_bias, error])
        print("error: ", error)
        print("#######################################")

    print("error list: ", trial_record_list)
    min_error = min(trial_record_list[i][1] for i in range(len(trial_record_list)))
    data_list.append([corrlen, error, min_error])
    # np.save("data_test.npy", np.array(data_list))
    print("**************************************************************************************************************************")

data_list=np.array(data_list)
np.save("data_test.npy", data_list)
