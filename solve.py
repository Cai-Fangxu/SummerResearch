from netket import exact
import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import time

import models as myModels
import auxillary as aux

n_sites = 8

graph  = nk.graph.Chain(n_sites, pbc=True)
#print("graph generated!")

hilbert = nk.hilbert.Spin(s=0.5, N=n_sites)
#print("hilbert space generated!")

s0, sx, sy, sz = [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]
s = [s0, sx, sy, sz]
J = jax.random.uniform(key=jax.random.PRNGKey(238), shape=(4,4))-0.5
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

model = myModels.mySymmRBM(alpha=2, filter_len=8)
#model = myModels.myRBM(alpha = 2)
#model = nk.models.RBM(alpha = 2)
#symmetries = graph.translations()
#model = nk.models.RBMSymm(symmetries=symmetries, alpha=2)
#print("model generated!")

#sampler = nk.sampler.MetropolisExchange(hilbert=hilbert, n_chains=20, graph=graph)
sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=20)
#sampler = nk.sampler.MetropolisHamiltonian(hilbert=hilbert, hamiltonian=hamiltonian, n_chains=10)

op = nk.optimizer.Sgd(learning_rate=0.01)
#op = nk.optimizer.Adam(learning_rate=0.01)
#op = nk.optimizer.Momentum(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.1)
vs = nk.variational.MCState(sampler, model, n_samples=500, sampler_seed=1141, seed=2308)
#print("variational state generated!")

gs = nk.VMC(
    hamiltonian=hamiltonian,
    optimizer=op,
    preconditioner=sr,
    variational_state=vs)

start = time.time()
gs.run(400, out='RBM2')
end = time.time()
print('The RBM calculation took',end-start,'seconds')

exact_results = aux.Exact(hamiltonian)
exact_gs_energy = exact_results.gs_energy
approx_gs_energy = jnp.real(vs.expect(hamiltonian).__dict__["mean"])
error = jnp.abs((approx_gs_energy-exact_gs_energy)/exact_gs_energy)
print('The exact ground-state energy is E0=',exact_gs_energy)
print('vstate expectation energy:', approx_gs_energy)
print('error is:', error)

print("*****************************************************************************************")
