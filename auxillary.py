import netket as nk
from netket.variational.mc_state import MCState
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from numpy.lib.shape_base import _kron_dispatcher

class Exact():

    def __init__(self, hamiltonian: nk.operator.LocalOperator):
        self.ha = hamiltonian
        self.exact_diag = nk.exact.lanczos_ed(self.ha, compute_eigenvectors=True)
        self.gs_energy = self.exact_diag[0]
        self.gs = self.exact_diag[1].reshape(-1)
        self.hi = self.ha.hilbert
        self.n_sites = self.hi.size

    def Expect(self, op: nk.operator.LocalOperator, state:np.ndarray=[]):

        if state == []:
            state = self.gs
        vec1 = op.apply(state)
        expect = np.dot(state.conjugate(), vec1)
        return expect  

    def Correlation(self, n=0, direction=1):

        n = n%self.n_sites
        
        if direction == 1:
            sigma = nk.operator.spin.sigmax
        elif direction == 2:
            sigma = nk.operator.spin.sigmay
        else:
            sigma = nk.operator.spin.sigmaz

        return self.Expect(sigma(self.hi, 0)*sigma(self.hi, n)) - self.Expect(sigma(self.hi, 0))*self.Expect(sigma(self.hi, n))
    
    def CorrLen(self, direction=0):

        def distance(n):
            return min(n, abs(n-self.n_sites))
        
        if direction in [1, 2, 3]:
            tmp1 = sum(np.abs(self.Correlation(n, direction))*distance(n) for n in range(self.n_sites))
            tmp2 = sum(np.abs(self.Correlation(n, direction)) for n in range(self.n_sites))
            return 1.0*tmp1/tmp2
        else:
            return np.max([self.CorrLen(direction) for direction in range(1,4)])

    def Long_Range_Corr(self, ):
        
        if self.n_sites%2 ==0:
            search_list = [self.n_sites/2-1, self.n_sites/2, self.n_sites/2+1]
        else:
            search_list = [(self.n_sites-1)/2, (self.n_sites+1)/2]

        long_corr = np.max(np.abs(
            [self.Correlation(n, direction) for direction in [1, 2, 3] for n in search_list]))

        return long_corr

    def GSMomentum(self):

        state_translated = self.gs.copy()

        for idx1 in range(2**self.n_sites):
         if idx1 & 1 == 0:
            idx2 = idx1 >> 1
        else:
            idx2 = (idx1 >> 1) + 2**(self.n_sites-1)
            state_translated[idx2]=self.gs[idx1]

        phase = np.dot(self.gs.conjugate(), state_translated)

        return -1j*np.log(phase)

    def IsGSDegenerate(self, precision=1.0e-5):
        energy = nk.exact.lanczos_ed(self.ha, k=2, compute_eigenvectors=False)
        if np.abs(energy[0]-energy[1]) < precision:
            return True
        else:
            return False

    def Show_Basis_Coeff(self, repeat=True, precision=1.0e-5, take_abs=False):
        tmp_array = np.array(self.gs)
        if take_abs is True:
            tmp_array = np.abs(tmp_array)
        tmp_array = [["{:b}".format(i).rjust(self.n_sites, "0"), tmp_array[i]] for i in range(len(tmp_array))]
        tmp_array = sorted(tmp_array, key=lambda x: -abs(x[1]))
        if repeat == True:
            return tmp_array
        else:
            idx = 1
            while idx < len(tmp_array):
                if abs(tmp_array[idx][1]-tmp_array[idx-1][1]) < precision:
                    del(tmp_array[idx])
                else: idx += 1
            return tmp_array


class Training():

    def __init__(self, variational_state: nk.variational.MCState, hamiltonian: nk.operator.LocalOperator):
        self.vs = variational_state
        self.ha = hamiltonian
        self.log = []

    def Run0(self, iter_num: int, lr=0.002):
        "just normal SGD optimization"

        pbar = tqdm(range(iter_num))

        for iter in pbar:
            self.vs.reset()
            energy, gradient = self.vs.expect_and_grad(self.ha)
            new_pars = jax.tree_multimap(lambda pars, grad: pars-lr*grad, self.vs.parameters, gradient)
            self.vs.parameters = new_pars
            self.log.append(energy.mean)
            pbar.set_postfix({"energy":energy.mean.round(4)})

    def Run1(self, iter_num: int, lr_list=[[0, 0.002]], sample_num_list=[[0, 500]]):
        "use user-defined learning rate sequence [[iter1, lr1], [iter2, lr2], ...]"
        "use user-defined sample_number sequence [[iter1, num1], [iter2, num2], ...]"

        pbar = tqdm(range(iter_num))
        lr_idx, sample_num_idx = 0, 0
        lr, sample_num = lr_list[0][1], sample_num_list[0][1]
        for iter in pbar:

            if iter == lr_list[lr_idx][0]:
                lr = lr_list[lr_idx][1]
                if lr_idx < len(lr_list)-1:
                    lr_idx += 1
            
            if iter == sample_num_list[sample_num_idx][0]:
                sample_num = sample_num_list[sample_num_idx][1]
                self.vs.n_samples = sample_num
                if sample_num_idx < len(sample_num_list)-1:
                    sample_num_idx += 1

            self.vs.reset()
            energy, gradient = self.vs.expect_and_grad(self.ha)
            new_pars = jax.tree_multimap(lambda pars, grad: pars-lr*grad, self.vs.parameters, gradient)
            self.vs.parameters = new_pars
            self.log.append(energy.mean)
            pbar.set_postfix({"energy":energy.mean.round(4)})   
        

class PostProcessing():

    def __init__(self, vs:nk.variational.MCState):
        self.vs = vs
        self.hi = vs.hilbert
        self.n_sites = self.hi.size

    def Show_Basis_Coeff(self, repeat=True, precision=1.0e-5, take_abs=False):
        tmp_array=np.array(self.vs.to_array())
        if take_abs is True:
            tmp_array = np.abs(tmp_array)
        tmp_array=[["{:b}".format(i).rjust(self.n_sites, "0"), tmp_array[i]] for i in range(len(tmp_array))]
        tmp_array = sorted(tmp_array, key=lambda x: -abs(x[1]))
        if repeat == True:
            return tmp_array
        else:
            idx = 1
            while idx < len(tmp_array):
                if abs(tmp_array[idx][1]-tmp_array[idx-1][1]) < precision:
                    del(tmp_array[idx])
                else: idx += 1
            return tmp_array


"****************************************************************************************************************"

if __name__ == '__main__':
    n_sites = 8
    J = 1

    graph  = nk.graph.Chain(n_sites, pbc=True)
    print("graph generated!")

    hilbert = nk.hilbert.Spin(s=0.5, N=n_sites)
    print("hilbert space generated!")

    for seed in range(1):

        s0, sx, sy, sz = [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]
        s = [s0, sx, sy, sz]
        J = jax.random.uniform(key=jax.random.PRNGKey(seed*2308), shape=(4,4))-0.5
        J = (J+J.T)/2.0
        bond = 0
        for idx1 in range(4):
            for idx2 in range(4):
                bond += J[idx1, idx2]*np.kron(s[idx1], s[idx2]) 

        bond_ops = [bond, ]
        site_ops = [-1*np.array(sx)]
        #ss = J*(np.kron(sz, sz))
        #hamiltonian = nk.operator.GraphOperator(hilbert=hilbert, graph=graph, site_ops=[], bond_ops=bond_ops)
        hamiltonian = nk.operator.LocalOperator(hilbert, dtype=complex)
        for i in range(n_sites):
            j = (i+1)%n_sites
            hamiltonian += nk.operator.LocalOperator(hilbert, bond, [i, j])
        #hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=graph, J=0.3)
        #print("Hamiltonian generated!")

        exact_result=Exact(hamiltonian)
        print("energy:", exact_result.gs_energy)
        print("corrlen:", exact_result.CorrLen())
        print("momentum:", exact_result.GSMomentum())
        print(exact_result.IsGSDegenerate())
        print("\n")
