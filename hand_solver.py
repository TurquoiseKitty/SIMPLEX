## SIMPLEX for canonical form, min c^T x, s.t. Ax = b, x>= 0
## Also we assume nondegeneracy

import numpy as np
import copy
import time
import os, psutil
from cvxpy_solve import ComputeLP

LB = 1e-9
PERT = 1e-6

class simplex_cano_solver:
    
    def __init__(self, A, b, c, basis = [], rule = "standard", perturb=False, demo = False):
        
        self.n_var = len(c)
        self.n_eq = len(b)
        self.base_finding_step = 0
        self.opti_finding_step = 0
        self.rule = rule
        self.demo = demo
        
        assert len(A) == self.n_eq
        assert len(A[0]) == self.n_var
        
        self.original_A = A
        self.original_b = b
        self.original_c = c

        A = A.astype("float64")
        b = b.astype("float64")
        c = c.astype("float64")

        
        neg_idx = np.where(b < 0)[0]
        
        A[neg_idx] = -A[neg_idx]
        b[neg_idx] = -b[neg_idx]

        if perturb:

            pert = np.random.uniform(-PERT, PERT, size = len(b))
            pert2 = np.random.uniform(-PERT, PERT, size = A.shape)

            b += pert

            A += pert2
        
        
        self.A = A
        self.b = b
        self.c = c
        
        self.basis_index = np.zeros(self.n_eq)
        self.curr_neg = 0
        
        if len(basis) == 0:
            
            self.find_basis()
            
        else:
            
            assert len(basis) == self.n_eq
                
            self.basis_index = basis
                
            self.reducefrom_basis()
            
    def get_min_val(self):
        
        return -self.curr_neg
    
    def rounding(self, lb):
        
        self.A[np.abs(self.A) < lb] = 0
        self.b[np.abs(self.b) < lb] = 0
        self.c[np.abs(self.c) < lb] = 0
            
    def reducefrom_basis(self, mini_turb = False, entering_idx = None):
        
        slice_mat = self.A[:, self.basis_index]
        
        if not mini_turb:
            
            trans_mat = np.linalg.inv(slice_mat)
            
        else:
            
            trans_mat = copy.deepcopy(slice_mat)
            
            assert sum(np.diag(slice_mat) - np.array([1.0] * self.n_eq)) < 1e-2
            
            trans_mat_idx = np.where(self.basis_index == entering_idx)[0][0]
            
            trans_mat[:,trans_mat_idx] = - trans_mat[:,trans_mat_idx]
            trans_mat[trans_mat_idx, trans_mat_idx] = 1.0
            
        
            
        self.A = trans_mat @ self.A
        self.b = trans_mat @ self.b
        
        self.curr_neg -= sum(self.b * self.c[self.basis_index])
        
        self.c = self.c - self.A.T @ self.c[self.basis_index] 
        
    def mini_replace(self, entering_idx, leaving_idx):

        
        replacehappen_row = np.where(self.basis_index == leaving_idx)[0][0]
        
        self.basis_index[replacehappen_row] = entering_idx
        
        self.b[replacehappen_row] /= self.A[replacehappen_row, entering_idx]
        
        self.A[replacehappen_row] /= self.A[replacehappen_row, entering_idx]
        
        self.reducefrom_basis(mini_turb = True, entering_idx = entering_idx)
        
        
    def solve(self, rule = None, terminate = 20000):
        
        if not rule:
            
            rule = self.rule
        
        assert np.min(self.b) > 0
        
        count = 0
        
        result_dict = {
            "degenerate" : False,
            "unbounded" : False,
            "early_terminate" : False,
            "solved" : False
        }

        if self.demo:

            start_time = time.time()
        
        while(count < terminate):

            if self.demo:

                if count%400 == 0:

                    print("iter ",count)
                    print("time elapse ",time.time()-start_time)
                    process = psutil.Process(os.getpid())
                    print("memory usage ",process.memory_info().rss)  # in bytes 


                    '''
                    temp = psutil.sensors_temperatures()
                    print("cpu temperature ",temp)

                    while True:

                        if temp < 90:

                            break

                        else:

                            time.sleep(5)
                            temp = psutil.sensors_temperatures()
                            print("cpu temperature ",temp)

                    '''
            
            ## rounding
            self.rounding(LB)
            
            ## search for a pivot
            if np.min(self.c) >= 0:
                result_dict["solved"] = True
                return result_dict

            
            
            ## pivoting
            if rule == "standard":
                
                # standard rule cannot deal with degenerate
                    
                min_idx = np.where(self.c == np.min(self.c))[0]
                
                entering_idx = np.random.choice(min_idx,1)[0]

                if max(self.A[:, entering_idx]) <= 0:
                        
                    result_dict["unbounded"] = True
                    return result_dict
                
                ## now the leaving_idx
                
                judge = self.b / np.clip(np.abs(self.A[:, entering_idx]), 1e-6, None) * np.sign(self.A[:, entering_idx])
                min_val = np.min(judge[self.A[:, entering_idx] > 0])
                
                if min_val == 0:
                    
                    result_dict['degenerate'] = True
                    return result_dict
                    
                
                leaving_idxes = np.where((self.A[:, entering_idx] > 0)&(judge == min_val) )[0]
                leaving_idx = self.basis_index[np.random.choice(leaving_idxes,1)[0]]
                
                                
            elif rule == "bland":

                entering_idx = np.where(self.c < 0)[0][0]

                if max(self.A[:, entering_idx]) <= 0:
                        
                    result_dict["unbounded"] = True
                    return result_dict
                
                ## now the leaving_idx
                
                judge = self.b / np.clip(np.abs(self.A[:, entering_idx]), 1e-6, None) * np.sign(self.A[:, entering_idx])
                min_val = np.min(judge[self.A[:, entering_idx] > 0])

                leaving_idx = self.basis_index[np.where((self.A[:, entering_idx] > 0)&(judge == min_val) )[0][0]]

            else:
                
                raise Exception("rule error")
            
            
            self.mini_replace(entering_idx, leaving_idx)
            
            count += 1
            self.opti_finding_step += 1
            
        result_dict["early_terminate"] = True
        return result_dict
        
        
        
        
    def find_basis(self):
        
        # consider a new problem to solve
        new_A = np.concatenate((self.A, np.identity(self.n_eq)), axis = 1)
        new_b = copy.deepcopy(self.b)
        
        
        new_c = np.zeros(self.n_var + self.n_eq)
        new_c[self.n_var:] = 1
        
        basis = np.where(new_c == 1)[0]
        
        new_probins = simplex_cano_solver(new_A, new_b, new_c, basis = basis, rule = self.rule, demo = self.demo)
        
        state = new_probins.solve()
        
        if not state["solved"]:
            
            print(state)
            
            raise Exception("basis not found!")
            
        opt = new_probins.get_min_val()
        
        if opt > 10 * PERT:
            
            print(opt)
            
            raise Exception("problem infeasible!")
            
        
            
        
        ## there might need to be one slightly tuning step
        ## like, removing redundant equations
        candidate_idx = new_probins.basis_index
        bad_idxes = candidate_idx[candidate_idx >= self.n_var]
        
        if len(bad_idxes) > 0:

            raise Exception("problem degenerate")

            
        else:
            
            self.base_finding_step = new_probins.opti_finding_step
            
            self.A = new_probins.A[:, :self.n_var]
            self.b = new_probins.b
            
            self.basis_index = candidate_idx
            
        self.reducefrom_basis()



if __name__ == "__main__":

    np.random.seed(1234)

    N = 1000

    A = np.random.randint(1, 2, size = (N, N))

    A = np.concatenate((A, np.identity(N)), axis= 1)

    b = np.random.randint(2*N, 4*N, size = N)

    c = np.random.randint(1, 2, size = 2*N)

    solver = simplex_cano_solver(A, b, c, rule = "bland", perturb = True, demo = True)

    solver.solve()

    print(solver.base_finding_step, solver.opti_finding_step)

    print(solver.get_min_val())

    _, crit = ComputeLP(A, b, c)

    print(crit)