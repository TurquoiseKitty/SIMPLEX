import scipy.io

from hand_solver import simplex_cano_solver
from cvxpy_solve import ComputeLP

mat = scipy.io.loadmat('NETLIB/AGG.mat')

A = mat["A"].toarray()

b = mat['b'].squeeze()

c = mat['c'].squeeze()

solver = simplex_cano_solver(A, b, c, perturb=True, demo = True, rule = "standard")

solver.solve()


print(solver.base_finding_step, solver.opti_finding_step)

print(solver.get_min_val())

_, crit = ComputeLP(A, b, c)

print(crit)