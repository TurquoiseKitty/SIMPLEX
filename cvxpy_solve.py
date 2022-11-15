# Compute LP
import numpy as np
import cvxpy as cp


def ComputeLP(A, b, c):
    (dim_constraints, dim_target) = A.shape
    x = cp.Variable(dim_target)
    constr = [A@x == b, x >= 0]
    obj = cp.Minimize(c.T@x)
    prob = cp.Problem(obj, constr)
    prob.solve()
    return x.value, prob.value