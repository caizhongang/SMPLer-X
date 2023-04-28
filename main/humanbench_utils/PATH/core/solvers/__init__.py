from .solver import Solver
from .solver_deter import SolverDeter
from .solver_multitask_dev import SolverMultiTaskDev

def solver_entry(C):
    return globals()[C.config['common']['solver']['type']](C)
