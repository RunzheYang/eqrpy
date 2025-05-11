# EqrPy: Equilibrium Solvers for Multi-Player Normal-Form Games

**EqrPy** is a Python library for experimenting with equilibrium solvers in normal-form games. It implements several algorithms for computing Nash and correlated equilibria, including various equilibrium refinements.

> **Note:** This library is experimental and was developed for my own research. The methods are not fully tested — use with care.

---

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/runzheyang/eqrpy.git
cd eqrpy
pip install -e .
```

## Equilibrium Solver
We impliment equilibrium solvers for multi-player normal-form games.

To obtain Nash Equilibria (NE) and the refinements of NEs
```python
from eqrpy.nash import Porter_Nudelman_Shoham as PNS

# nplayer = |N|, nactions = [|A1|, |A2|, ..., |An|], utilites = u(a1,a2,...,an)
solver = PNS(nplayer, nactions, utilities)

# find all pure and mixed strategy Nash Equilibria
solver.solveNE(find_all=True)

# find all Pareto-Optimal Nash Equilibria
solver.solvePONE(find_all=True)

# find all Strong Nash Equilibria
solver.solveSNE(find_all=True)

# find all Coalition-Proof Nash Equilibria
solver.solveCPNE(find_all=True)
```

To obtain Correlated Equilibria: 

```python
from eqrpy.correlated import ConvPolytope as CP

# nplayer = |N|, nactions = [|A1|, |A2|, ..., |An|], utilites = u(a1,a2,...,an)
solver = CP(nplayer, nactions, utilities)

# find all basic Correlated Equilibria (any linear combinations are CEs)
solver.solveCE(backend='ppl')

# find special Correlated Equilibria w/ linear preferences
solver.solveCE(final_special=[pref1, pref2, ...])
```
See [`examples.ipynb`](examples/examples.ipynb) for equilibria of examples games.
