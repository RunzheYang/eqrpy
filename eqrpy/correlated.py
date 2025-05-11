from scipy import optimize
import numpy as np
import itertools
from scipy.optimize import LinearConstraint
from scipy.spatial import ConvexHull

class ConvPolytope():
    """
    ConvPolyhedron algorithm: 
    Directely solve an LP similar to Primary LP in Papadimitriou and Roughgarden (2008)
    LP inequilities are H-representations of Polyhedron
    Policy Generators are V-representations of Polyhedron
    """
    def __init__(self, nplayer, nactions, utilities):
        """
        # nplayer = |N|
        # nactions = [|A1|, |A2|, ..., |An|]
        # utilites = u(a1,a2,...,an)
        """
        self.n = nplayer
        self.nA = nactions
        self.A = [list(range(a)) for a in nactions]
        self.u = np.array(utilities)
        self.action_profiles = np.array(list(itertools.product(*self.A)))
        self.u_profiles = np.concatenate([self.u[tuple(a)].reshape(-1,self.n) 
                                          for a in self.action_profiles], axis=0)
    
    
    def generate_inequality(self):
        n_profile = len(self.action_profiles)
        res = []
        
        # generate inequalities describing equilibria
        for p in range(self.n):
            for ap in self.A[p]:
                for ap_prime in self.A[p]:
                    if ap == ap_prime: continue
                    ieq = np.zeros(n_profile)
                    mask_ = (self.action_profiles[:,p] == ap)
                    a_contains_ap = self.action_profiles[mask_]
                    utility_diff = []
                    for a in a_contains_ap:
                        a = tuple(a)
                        a_prime = a[:p] + (ap_prime,) + a[p+1:]
                        utility_diff.append(self.u[a][p] - self.u[a_prime][p])
                    ieq[mask_] = np.array(utility_diff)
                    res.append(ieq)
        res = np.array(res)
    
        # non-negative constraints of probability
        res = np.append(res, np.eye(n_profile), axis=0)    
        # unity constrain of probability
        res = np.append(res, np.ones((1, n_profile)), axis=0)
        
        return res
    
    
    def _estimate_utility(self, action_gen):
        res = 0
        for i, a in enumerate(self.action_profiles):
            res += self.u[tuple(a)] * action_gen[i]
        return (self.u_profiles.T * action_gen).sum(1)
    
    
    def solveCE(self, final_special=None, backend='ppl'):
        CEs = []
        # matrix [-b, A] in H-rep representes Ai * xi >= bi
        A = self.generate_inequality()
        b = np.zeros((len(A), 1))
        n_eq = len(A)
        b[n_eq - 1, 0] = 1
        H_rep = np.append(-b, A, axis=1)
        if final_special is None:
            if backend == 'cdd':
                import cdd
                # create CE polyhedron
                polyhedron = cdd.Matrix(H_rep, linear=True, number_type='float')
                polyhedron.lin_set = [n_eq - 1]
                polyhedron.rep_type = cdd.RepType.INEQUALITY
                polyhedron.canonicalize()
                # enumerate vertices of the CE polyhedron
                poly_gens = cdd.Polyhedron(polyhedron).get_generators()
                action_gens = np.array(poly_gens)[:,1:]
                for a in action_gens:
                    CEs.append((a, self._estimate_utility(a)))
            elif backend == 'ppl':
                import ppl
                cs = ppl.Constraint_System()
                for coef in H_rep:
                    n_aprofile = len(self.action_profiles)
                    lhs = np.sum(np.array([ppl.Variable(i) for i in range(n_aprofile)]) * coef[1:])
                    if coef[0] == 0:
                        cs.insert(lhs >= 0)
                    elif coef[0] == -1:
                        cs.insert(lhs == 1)
                poly = ppl.C_Polyhedron(cs)
                point2list = lambda x: np.array([float(y/x.divisor()) for y in x.coefficients()])
                ploy_gens = poly.minimized_generators()
                action_gens = [point2list(pg) for pg in ploy_gens]
                for a in action_gens:
                    CEs.append((a, self._estimate_utility(a)))
            else:
                print(f"Unknown support: {backend}")
        else:
            for criterion in final_special:
                CEs.append(self.solve_special_CE(criterion))
        return CEs
    
    
    def solve_special_CE(self, criterion):
        criterion = np.array(criterion)
        A = self.generate_inequality()
        b = np.zeros(len(A))
        n_eq = len(A)
        b[n_eq - 1] = 1
        n_aprofile = len(self.action_profiles)
        c = -(self.u_profiles * criterion).sum(1)
        res = optimize.linprog(c, 
                         method='interior-point',
                         options={"sym_pos":False},
                         A_ub=-A[:-1], b_ub=-b[:-1], 
                         A_eq=-A[-1:], b_eq=-b[-1:], 
                         bounds=[[0, 1]]*n_aprofile)
        return res.x, self._estimate_utility(res.x)
    
    
    def pareto_dominates(self, a, b):
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    
    def pareto_frontier(self, vectors):
        frontier, frontier_ids = [], []
        for i, vector in enumerate(vectors):
            if not any(self.pareto_dominates(x, vector) for x in vectors):
                frontier.append(vector)
                frontier_ids.append(i)
        return frontier, frontier_ids
    
    
    def solvePOCE(self, backend='ppl'):
        CEs = self.solveCE(backend=backend)
        CEs_strategies, CE_utilities = [c[0] for c in CEs], [c[1] for c in CEs]
        CE_utilities = np.array(CE_utilities)
        try:
            conv = ConvexHull(CE_utilities)
            expanded_conv_id = []
            for idx in conv.vertices:
                expanded_conv_id += list(np.where(np.equal(CE_utilities, CE_utilities[idx]).all(axis=1))[0])
            pareto_inds = self.pareto_frontier(CE_utilities[expanded_conv_id])[1]
            return list(np.array(CEs, dtype=object)[conv.vertices[pareto_inds]])
        except:
            pareto_inds = self.pareto_frontier(CE_utilities)[1]
            return list(np.array(CEs, dtype=object)[pareto_inds])