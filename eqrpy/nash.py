from scipy import optimize
import numpy as np
import itertools

class Porter_Nudelman_Shoham():
    """
    PNS algorithm: Simple search methods for finding a Nash equilibrium. 2004
    https://cs.uwaterloo.ca/~klarson/teaching/W10-798/papers/Porter08GEB.pdf
    code reference: https://github.com/wendingliu/NE-solver
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
        
        # cache for coalition-proof NEs
        self.CPNE_dict = {}
        
        
    def solveNE(self, find_all=True, method="hybr", rounding_err=1e-6):
        if self.n == 2:
            return self.algorithm1(find_all, method, rounding_err)
        else:
            return self.algorithm2(find_all, method, rounding_err)
        
        
    def algorithm1(self, find_all=True, method="hybr", rounding_err=1e-6):
        """
        find all NEs in 2-player game if find_all == True
        find_all: return one sample NE if False
        method: algorithm used in scipy solver
        """
        NEs = []
        A0, A1 = self.A 
        
        # all support sizes
        xs = np.array([[x0, x1] for x0 in range(1, self.nA[0]+1) 
                                for x1 in range(1, self.nA[1]+1)])
        xs_balance = np.abs(np.diff(xs).ravel())  
        xs_size = np.sum(xs, axis=1)
        xs = xs[np.lexsort((xs_size, xs_balance))]
        
        # precompute all candidate supports for player 0 to save time 
        all_S0, all_S0_sizes = self.combinations(A0, return_size=True)  
        
        for x in xs:
            x0, x1 = x
            for S0 in all_S0[all_S0_sizes == x0]:
                A1prime = [a1 for a1 in A1 if not self.conditional_dominated(i=1, a=a1, Ai=A1, R=[[a0] for a0 in S0])]
                S0_dom_by_A1prime = [self.conditional_dominated(i=0, a=a0, Ai=S0, R=[[a1] for a1 in A1prime]) 
                                     for a0 in S0]
                if not np.any(S0_dom_by_A1prime):
                    for S1 in self.combinations(A1prime, fixed_size=x1, return_size=False):
                        S0_dom_by_S1 = [self.conditional_dominated(i=0, a=a0, Ai=S0, R=[[a1] for a1 in S1]) 
                                        for a0 in S0]
                        if not np.any(S0_dom_by_S1):
                            try:
                                S = [list(S0), list(S1)]
                                (sigma, v) = self.feasibility(S, method=method, 
                                                              rounding_err=rounding_err)
                                if not find_all:
                                    return (sigma, v)
                                if not self.check_duplicates(sigma, NEs):
                                    NEs.append((sigma, v))
                            except TypeError:
                                pass
        return NEs
    
    
    def algorithm2(self, find_all=True, method="hybr", rounding_err=1e-6):
        """
        find all NEs in n-player game if find_all == True
        find_all: return one sample NE if False
        method: algorithm used in scipy solver
        """
        NEs = []
        
        allS, allS_size = [], []
        for i in range(self.n):
            allSi, allSi_size = self.combinations(self.A[i], return_size=True)
            allS.append(allSi)
            allS_size.append(allSi_size)
        
        # all support sizes
        loop_nest = [list(range(1, self.nA[i] + 1)) for i in range(self.n)]
        xs = np.array(list(itertools.product(*loop_nest)))
        xs_balance = np.max(xs, axis=1) - np.min(xs, axis=1)
        xs_size = np.sum(xs, axis=1)
        ind = np.lexsort((xs_balance, xs_size))
        xs = xs[np.lexsort((xs_balance, xs_size))]
        
        for x in xs:
            S = [[] for i in range(self.n)]
            D = [allS[i][allS_size[i] == x[i]].tolist() for i in range(self.n)]
            if find_all == False:
                try:
                    sigma, v = self.recursive_backtracking(S=S, D=D, i=0, NEs=[], 
                                                           find_all=False, method=method, 
                                                           rounding_err=rounding_err)
                    return sigma, v
                except TypeError:
                    pass
            else:
                result = self.recursive_backtracking(S=S, D=D, i=0, NEs=[], 
                                                     find_all=True, method=method, 
                                                     rounding_err=rounding_err)
                if result:
                    NEs += result
        return NEs
        
        
    def recursive_backtracking(self, S, D, i, find_all=True, method="hybr", NEs=[], rounding_err=1e-6):
        """
        Recursive-Backtracking
        """
        if i == self.n:
            try:
                sigma, v = self.feasibility(S, method, rounding_err)
                if not find_all:
                    return sigma, v
                else:
                    if not self.check_duplicates(sigma, NEs):
                        NEs.append((sigma, v))
            except TypeError:
                pass
    
        # when some Si is empty
        else:  
            for Si in D[i]:
                S[i] = Si
                Dnew = [[S[k]] for k in range(i+1)] + D[i+1:]
                
                Dnew = self.IRSDS(Dnew)
                
                if not Dnew is None:
                    try:
                        if not find_all:  # call self.rb recursively until finding the first NE
                            sigma, v = self.recursive_backtracking(S=S, D=Dnew, i=i+1, NEs=[],
                                                                   find_all=False, method=method, 
                                                                   rounding_err=rounding_err)
                            return sigma, v
                        else:
                            NEs = self.recursive_backtracking(S=S, D=Dnew, i=i+1, NEs=NEs, 
                                                              find_all=True, method=method, 
                                                              rounding_err=rounding_err)
                    except TypeError:
                        pass
                    
        return NEs
    
    
    def IRSDS(self, D):
        """
        Iterated Removal of Strictly Dominated Strategies
        """
        changed = True
        while changed:
            changed = False
            for i in range(self.n):
                for ai in set(itertools.chain.from_iterable(D[i])):
                    D_js = D[:i] + D[i+1:]
                    for j in range(len(D_js)):
                        D_js[j] = set(itertools.chain.from_iterable(D_js[j]))
                    # R: nested list of other players' action profile
                    # i.e., the union set of S(-i) for all all S(-i) in D(-i)
                    R = list(itertools.product(*D_js))
                    if self.conditional_dominated(i=i, a=ai, Ai=self.A[i], R=R):
                        D[i] = [Si for Si in D[i] if ai not in Si]
                        changed = True
                        if len(D[i]) == 0:
                            return None
        return D
    
    
    def conditional_dominated(self, i, a, Ai, R):
        """
        ∃ a' in Ai, ∀ a_{-i} in R
        u(a, a_{-i}) < u(a', a_{-i})
        """
        for ai in (set(Ai) - {a}):
            dominated = True
            for b in R:
                u_original = self.u[tuple(b[:i]) + (a,) + tuple(b[i:])][i]
                u_replaced = self.u[tuple(b[:i]) + (ai,) + tuple(b[i:])][i]
                if u_original >= u_replaced:
                    break
            else:
                return True
        else:
            return False
        
        
    def feasibility(self, S, method="hybr", rounding_err=1e-6):
        """
        find the mixed NE in n-player game given supports S=(S0, ..., Sn-1)
        it's nonlinear when n>2
        """
        
        try:
            sum_nA = np.sum(self.nA)
            f = lambda z: self.equality(z, S)
            sol = optimize.root(f,  
                                x0=[1/na for na in self.nA for _ in range(na)] + [0] * self.n, 
                                method=method)
            
            z = sol.x
            
            # rounding to prevent duplicates due to numerical error
            z[:sum_nA][(1 - rounding_err) < z[:sum_nA]] = 1.0
            z[np.abs(z) < rounding_err] = 0.0
            
            # remove redundant pure NE candidate when |Si| > 1
            if np.any(np.array([len(Si) for Si in S]) != 1) and set(z[:sum_nA]) == {0.0, 1.0}:
                return None
            else:
                sigma, v = self.parse(z)
        
        # infeasible (equalities do not hold)
        except ValueError:
            return None
        
        if np.any(np.abs(f(z)) > rounding_err):
            return None
        elif self.inequality(sigma, v, S):
            return sigma, v
        else:
            # infeasible (inequalities do not hold / not an NE)
            return None
        
        
    def equality(self, z, S):
        """
        equality conditions in Feasibility Program
        """
        support = list(itertools.product(*S))
        sigma, v = self.parse(z)
        eqs = []
        
        # 1. sigma(ai) = 0 for ai not in Si
        for i in range(self.n):
            if set(S[i]) < set(self.A[i]):
                sigmai = sigma[i]
                eqs += [sigmai[j] - 0 for j in set(self.A[i]) - set(S[i])]
        
        # 2. sum_a {simga_i(a)} = 1
        eqs += [np.sum(sigmai) - 1 for sigmai in sigma]
        
        # 3. sum_{a_{-i}} sigma(a_{-i})u(ai, a_{-i}) = vi
        for i in range(self.n):
            vi = v[i]
            for ai in S[i]:
                utility = 0
                for a in support:
                    if a[i] == ai:
                        sigma_js = np.product([sigma[j][a[j]] for j in range(self.n) if j != i])
                        u = self.u[a][i]
                        utility += sigma_js * u
                eqs.append(utility - vi)
                
        return eqs
    
    
    def inequality(self, sigma, v, S):
        """
        inequality conditions in Feasibility Program
        """
        # 1. sigma_i(a) >= 0 for all a
        for sigma_i in sigma:
            if np.any(np.array(sigma_i) < 0):
                return False 
            
        # 2. E[u(ai, a_{-i})] ≥ E[u(ai', a_{-i})] for ai in Si and ai' in Ai/Si
        support_counterfactual = []
        if self.A != S:
            for i in range(self.n):
                if set(S[i]) < set(self.A[i]):
                    S_counterfactual = (S[:i] + [list(set(self.A[i]) - set(S[i]))] + S[i+1:]) 
                    support_counterfactual += list(itertools.product(*S_counterfactual))
        if len(support_counterfactual) > 0:
            for i in range(self.n):
                if set(self.A[i]) - set(S[i]):
                    vi = v[i]
                    for ai in set(self.A[i]) - set(S[i]):
                        utility = 0 
                        for a in support_counterfactual:
                            if a[i] == ai:
                                # probability of s happens conditional on i plays ai (ai not in Si)
                                # prob(a_{-i}) = product(pj(aj)),j!=i
                                sigma_js = np.product([sigma[j][a[j]] for j in range(self.n) if j != i])
                                u = self.u[a][i]
                                utility += sigma_js * u
                        if utility - vi > 1e-6:
                            return False
        return True
    
    
    def parse(self, z):
        """
        z: an array of size (sum(nA)+n), z=[sigma,v]=[sigma_0,sigma_1,...,sigma_n,v],
        """
        indices = np.cumsum(self.nA)
        v = z[np.sum(self.nA):]
        
        p = [] 
        for i in range(self.n):  
            start = indices[i - 1] if i >= 1 else 0
            end = indices[i] 
            pi = z[start:end] 
            p.append(pi)
            
        return p, v
    
    
    def combinations(self, iterable, fixed_size=None, return_size=True): 
        s = list(iterable)
        res, res_size = [], []
        if fixed_size:
            all_comb = itertools.combinations(s, fixed_size)
        else:
            all_comb = itertools.chain.from_iterable(
                            itertools.combinations(s, r) 
                        for r in range(1, len(s) + 1))
        res = list(all_comb)
        
        if return_size:
            res_size = [len(r) for r in res]
                
        if return_size:
            return np.array(res, dtype=object), np.array(res_size)
        else:
            return res
        
        
    def check_duplicates(self, sigma, NEs, rounding_err=1e-6):
        """
        check if a sigma is already in NEs
        """
        for sig, u in NEs:
            err = np.abs(np.concatenate(sig) - np.concatenate(sigma)).max()
            if err < rounding_err:
                return True
        return False
    
    
    def pareto_dominates(self, a, b):
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


    def pareto_frontier(self, vectors):
        frontier, frontier_ids = [], []
        for i, vector in enumerate(vectors):
            if not any(self.pareto_dominates(x, vector) for x in vectors):
                frontier.append(vector)
                frontier_ids.append(i)
        return frontier, frontier_ids
    
    
    # Pareto-Optimal NE in NE
    def solvePONE(self, find_all=True, method="hybr", rounding_err=1e-6):
        if self.n == 2:
            NEs = self.algorithm1(find_all, method, rounding_err)
            return list(np.array(NEs, dtype=object)[self.pareto_frontier([sol[1] for sol in NEs])[1]])
        else:
            NEs = self.algorithm2(find_all, method, rounding_err)
            return list(np.array(NEs, dtype=object)[self.pareto_frontier([sol[1] for sol in NEs])[1]])
 
    
    # NE that are resistant to pure multilateral deviation, only used in Strong Nash Definition
    def solvePONE_all(self, find_all=True, method="hybr", rounding_err=1e-6):
        PONEs = self.solvePONE(find_all, method, rounding_err)
        POUs = self.pareto_frontier(self.u.reshape(-1, self.n))[0]
        res = []
        for i, sol in enumerate(PONEs):
            if not any(self.pareto_dominates(u, sol[1]) for u in POUs):
                res.append(sol)
        return res

        
    def reduce_list(self, a, b):
        return [value for index, value in enumerate(a) if index not in b]
        
        
    def reduce_utility(self, u, sigma, fixed_players = []):
        res = 0
        coalition = tuple(self.reduce_list(range(u.shape[-1]), fixed_players))
        reduction_dim = tuple([slice(None)] * (u.shape[-1] - len(fixed_players)) + [coalition])
        action_candidates = []
        for i in range(u.shape[-1]):
            if i in fixed_players:
                action_candidates.append(list(range(2)))
            else:
                action_candidates.append([slice(None)])
        for a in itertools.product(*action_candidates):
            pa = np.prod([sigmai[ai] for sigmai, ai in zip(sigma, a) if ai != slice(None)])
            res += u[a] * pa
        return res[reduction_dim]
    
    
    # Strong NE, only necessary condition (only resistant to pure multilateral deviation and other NE)
    # A necessary and sufficient condition is still open. See more discussion in Gatti, Rocco and Sandholm.
    def solveSNE(self, find_all=True, method="hybr", rounding_err=1e-6):
        PONEs = self.solvePONE_all(find_all=find_all, method=method, rounding_err=rounding_err)
        if self.n == 2:
            return PONEs
        else:
            if len(PONEs) == 0:
                return PONEs
            else:
                fixed_player_plans = []
                for r in range(1, self.n - 1):
                    fixed_player_plans.extend(itertools.combinations(range(self.n), r))

                strong = [True] * len(PONEs)
                for i, sol in enumerate(PONEs):
                    sigma, utility = sol
                    for fixed_players in fixed_player_plans:
                        sub_utility = self.reduce_list(utility, fixed_players)
                        sub_game = Porter_Nudelman_Shoham(self.n-len(fixed_players), 
                                                          self.reduce_list(self.nA, fixed_players), 
                                                          self.reduce_utility(self.u, sigma, fixed_players))
                        sub_sols = sub_game.solvePONE_all(find_all=True, method=method, rounding_err=rounding_err)
                        u_frontier = [s[1] for s in sub_sols] + self.pareto_frontier(sub_game.u.reshape(-1, sub_game.n))[0]
                        if any(self.pareto_dominates(u, sub_utility) for u in u_frontier):
                            strong[i] = False
                            break

                return list(np.array(PONEs, dtype=object)[strong])


    # Recursive Definition of Coalition-Proof NE
    def find_all_enforcing_strategy(self, nplayer, nactions, utilities, method="hybr", rounding_err=1e-6, cache=True):
        sub_game = Porter_Nudelman_Shoham(nplayer, nactions, utilities)
        if nplayer == 2:
            return sub_game.solveNE(find_all=True, method=method, rounding_err=rounding_err)
        else:
            enforce_candi = sub_game.solveNE(find_all=True, method=method, rounding_err=rounding_err)
            CPNE = [True] * len(enforce_candi)
            fixed_player_plans = []
            for r in range(1, nplayer - 1):
                fixed_player_plans.extend(itertools.combinations(range(nplayer), r))
                
            for i, sol in enumerate(enforce_candi):
                sigma, utility = sol
                for fixed_players in fixed_player_plans:
                    sub_sigma = self.reduce_list(sigma, fixed_players)
                    sub_utility = self.reduce_list(utility, fixed_players)
                    if not self.is_CPNE(sub_sigma, sub_utility,
                                   nplayer=nplayer-len(fixed_players), 
                                   nactions=self.reduce_list(nactions, fixed_players), 
                                   utilities=self.reduce_utility(utilities, sigma, fixed_players), 
                                   method=method, rounding_err=rounding_err, cache=cache):
                        CPNE[i] = False
                        break
            
            return list(np.array(enforce_candi, dtype=object)[CPNE])
        
        
    def is_CPNE(self, sub_sigma, sub_utility, nplayer, nactions, utilities, method="hybr", rounding_err=1e-6, cache=True):
        sub_game = Porter_Nudelman_Shoham(nplayer, nactions, utilities)
        if nplayer == 2:
            PONEs = None
            if cache:
                gamekey = str(nplayer) + "|" + str(nactions) + "|" + str(utilities)
                if gamekey in self.CPNE_dict.keys():
                    PONEs = self.CPNE_dict[gamekey]
                else:
                    PONEs = sub_game.solvePONE(find_all=True, method=method, rounding_err=rounding_err)
                    self.CPNE_dict[gamekey] = PONEs
            else:
                PONEs = sub_game.solvePONE(find_all=True, method=method, rounding_err=rounding_err)
            for sol in PONEs:
                if np.sum(np.abs(sol[1] - sub_utility)) < rounding_err:
                    return True
            return False
        else:
            POCPNE_candi = None
            if cache:
                gamekey = str(nplayer) + "|" + str(nactions) + "|" + str(utilities)
                if gamekey in self.CPNE_dict.keys():
                    POCPNE_candi = self.CPNE_dict[gamekey]
                else:
                    CPNE_candi = self.find_all_enforcing_strategy(nplayer, nactions, utilities, method=method, rounding_err=rounding_err, cache=cache)
                    POCPNE_candi = list(np.array(CPNE_candi, dtype=object)[self.pareto_frontier([sol[1] for sol in CPNE_candi])[1]])
                    self.CPNE_dict[gamekey] = POCPNE_candi
            else:
                CPNE_candi = self.find_all_enforcing_strategy(nplayer, nactions, utilities, method=method, rounding_err=rounding_err, cache=cache)
                POCPNE_candi = list(np.array(CPNE_candi, dtype=object)[self.pareto_frontier([sol[1] for sol in CPNE_candi])[1]])
                
            for sol in POCPNE_candi:
                if np.sum(np.abs(sol[1] - sub_utility)) < rounding_err:
                    return True
            return False
            
                    
    def solveCPNE(self, find_all=True, method="hybr", rounding_err=1e-6, cache=True):
        if self.n == 2:
            return self.solvePONE(find_all=find_all, method=method, rounding_err=rounding_err)
        else:
            NEs = self.solveNE(find_all=find_all, method=method, rounding_err=rounding_err)
            CPNE = [True] * len(NEs)
            for i, sol in enumerate(NEs):
                sigma, utility = sol
                if not self.is_CPNE(sigma, utility, self.n, self.nA, self.u, method=method, rounding_err=rounding_err, cache=cache):
                    CPNE[i] = False
            return list(np.array(NEs, dtype=object)[CPNE])
            
        