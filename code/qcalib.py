"""Minimal but complete fsQCA engine (Ragin 2008; Schneider & Wagemann 2012).
Direct calibration, necessity (consistency / coverage / RoN), truth table,
Quine-McCluskey minimisation, complex / parsimonious / intermediate (easy
counterfactuals under directional expectations) solutions, solution metrics.
Source: the authors' own implementation, reused unchanged from their earlier fsQCA study."""
import numpy as np, pandas as pd, itertools

# ---------------------------------------------------------------- calibration
def _direct(x, non, cross, full):
    x = np.asarray(x, float)
    lo = np.where(cross > non, 3.0 / (cross - non), 3.0 / 1e-9)
    hi = np.where(full > cross, 3.0 / (full - cross), 3.0 / 1e-9)
    lod = np.where(x >= cross, (x - cross) * hi, (x - cross) * lo)
    m = 1 / (1 + np.exp(-lod))
    m = np.where(np.abs(m - 0.5) < 1e-9, 0.501, m)   # avoid exact 0.5
    return m

def calibrate_grouped(x, groups, pcts=(5, 50, 95)):
    """Direct-method calibration with percentile anchors computed inside each group."""
    x = pd.Series(x).reset_index(drop=True); g = pd.Series(groups).reset_index(drop=True)
    out = np.empty(len(x))
    gl = np.nanpercentile(x, pcts)
    for k in g.unique():
        idx = (g == k).values
        v = x[idx]
        a = np.nanpercentile(v, pcts)
        if not (a[0] < a[1] < a[2]):                 # degenerate anchors -> pooled anchors
            a = gl
            if not (a[0] < a[1] < a[2]):
                a = (a[0] - 1e-6, a[1], a[2] + 1e-6)
        out[idx] = _direct(v, *a)
    return out

# ------------------------------------------------------------------ measures
def cons(X, Y):  return np.minimum(X, Y).sum() / max(X.sum(), 1e-12)
def cov(X, Y):   return np.minimum(X, Y).sum() / max(Y.sum(), 1e-12)
def pri(X, Y):
    nY = 1 - Y; num = np.minimum(X, Y).sum() - np.minimum.reduce([X, Y, nY]).sum()
    den = X.sum() - np.minimum.reduce([X, Y, nY]).sum()
    return num / max(den, 1e-12)

def necessity(M, Y, names):
    rows = []
    for c in names:
        for lab, X in ((c, M[c].values), ('~' + c, 1 - M[c].values)):
            ron = (1 - X).sum() / max((1 - np.minimum(X, Y)).sum(), 1e-12)
            rows.append(dict(cond=lab, consistency=np.minimum(X, Y).sum() / Y.sum(),
                             coverage=np.minimum(X, Y).sum() / X.sum(), RoN=ron))
    return pd.DataFrame(rows)

# --------------------------------------------------------------- truth table
def truth_table(M, Y, names):
    k = len(names); A = M[names].values
    Xs = np.stack([A, 1 - A], 0)                       # [present, absent]
    rows = []
    for idx in range(2 ** k):
        bits = [(idx >> (k - 1 - j)) & 1 for j in range(k)]      # 1 = present
        mem = np.ones(len(Y))
        for j, b in enumerate(bits):
            mem = np.minimum(mem, A[:, j] if b else 1 - A[:, j])
        n = int((mem > 0.5).sum())
        if mem.sum() > 0:
            rows.append(dict(row=idx, bits=tuple(bits), n=n, incl=cons(mem, Y), PRI=pri(mem, Y), mem_sum=mem.sum()))
        else:
            rows.append(dict(row=idx, bits=tuple(bits), n=0, incl=np.nan, PRI=np.nan, mem_sum=0.0))
    return pd.DataFrame(rows)

def code_rows(tt, n_cut, incl_cut, pri_cut):
    on = (tt.n >= n_cut) & (tt.incl >= incl_cut) & (tt.PRI >= pri_cut)
    rem = tt.n < n_cut
    return on.values, rem.values

# ----------------------------------------------------- Quine-McCluskey (k<=10)
def _merge(a, b):
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if len(diff) == 1 and a[diff[0]] != '-' and b[diff[0]] != '-':
        c = list(a); c[diff[0]] = '-'; return tuple(c)
    return None

def prime_implicants(minterms, k):
    cur = {tuple((m >> (k - 1 - j)) & 1 for j in range(k)) for m in minterms}
    primes = set()
    while cur:
        nxt, used = set(), set()
        cl = list(cur)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                m = _merge(cl[i], cl[j])
                if m is not None:
                    nxt.add(m); used.add(cl[i]); used.add(cl[j])
        primes |= (cur - used); cur = nxt
    return sorted(primes, key=lambda t: tuple(str(v) for v in t))  # deterministic order across runs

def covers(pi, row_bits):
    return all(p == '-' or p == b for p, b in zip(pi, row_bits))

def minimal_covers(on_rows, pis, k, max_models=8):
    on_bits = {r: tuple((r >> (k - 1 - j)) & 1 for j in range(k)) for r in on_rows}
    cov_sets = [frozenset(r for r in on_rows if covers(p, on_bits[r])) for p in pis]
    best = [10 ** 9]; sols = []
    def dfs(unc, chosen):
        if len(chosen) > best[0]: return
        if not unc:
            if len(chosen) < best[0]: best[0] = len(chosen); sols.clear()
            if len(chosen) == best[0] and len(sols) < 200: sols.append(frozenset(chosen))
            return
        r = min(unc)
        cands = [i for i, c in enumerate(cov_sets) if r in c]
        cands.sort(key=lambda i: -len(cov_sets[i] & unc))
        for i in cands:
            dfs(unc - cov_sets[i], chosen + [i])
    dfs(frozenset(on_rows), [])
    uniq = sorted({s for s in sols}, key=lambda s: sorted(s))
    return [[pis[i] for i in sorted(s)] for s in uniq]

def minimise(on_idx, dc_idx, k):
    if len(on_idx) == 0: return []
    pis = prime_implicants(list(on_idx) + list(dc_idx), k)
    # keep only PIs that cover >= 1 ON row
    on_bits = {r: tuple((r >> (k - 1 - j)) & 1 for j in range(k)) for r in on_idx}
    pis = [p for p in pis if any(covers(p, on_bits[r]) for r in on_idx)]
    return minimal_covers(list(on_idx), pis, k)

# ------------------------------------------------------ solution terms / fit
def term_membership(term, M, names):
    mem = np.ones(len(M))
    for j, t in enumerate(term):
        if t == '-': continue
        v = M[names[j]].values
        mem = np.minimum(mem, v if t == 1 else 1 - v)
    return mem

def term_label(term, names, sep=' * '):
    return sep.join((names[j] if t == 1 else '~' + names[j]) for j, t in enumerate(term) if t != '-')

def solution_fit(terms, M, Y, names):
    T = [term_membership(t, M, names) for t in terms]
    U = np.maximum.reduce(T) if T else np.zeros(len(Y))
    out = dict(sol_cons=cons(U, Y), sol_cov=cov(U, Y), sol_pri=pri(U, Y), terms=[])
    for i, t in enumerate(terms):
        others = [T[j] for j in range(len(T)) if j != i]
        Uo = np.maximum.reduce(others) if others else np.zeros(len(Y))
        out['terms'].append(dict(term=t, label=term_label(t, names), cons=cons(T[i], Y), PRI=pri(T[i], Y),
                                 raw_cov=cov(T[i], Y), uniq_cov=out['sol_cov'] - cov(Uo, Y)))
    return out

# ----------------------------------------------------------- intermediate sol
def easy_remainders(rem_idx, on_idx, k, DE):
    """DE[j] in {1: present expected to contribute, 0: absent expected, None: no expectation}.
    A remainder r is an easy counterfactual if some sufficient row s satisfies the directional order."""
    bits = lambda r: tuple((r >> (k - 1 - j)) & 1 for j in range(k))
    ons = [bits(r) for r in on_idx]; keep = []
    for r in rem_idx:
        rb = bits(r)
        for sb in ons:
            ok = True
            for j in range(k):
                if DE[j] == 1:   ok &= rb[j] >= sb[j]
                elif DE[j] == 0: ok &= rb[j] <= sb[j]
                else:            ok &= rb[j] == sb[j]
                if not ok: break
            if ok: keep.append(r); break
    return keep

def solve(M, Y, names, n_cut, incl_cut, pri_cut, DE, tt=None):
    k = len(names)
    tt = truth_table(M, Y, names) if tt is None else tt
    on, rem = code_rows(tt, n_cut, incl_cut, pri_cut)
    on_idx = tt.row[on].tolist(); rem_idx = tt.row[rem].tolist()
    res = dict(tt=tt, on_idx=on_idx, rem_idx=rem_idx, k=k)
    if not on_idx:
        res.update(complex=None, parsimonious=None, intermediate=None); return res
    # complex: remainders treated as 0
    cx = minimise(on_idx, [], k)
    ps = minimise(on_idx, rem_idx, k)
    easy = easy_remainders(rem_idx, on_idx, k, DE)
    im = minimise(on_idx, easy, k)
    def pick(models):                      # model with highest solution coverage, then consistency
        if not models: return None
        fits = [(solution_fit(m, M, Y, names), m) for m in models]
        fits.sort(key=lambda t: (-t[0]['sol_cov'], -t[0]['sol_cons']))
        f, m = fits[0]; f['n_models'] = len(models); return dict(terms=m, fit=f)
    res.update(complex=pick(cx), parsimonious=pick(ps), intermediate=pick(im), n_easy=len(easy))
    return res

def core_peripheral(inter_term, pars_terms):
    """A condition is 'core' if it also appears in a parsimonious term that is a subset of the intermediate term."""
    core = set()
    for p in pars_terms:
        pj = {j: v for j, v in enumerate(p) if v != '-'}
        if all(inter_term[j] == v for j, v in pj.items()):
            core |= set(pj.keys())
    return core
