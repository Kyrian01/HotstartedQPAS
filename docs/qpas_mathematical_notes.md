# Mathematical Notes on the Hot-Started QPAS Implementation

*A companion to Chapter 4 of "Hot-Started Active-Set Method for Regularized Column Generation"*

These notes record the mathematics behind the implementation in `src/qpas/` and
`src/backends/cholmod/`. Where the code departs from the thesis text or adds to it, the
argument for the choice is given here. Each section names the code that realises it.
Numbers marked *measured* come from runs of this code on the generated instances I1–I3
(Apple M-series, Julia 1.13, single thread). They illustrate trends; they are not
benchmarks to cite.

---

## 0. Setting and notation

At column generation iteration $k$ the restricted regularized master problem (thesis eq. 4.6) is

$$
\min_{x\in\mathbb{R}^n}\; f(x) = \frac{1}{2\delta}\,x^\top A^\top A x + \tilde c^\top x
\qquad\text{s.t.}\qquad Cx = \mathbf 1,\quad x \ge 0,
$$

with $\tilde c = c - A^\top(\hat\pi + b/\delta)$, where

| symbol | meaning |
|---|---|
| $A = AX \in \mathbb{R}^{m\times n}$ | coupling coefficients of the generated columns, $a_j$ = column $j$ |
| $c\in\mathbb{R}^n$ | original column costs |
| $C\in\{0,1\}^{p\times n}$ | convexity matrix, $\psi(j)$ = block of column $j$ |
| $b,\ \hat\pi,\ \delta$ | coupling right-hand side, stability center, proximal parameter |
| $[S, W, N]$ | reordering of thesis Thm. 4.2.3; $\mathrm{rep}(j) = S[\psi(j)]$ |
| $z_j = e_j - e_{\mathrm{rep}(j)}$ | nullspace vectors, $j\in N$ (thesis Cor. 4.3.4) |
| $u_j = Az_j = a_j - a_{\mathrm{rep}(j)}$ | columns of $AZ$ |
| $H = (AZ)^\top AZ$, $H_{kj} = u_k^\top u_j$ | unscaled reduced Hessian (thesis eq. 4.31) |
| $\bar\nu$ | average number of nonzeros per column $a_j$ |

---

## 1. The gradient is a reduced cost: working with $\pi$ instead of $g$

**Proposition 1 (gradient–dual identity).** Define
$\pi(x) = \hat\pi + \frac{1}{\delta}(b - Ax)$. Then

$$
\nabla f(x) = c - A^\top \pi(x).
$$

*Proof.*
$\nabla f(x) = \frac1\delta A^\top A x + c - A^\top\hat\pi - \frac1\delta A^\top b
= c - A^\top\big(\hat\pi + \tfrac1\delta (b - Ax)\big)$. $\square$

$\pi(x)$ is the dual estimate that thesis eq. 4.4 recovers *after* the solve. The identity
shows it is available throughout the solve, and it has three consequences.

**(a) The QPAS multipliers are column generation reduced costs.** At a subspace minimizer
$p = 0$, so $h = \frac1\delta A^\top Ap + g = g$. By thesis Thm. 4.3.5,

$$
\mu_b = g_{S[b]}, \qquad
\lambda_j = g_j - \mu_{\psi(j)} = c_j - a_j^\top\pi - \mu_{\psi(j)} \quad (j\in W).
$$

This is exactly the reduced cost of column $j$ that pricing evaluates (thesis eq. 1.25). The
QPAS optimality test "$\lambda_j \ge -\varepsilon$ for all $j\in W$" and the pricing test
"no column with reduced cost $< -\varepsilon_{\text{rc}}$" are therefore the same test, applied
inside and outside the restricted master. To ensure that pricing never regenerates a column
that is already in the master, the code uses
$\varepsilon = 10^{-10}\max(1, \max_j|c_j|) < \varepsilon_{\text{rc}} = 10^{-6}$.
A duplicate-column guard remains as a safety net (`ColumnPool.signatures`).

At a subspace minimizer the reduced system gives $Z^\top g = 0$, i.e. $g_j = g_{\mathrm{rep}(j)}$
for every $j\in N$. All positive-weight columns of a block share the value $\mu_b$: their reduced cost is zero.

**(b) Store an $m$-vector, not an $n$-vector.** The solver keeps $\pi$ (length $m$) and
evaluates $g_j = c_j - a_j^\top\pi$ only where it is needed, in $O(\bar\nu)$ per entry. A
primal step $x \leftarrow x + \alpha p$ changes $\pi$ by

$$
\pi \leftarrow \pi - \frac{\alpha}{\delta}\,Ap, \qquad Ap = \sum_{j\in N} y_j\,u_j = \sum_{k\in N\cup S} p_k\, a_k,
$$

which costs $O((|N| + p)\,\bar\nu + m)$. Keeping $g$ instead (as in the sketch `claude_context/qpas.jl`)
requires $g \leftarrow g + \frac\alpha\delta A^\top(Ap)$ at every step, which is $O(\mathrm{nnz}(A))$
and grows with the number of generated columns.

**(c) The objective needs no extra work.** With $w = \pi - \hat\pi$ in thesis eq. 4.3a,

$$
\text{master objective} = c^\top x + \frac\delta2\|\pi\|^2 - \frac\delta2\|\hat\pi\|^2 .
$$

**(d) $\pi$ is unique at the optimum, even when $x$ is not.** If $x_1 \neq x_2$ are both
optimal with $Ax_1 \neq Ax_2$, then the midpoint $\bar x$ is feasible and, since
$\|\cdot\|^2$ is strictly convex while the linear term is affine,
$f(\bar x) < \frac12(f(x_1)+f(x_2))$, a contradiction. Hence $Ax^\star$, and with it $\pi^\star$,
is unique. The tests use this to compare the QPAS and HiGHS solutions through $\pi$ rather
than through $x$ (`test/test_qpas.jl`).

*Code:* `QPASState.π`, `gradient`, `refresh_duals!`, `take_step!`, `most_negative_multiplier`.

---

## 2. Where gradient entries are needed, and the full-step shortcut

| quantity | indices | when |
|---|---|---|
| $(Z^\top g)_j = g_j - g_{\mathrm{rep}(j)}$ | $j \in N$, plus the $p$ representatives | every direction solve |
| $\lambda_j = g_j - g_{\mathrm{rep}(j)}$ | $j\in W$ | only at a subspace minimizer |
| ratio test, step | support of $p$, which is $N\cup S$ | every step |

So a **step iteration** costs $O((|N|+p)\bar\nu)$ plus one triangular solve pair. It does not
depend on $n$. Only a **multiplier check** touches every column ($O(\mathrm{nnz}(A))$).

**Proposition 2 (full step).** If the ratio test returns $\alpha = 1$, the new iterate minimizes
$f$ over $x + \mathrm{span}(Z)$, so the next direction is $p = 0$.

*Proof.* On the affine subspace $\{x + Zy\}$, $f$ is the convex quadratic
$\phi(y) = f(x) + (Z^\top g)^\top y + \frac1{2\delta}y^\top H y$. Its minimizer solves
$Hy = -\delta Z^\top g$, which is exactly the computed $y$. $\square$

After a full step the code therefore skips the solve and checks the multipliers directly.
On I3 this saves about 40k of 120k solves (*measured*: 40 494 of 78 216 steps were full).

---

## 3. The nullspace basis and $AZ$ are implicit

The ordering $[S,W,N]$ determines $Z$ completely, so neither $Z$ nor $AZ$ is stored. Column
$j$ of $H$ (restricted to $N$) is assembled on demand:

$$
H_{kj} = u_k^\top u_j = a_k^\top v - a_{\mathrm{rep}(k)}^\top v, \qquad v = u_j \text{ (dense scatter)},
\qquad H_{jj} = a_j^\top v - a_{\mathrm{rep}(j)}^\top v .
$$

The products $a_{S[b]}^\top v$ are computed once per block and cached. The cost of one column is

$$
O\big((|N| + p)\,\bar\nu\big),
$$

independent of $n$. The sketch `claude_context/KKTFactor.jl` instead maintains $AZ$ and its row
lists (`AZt_cols`) explicitly. Every working-set change must then update those structures
consistently, with sorted inserts and deletes, and a representative swap rewrites $|\nu|$
columns of $AZ$.

**Representative swaps reduce to recomputation.** Let the representative $q$ of block $b$
block, with replacement $r \in \nu = \{j\in N : \psi(j) = b\}$. For $j\in\nu\setminus\{r\}$,

$$
u_j' = a_j - a_r = (a_j - a_q) - (a_r - a_q) = u_j - u_r ,
$$

so for $k\notin\nu$ and $i, j\in\nu\setminus\{r\}$

$$
H'_{kj} = H_{kj} - H_{kr}, \qquad H'_{ij} = H_{ij} - H_{ir} - H_{rj} + H_{rr},
$$

which is thesis Thm. 4.4.4 / eq. 4.57. The formula needs the old column $H_{:,r}$, which is
not stored either. Assembling $u_j'$ directly costs the same and needs no extra bookkeeping.
The code therefore performs, for each affected column, `rowdel` followed by `rowadd` of the
freshly assembled column (`_swap_representative!`).

**Bound on the nullspace dimension.** $H$ positive definite $\iff$ the $u_j$ ($j\in N$) are
linearly independent, so

$$
|N| \le \operatorname{rank}(A) \le m .
$$

The factor never exceeds an $m\times m$ dense block, no matter how many columns are
generated (*measured* on I3: $m = 360$, final $|N| = 357$).

---

## 4. The factor: fixed size, natural ordering, identity embedding

### 4.1 Construction without ever forming $H$

The factor has the fixed dimension $n_{\max}$ (thesis §4.5.1) and represents

$$
\bar H = \begin{cases} H_{kj} & k, j\in N\\ \delta_{kj} & \text{otherwise.}\end{cases}
$$

It is created as the identity by converting a CHOLMOD *symbolic* factor to a numeric
simplicial $LDL^\top$ (`cholmod_l_change_factor`), which CHOLMOD initialises to $L = D = I$.
From then on, every change is a `rowadd` (index enters $N$) or a `rowdel` (index leaves $N$).
No analysis or factorization of an assembled matrix ever happens. The ordering is the natural
one, so index $k$ of the factor *is* column $k$.

**Lemma 3 (decoupling).** For every inactive $k\notin N$, row and column $k$ of $L$ are $e_k$
and $D_{kk} = 1$. Consequently, for a right-hand side supported on $N$ the solution is
supported on $N$, and its $N$-part is $H^{-1}$ applied to the $N$-part of the right-hand side.

*Proof.* `rowdel(k)` leaves the factorization of the matrix with row and column $k$ replaced
by $e_k$. Because $\bar H$ is block diagonal (up to permutation) between $N$ and its
complement, $LDL^\top$ factorizations without pivoting preserve this structure. $\square$

### 4.2 Sparse right-hand-side solves

Lemma 3 lets the solve be restricted to the pattern $N$. The code calls
`cholmod_l_solve2` with $B_{\text{set}} = N$; CHOLMOD then works only on
$X_{\text{set}} = \operatorname{reach}_L(N)$. The cost is proportional to the number of
entries of $L$ in those columns, independent of $n_{\max}$.

| $n_{\max}$ | $\lvert N\rvert$ | dense solve | sparse solve (*measured*) |
|---|---|---|---|
| 20 000 | 50 | 25.9 µs | 2.7 µs |
| 20 000 | 200 | 48.2 µs | 27.3 µs |
| 5 000 | 200 | 28.6 µs | 26.6 µs |

### 4.3 Cost of `rowadd` and sorted refactorization

With natural ordering, `rowadd(k)` computes row $k$ of $L$ by a triangular solve with the
leading block (indices $< k$). It then applies a rank-one update to the trailing columns
(indices $> k$). If $k$ exceeds every active index, the trailing part is empty and the call
costs a single triangular solve.

Rebuilding the factor from the current set $N$ therefore inserts the indices **in increasing
order** (`refactor!`). Each insertion appends a last row, and the whole rebuild costs about
$\sum_{t=1}^{|N|} t^2 \approx |N|^3/3$ flops, the cost of a dense Cholesky factorization. It
also needs $|N|$ Hessian columns at $O((|N|+p)\bar\nu)$ each.

### 4.4 The structural-zero problem ("fill-in" that is not fill-in)

`rowdel(k)` makes row and column $k$ *numerically* equal to $e_k$, but CHOLMOD keeps the
positions of the removed entries in the sparsity pattern of $L$. In a simplicial factor the
pattern of a column only grows. Over a run the pattern of $L$ therefore becomes the
**union of the patterns of every factor that ever existed**:

$$
\operatorname{pattern}(L_t) \;\supseteq\; \bigcup_{s\le t} \operatorname{pattern}(L_s^{\text{exact}}).
$$

Every solve and every update pays for these stored zeros, because the reach of $N$ includes
indices that are no longer active but are still structurally linked.

*Measured* on I2 after a full run (28k row updates, final $|N| = 215$, $n_{\max} = 20\,000$):

| | stored entries of $L$ | $\lvert\operatorname{reach}(N)\rvert$ | QPAS CG time |
|---|---|---|---|
| never refactored | 438 391 | 915 | 8.86 s |
| refactored every 200 updates | 49 611 | 244 | 1.07 s |

The remedy is periodic refactorization from scratch (§4.3), which resets the pattern to that
of the current $\bar H$. The interval $\tau$ (`QPASOptions.refactor_interval`) trades the
$O(|N|^3)$ rebuild against the growth of stored zeros, which is roughly linear in the number of
updates. The optimum is flat (*measured* total time):

| $\tau$ | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| I2 | 1.26 s | 1.15 s | 1.09 s | 1.08 s | 1.14 s |
| I3 | 8.14 s | 7.23 s | 6.85 s | 6.69 s | 6.96 s |

The default is $\tau = 150$. Because $H$ is nearly dense on $N$ (columns of one block share
their representative, and tasks are banded), a clean factor is itself close to a dense lower
triangle on $N$. A fill-reducing ordering would therefore gain little, which justifies the
natural ordering.

### 4.5 The factor does not depend on $\delta$

Only the right-hand side $-\delta Z^\top g$ contains $\delta$; the factor is of the *unscaled*
$H$. The proximal parameter could therefore be changed between column generation iterations
(for example an adaptive $\delta$ during the tail of §7) at **zero** factorization cost: $x$ stays
feasible, and $\pi$ is recomputed at the start of every solve. The HiGHS masters would have
to re-pass their Hessians instead.

---

## 5. Singular reduced Hessians (thesis §4.4.5, revisited)

Expansions only remove columns of $AZ$, so they cannot create singularity. A reduction adds
$u_r$; with $h = (u_k^\top u_r)_{k\in N}$ the extended matrix is

$$
H^+ = \begin{pmatrix} H & h\\ h^\top & H_{rr}\end{pmatrix},
\qquad \det H^+ = \det H \cdot s,\quad s = H_{rr} - h^\top H^{-1}h,
$$

and $s = 0 \iff u_r \in \operatorname{span}\{u_k : k\in N\}$.

### 5.1 Detection after the update instead of before

Testing $s$ before the `rowadd` costs one extra solve per reduction. The code instead performs
the `rowadd` and then inspects the relative pivots $D_{kk}/H_{kk}$ on $N\cup\{r\}$. The solver
maintains the invariant that every pivot of the factor exceeds `singular_tolerance`
($10^{-10}$). The product of the pivots equals $\det H^+ = \det H\cdot s$, so a vanishing $s$
must show up as a small (or non-positive) pivot. Because of the invariant, that pivot can only
be due to $r$. This detection is a numerical heuristic, not a theorem about floating-point
$LDL^\top$, but it is consistent with the determinant identity, and the debug-mode tests
verify it.

In the rare dependent case the update is undone by refactoring without $r$ (§4.3); this also
discards any large entries produced by dividing by a tiny pivot. Only then is
$t = H^{-1}h$ computed. *Measured* on I3: 38 091 reductions, of which 12 were dependent, so
roughly 38k solves were saved at the cost of 12 refactorizations.

### 5.2 Resolution by a zero-curvature step

**Proposition 4.** Let $x$ be a subspace minimizer for the working set $W$, let $r\in W$ with
$\lambda_r < 0$, and suppose $u_r = \sum_{k\in N} t_k u_k$. Define

$$
d = z_r - \sum_{k\in N} t_k z_k .
$$

Then (i) $Cd = 0$ and $d_j = 0$ for $j\in W\setminus\{r\}$; (ii) $Ad = 0$;
(iii) $\nabla f(x)^\top d = \lambda_r < 0$; and (iv) the ray $x + \alpha d$ leaves the
feasible set at a finite $\alpha$.

*Proof.* (i) holds because every $z_j$ satisfies $Cz_j = 0$ and is supported on $\{j, \mathrm{rep}(j)\}$.
(ii) $Ad = u_r - \sum t_k u_k = 0$.
(iii) At a subspace minimizer $Z^\top g = 0$, i.e. $g^\top z_k = 0$ for $k\in N$, so
$g^\top d = g^\top z_r = g_r - g_{\mathrm{rep}(r)} = \lambda_r$.
(iv) Every block sum of $d$ is zero and $d_r = 1 > 0$, so block $\psi(r)$ contains a
negative component of $d$; $x \ge 0$ then bounds $\alpha$. $\square$

By (ii) and (iii), $f(x+\alpha d) = f(x) + \alpha\lambda_r$: the objective decreases linearly,
with zero curvature. The code steps to the first blocking index $q$ and adds it to $W$
(`_singular_step!`).

**Proposition 5.** After this step, adding $r$ to $N$ gives a nonsingular reduced Hessian.

*Proof.* Let $V = \operatorname{null}(D_{W\setminus\{r\}})$. Since $H$ (on $N$) is positive
definite and $u_r$ is dependent, $\ker A\cap V = \operatorname{span}\{d\}$. The new subspace is
$V' = \operatorname{null}(D_{(W\setminus\{r\})\cup\{q\}}) \subset V$, and $d\notin V'$ because
$d_q \ne 0$. Hence $\ker A\cap V' \subseteq \operatorname{span}\{d\}\cap V' = \{0\}$, i.e. $AZ'$
has full column rank. $\square$

The gradient is unchanged ($Ad = 0$). If the blocking index caused a representative swap, the
new reduced gradient is still zero:
$g_j - g_{r'} = (g_j - g_q) - (g_{r'} - g_q) = 0$. The iterate therefore stays a subspace
minimizer, and the retried reduction proceeds with $p_r > 0$.

**Special case.** If $q$ is the representative of $\psi(r)$, then $r$, which now has positive
weight, simply becomes the new representative. This completes the reduction at once.

**Comparison with the thesis rule.** Thesis §4.4.5 moves $k = \arg\max_{k\in N}|t_k|$ into $W$
whatever its weight. If $x_k > 0$, the working-set constraint $x_k = 0$ is then violated by the
iterate, and the multiplier $\lambda_k$ no longer certifies complementarity: the method can
stop with $x_k > 0$ and $\lambda_k > 0$. Proposition 4 removes this problem. The invariant
$x_W = 0$ holds exactly throughout, and the choice of index comes from a ratio test rather
than a heuristic. Both branches are exercised deterministically in `test/test_qpas.jl`.

---

## 6. Choosing a replacement representative

When the representative $q$ of block $b$ blocks, any $r\in\nu$ is a valid replacement (thesis
Lemma 4.4.1). The code chooses $r = \arg\max_{j\in\nu} x_j$. A swap costs $2|\nu|$ factor
updates (§3); the column furthest from its bound is the least likely to block soon and trigger
another swap. Because representatives carry the largest weights, swaps are rare
(*measured* on I3: 477 swaps among 37 734 expansions).

---

## 7. Column generation level: exact termination and the proximal tail

**Proposition 6 (exact stopping).** Suppose the regularized master is solved to optimality,
$\pi = \hat\pi$, and a complete pricing round finds no column with negative reduced cost. Then
$\lambda$ is optimal for the full (unrestricted) LP master.

*Proof.* $\pi = \hat\pi$ means $w = 0$, so $A\lambda = b$ and $\lambda$ is primal feasible.
The QP optimality conditions give $c_j - a_j^\top\pi - \mu_{\psi(j)} \ge 0$ for the generated
columns, with equality where $\lambda_j > 0$ (§1a). Pricing extends the inequality to all
columns, so $(\pi,\mu)$ is dual feasible for the LP master with complementary slackness. $\square$

This is the stopping rule in `solve_column_generation` (with tolerances). It replaces the
"five consecutive non-improving iterations" heuristic. If pricing finds nothing while
$\pi\ne\hat\pi$, the center is moved to $\pi$ and the master is re-solved without new columns:
this is one proximal-point step on the current restricted model.

**The Lagrangian bound with range rows.** For coupling rows
$\ell \le \sum_k A_k x_k \le u$,

$$
\theta(\pi) = \min_{\ell\le v\le u} \pi^\top v + \sum_k \min_{x_k\in P_k}(c_k - A_k^\top\pi)^\top x_k,
\qquad \min_{\ell_i\le v_i\le u_i}\pi_i v_i =
\begin{cases}\pi_i\ell_i & \pi_i>0\\ \pi_i u_i & \pi_i<0\end{cases}
$$

($-\infty$ if the relevant bound is infinite). For an inequality row this reproduces the
sign condition on $\pi_i$ (`coupling_dual_term`).

**The proximal tail.** Exact termination exposes a phase that heuristic stopping hides. Once
the restricted model contains the optimal columns, $\pi$ still converges to $\hat\pi$ only
linearly under a fixed $\delta$. *Measured* on I2: the stabilized methods add about 920
columns against about 2 400 for LP CG, but then perform about 1 100 center-only iterations.
For the QPAS these are cheap (hot start, no new column), which is why the total time is
still dominated by the column-adding phase. §4.5 shows that an adaptive $\delta$ for this phase
would cost the QPAS nothing in factorization work.

---

## 8. Hot-start invariants (thesis §4.5.2)

| event | ordering | $Z$, $H$, factor | $x$ | $\pi$ |
|---|---|---|---|---|
| new column $j$ | $j\to W$ | unchanged | $x_j = 0$ | unchanged |
| new center $\hat\pi$ | unchanged | unchanged | unchanged | recomputed, $O(\mathrm{nnz}(A))$ |

A new center changes $g$, so the iterate is in general no longer a subspace minimizer. Every
solve therefore starts with a direction computation on the inherited factor.

---

## 9. Cost per operation

| operation | cost | depends on $n$? |
|---|---|---|
| direction ($Z^\top g$, solve, $p = Zy$) | $O((\lvert N\rvert+p)\bar\nu) + O(\mathrm{nnz}(L_{\operatorname{reach}(N)}))$ | no |
| ratio test, step, $\pi$ update | $O((\lvert N\rvert+p)\bar\nu + m)$ | no |
| multiplier check | $O(\mathrm{nnz}(A))$ | yes |
| reduction (Hessian column + `rowadd` + pivot check) | $O((\lvert N\rvert+p)\bar\nu) + O(\mathrm{rowadd}) + O(\lvert N\rvert)$ | no |
| expansion, $q\in N$ | one `rowdel` | no |
| expansion, $q\in S$ | $2\lvert\nu\rvert$ updates + $\lvert\nu\rvert$ Hessian columns | no |
| refactorization | $\approx \lvert N\rvert^3/3 + \lvert N\rvert(\lvert N\rvert+p)\bar\nu$ | no |
| new column / new center | $O(1)$ / $O(\mathrm{nnz}(A))$ | — / yes |

---

## 10. Inequality coupling rows (implemented)

*This section was the original outlook. The extension is now implemented along a slightly
different route (effective reduced Hessian, rank-one updates); see
[`qpas_inequality_extension.md`](qpas_inequality_extension.md).*

For a row $a_i^\top\lambda \ge b_i$ the auxiliary variable cannot be eliminated. With a
surplus $s_i \ge 0$, $w_i = (b_i + s_i - a_i^\top\lambda)/\delta$, so the eliminated QP gains one
column per inequality row, with coefficient $-e_i$, cost $0$ and **no convexity block**. In the
$[S,W,N]$ framework:

- its nullspace vector is $z_j = e_j$ (no representative), so $u_j = -e_i$;
- its gradient is $g_j = 0 - (-e_i)^\top\pi = \pi_i$ (Proposition 1), and its multiplier in $W$
  is $\lambda_j = g_j = \pi_i$ (no $\mu$ term).

The optimality condition $\lambda_j \ge 0$ is then exactly the dual sign condition
$\pi_i \ge 0$ of the stabilized dual (thesis eq. 2.9), and complementarity gives
$s_i > 0 \Rightarrow \pi_i = 0$. The extension therefore needs a "no block" category in the
ordering and $a_{\mathrm{rep}} := 0$ for such columns in the Hessian assembly. The factor, the
updates and the driver stay as they are.
