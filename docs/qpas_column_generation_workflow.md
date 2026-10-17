# The Hot-Started QPAS inside Column Generation: Complete Workflow

*How the code in `src/cg/`, `src/master/qpas_master.jl`, `src/qpas/` and
`src/backends/cholmod/` solves the LP relaxation of the multi-pilot crew scheduling problem,
step by step and with every formula that is evaluated.*

> **Note.** Thesis numbers in this document refer to the earlier thesis version
> ([`Masterthesis_Kyrian_Baumer.pdf`](Masterthesis_Kyrian_Baumer.pdf)). The current thesis
> ([`thesis/`](../thesis/main.pdf)) covers this material in: Chapter 2 (sections 2–3 below:
> Dantzig–Wolfe master, pricing, Lagrangian bound), Chapter 3 (section 4: explicit and
> eliminated regularized master; sections 10–12: center updates, exact termination, §3.4),
> Chapters 4–5 (sections 5–6: optimality conditions, ordering, nullspace basis, reduced system,
> multipliers), Chapters 5–7 (section 7: direction, expansion §6.2, reduction §6.3, singular
> case Ch. 7), §6.1 and §6.4–6.5 (section 8: the fixed-size factor and its updates), §6.6
> (sections 9 and 11: hot start, center-only iterations) and Alg. 6.1 (section 13).

This document follows one run from start to finish. Pricing proposes columns, the proximal
master is solved by the active-set method, the reduced-Hessian factor is updated in place,
the state is carried into the next iteration, and the run stops through the center-only
iterations at the end. Design decisions and measurements that are argued in more depth live
in the companion [`qpas_mathematical_notes.md`](qpas_mathematical_notes.md). This file gives
the whole process in one place.

The description below is for equality coupling rows. Inequality rows (`≥`, `≤`) are handled
by slack variables and an effective reduced Hessian with rank-one updates; the changes to
every step are in [`qpas_inequality_extension.md`](qpas_inequality_extension.md).

---

## Contents

1. [Notation](#1-notation)
2. [The problem and its Dantzig–Wolfe master](#2-the-problem-and-its-dantzigwolfe-master)
3. [Pricing and the Lagrangian bound](#3-pricing-and-the-lagrangian-bound)
4. [Proximal stabilization: from the explicit QP to the eliminated QP](#4-proximal-stabilization-from-the-explicit-qp-to-the-eliminated-qp)
5. [Optimality conditions of the eliminated QP](#5-optimality-conditions-of-the-eliminated-qp)
6. [The active-set framework: ordering, nullspace basis, reduced system](#6-the-active-set-framework-ordering-nullspace-basis-reduced-system)
7. [One QPAS solve, operation by operation](#7-one-qpas-solve-operation-by-operation)
8. [The KKT factor: a fixed-size LDLᵀ updated by row additions and deletions](#8-the-kkt-factor-a-fixed-size-ldlᵀ-updated-by-row-additions-and-deletions)
9. [Hot start: what survives from one column generation iteration to the next](#9-hot-start-what-survives-from-one-column-generation-iteration-to-the-next)
10. [The column generation driver](#10-the-column-generation-driver)
11. [Center-only iterations (no new column, only a new center)](#11-center-only-iterations-no-new-column-only-a-new-center)
12. [Termination and the accuracy it guarantees](#12-termination-and-the-accuracy-it-guarantees)
13. [The complete algorithm in pseudocode](#13-the-complete-algorithm-in-pseudocode)
14. [Cost of every operation and map to the code](#14-cost-of-every-operation-and-map-to-the-code)

---

## 1. Notation

| symbol | meaning | code |
|---|---|---|
| $m$ | number of coupling rows (tasks) | `pool.n_rows` |
| $P$ | number of blocks (pilots) = number of convexity rows | `pool.n_blocks` |
| $n$ | number of generated columns so far | `pool.n` |
| $n_{\max}$ | column capacity, fixed for the whole run | `pool.capacity`, `max_columns` |
| $a_j\in\mathbb R^m$ | coupling coefficients of column $j$ | `column_rows/vals(pool, j)` |
| $c_j$ | cost of column $j$ | `pool.cost[j]` |
| $\psi(j)\in\{1,\dots,P\}$ | block (pilot) of column $j$ | `pool.block[j]` |
| $A = [a_1,\dots,a_n]$ | coupling matrix of the restricted master ("$AX$" in the thesis) | |
| $C\in\{0,1\}^{P\times n}$ | convexity matrix, $C_{bj} = 1 \iff \psi(j) = b$ | |
| $b\in\mathbb R^m$ | coupling right-hand side | `rhs`, `task_lower` |
| $x\in\mathbb R^n$ | column weights (the master variables, "$\lambda$" in the thesis) | `QPASState.x` |
| $\pi\in\mathbb R^m$ | coupling duals | `QPASState.π` |
| $\mu\in\mathbb R^P$ | convexity duals | `QPASState.μ` |
| $\hat\pi\in\mathbb R^m$ | stability center | `QPASState.center` |
| $\delta>0$ | proximal parameter (larger = stronger stabilization) | `delta` |
| $\lambda_j$ | multiplier of the bound $x_j\ge 0$ for $j$ in the working set | |
| $S, W, N$ | representatives, working set, nullspace set | `ColumnOrdering` |
| $\mathrm{rep}(j) = S[\psi(j)]$ | representative of column $j$'s block | `representative` |
| $p\in\mathbb R^n$ | primal search direction | `QPASState.p` |
| $\bar\nu$ | average number of nonzeros of a column $a_j$ | |
| $\lVert\cdot\rVert$ | Euclidean norm unless marked otherwise | |

---

## 2. The problem and its Dantzig–Wolfe master

### 2.1 The edge formulation

The instance is a time-ordered DAG $G = (V, E)$ with a source and a sink. Pilot $k$ chooses
one source–sink path, encoded by $\xi_k\in\{0,1\}^{|E|}$ with flow conservation
$F\xi_k = s$ ($F$ the node–edge incidence matrix, $s$ the supply vector). Edge $e$ costs
pilot $k$ the amount $c_{ek}$. The task matrix $A_{\text{task}}\in\mathbb R^{m\times|E|}$
records which tasks an edge covers, and every task must be covered:

$$
\min_{\xi_1,\dots,\xi_P}\;\sum_{k=1}^P c_k^\top\xi_k
\quad\text{s.t.}\quad
\sum_{k=1}^P A_{\text{task}}\,\xi_k = b,\qquad
F\xi_k = s,\;\; \xi_k\ge 0\quad(k = 1,\dots,P).
$$

The LP relaxation of this model is what `solve_exact` solves directly (thesis eq. 5.2). It is
the reference optimum for every column generation run.

### 2.2 Dantzig–Wolfe reformulation

Each polytope $\mathcal P_k = \{\xi\ge 0: F\xi = s\}$ is a path polytope whose vertices are
the source–sink paths. Writing $\xi_k$ as a convex combination of paths gives one column per
(pilot, path) pair:

$$
a_j = A_{\text{task}}\,\xi^{(j)}, \qquad c_j = c_{\psi(j)}^\top\,\xi^{(j)} .
$$

The **full master problem** is

$$
\min_{x\ge 0}\; c^\top x \quad\text{s.t.}\quad Ax = b\;\;(\text{dual } \pi),\qquad Cx = \mathbf 1\;\;(\text{dual } \mu).
$$

Its LP dual is

$$
\max_{\pi,\mu}\; b^\top\pi + \mathbf 1^\top\mu \quad\text{s.t.}\quad
c_j - a_j^\top\pi - \mu_{\psi(j)} \ge 0\quad\text{for all columns } j .
$$

The **restricted master problem** contains only the $n$ columns generated so far.
Column generation alternates between solving the restricted master and asking the pricing
problem for a column whose dual constraint is violated.

---

## 3. Pricing and the Lagrangian bound

### 3.1 Reduced cost

Given duals $(\pi,\mu)$, the **reduced cost** of a column is

$$
\bar c_j = c_j - a_j^\top\pi - \mu_{\psi(j)} .
$$

For pilot $k$ the most negative one over *all* paths is

$$
\min_{\xi\in\mathcal P_k}\big(c_k - A_{\text{task}}^\top\pi\big)^\top\xi \;-\;\mu_k ,
$$

a shortest-path problem with edge weights $c_{ek} - (A_{\text{task}}^\top\pi)_e$. These
weights can be negative, but the graph is a DAG numbered topologically, so one forward pass
of dynamic programming solves it exactly in $O(|V| + |E|)$:

$$
d(\text{source}) = 0, \qquad d(h) = \min_{e = (v,h)}\; d(v) + c_{ek} - (A_{\text{task}}^\top\pi)_e ,
$$

taken over the nodes $v$ in increasing order (`price_block!`). The path is recovered from
predecessor pointers. The new column is $a = A_{\text{task}}\xi$ and $c = c_k^\top\xi$
(`path_coverage!`).

A column is **improving** if $d(\text{sink}) - \mu_k < -\varepsilon_{\text{rc}}$, with
$\varepsilon_{\text{rc}} = 10^{-6}$ (`reduced_cost_tolerance`).

### 3.2 Pricing rules

* `first_improving` (default): the pilots are visited in a random order (seeded RNG), and
  the first improving column is added. If no pilot gives one, all $P$ pilots were priced.
* `all_blocks`: every pilot is priced, and each improving one contributes one column.

A column that is already in the pool is never added twice (`contains_column`, hash of
pilot and path). With the QPAS tolerances (§5.3) this guard should never trigger.

### 3.3 The Lagrangian bound

Dualizing the coupling rows with any $\pi$ gives a lower bound on the LP optimum:

$$
\theta(\pi) = b^\top\pi + \sum_{k=1}^P\;\min_{\xi\in\mathcal P_k}\big(c_k - A_{\text{task}}^\top\pi\big)^\top\xi
\;\le\; z_{\text{LP}} .
$$

It is available only when every pilot was priced in the same round, and the driver records
it whenever that happens (`bound = θ` if `priced_all`). For range rows
$\ell\le\sum_k A_{\text{task}}\xi_k\le u$ the term $b^\top\pi$ becomes
$\sum_i\min_{\ell_i\le v_i\le u_i}\pi_i v_i$ (`coupling_dual_term`). The QPAS handles
equality rows only, so for it this is $b^\top\pi$.

---

## 4. Proximal stabilization: from the explicit QP to the eliminated QP

### 4.1 Why stabilize

The plain LP master gives extreme-point duals that jump between iterations. Proximal
stabilization looks for a dual point that improves the restricted dual while staying close
to a **center** $\hat\pi$:

$$
\max_{\pi,\mu}\; b^\top\pi + \mathbf 1^\top\mu - \frac{\delta}{2}\lVert\pi-\hat\pi\rVert^2
\quad\text{s.t.}\quad c_j - a_j^\top\pi - \mu_{\psi(j)}\ge 0\quad(j = 1,\dots,n).
\tag{D$_\delta$}
$$

This is one step of the proximal point method on the restricted dual function.

### 4.2 The explicit primal QP (thesis eq. 4.3)

The primal counterpart of (D$_\delta$) has an auxiliary vector $w\in\mathbb R^m$:

$$
\min_{x\ge 0,\,w}\; c^\top x + \frac\delta2\lVert w + \hat\pi\rVert^2 - \frac\delta2\lVert\hat\pi\rVert^2
\quad\text{s.t.}\quad Ax + \delta w = b\;\;(\pi),\qquad Cx = \mathbf 1\;\;(\mu).
\tag{P$_\delta$}
$$

*Check of the duality.* The Lagrangian is
$\mathcal L = c^\top x + \frac\delta2\lVert w+\hat\pi\rVert^2 - \frac\delta2\lVert\hat\pi\rVert^2
+ \pi^\top(b - Ax - \delta w) + \mu^\top(\mathbf 1 - Cx)$.
Stationarity in $w$ gives $\delta(w + \hat\pi) - \delta\pi = 0$, i.e.

$$
\boxed{\;\pi = \hat\pi + w\;}
\tag{4.1}
$$

Substituting back,
$\frac\delta2\lVert\pi\rVert^2 - \frac\delta2\lVert\hat\pi\rVert^2 - \delta\pi^\top(\pi - \hat\pi)
= -\frac\delta2\lVert\pi - \hat\pi\rVert^2$, and minimizing over $x\ge 0$ requires
$c - A^\top\pi - C^\top\mu\ge 0$. This is exactly (D$_\delta$).

`HighsExplicitQPMaster` solves (P$_\delta$) as written. Its Hessian is $\delta I$ on $w$
and zero on $x$.

### 4.3 Eliminating $w$ (thesis eq. 4.5/4.6)

With equality coupling rows, $w = (b - Ax)/\delta$ can be substituted. Write $r = b - Ax$
(the coupling residual, i.e. the infeasibility of $x$ in the unstabilized master). Then

$$
c^\top x + \frac{1}{2\delta}\lVert r\rVert^2 + \hat\pi^\top r
= \frac{1}{2\delta}x^\top A^\top A x + \underbrace{\big(c - A^\top(\hat\pi + b/\delta)\big)^\top}_{\tilde c^\top} x
\;+\; \underbrace{\frac1{2\delta}\lVert b\rVert^2 + \hat\pi^\top b}_{\text{constant}} .
$$

Dropping the constant gives the QP that the QPAS solves:

$$
\boxed{\;
\min_{x\in\mathbb R^n}\; f(x) = \frac{1}{2\delta}x^\top A^\top A\,x + \tilde c^\top x
\quad\text{s.t.}\quad Cx = \mathbf 1,\;\; x\ge 0,
\qquad \tilde c = c - A^\top\Big(\hat\pi + \frac b\delta\Big).
\;}
\tag{QP}
$$

Its Hessian is $G = A^\top A/\delta$, positive *semi*definite (and in general singular,
because $n\gg m$). `HighsEliminatedQPMaster` solves the same QP with HiGHS.

### 4.4 The dual vector is a function of $x$

Equation (4.1) with $w = r/\delta$ gives

$$
\boxed{\;\pi(x) = \hat\pi + \frac{1}{\delta}\,(b - Ax)\;}
\tag{4.2}
$$

and the gradient of (QP) is

$$
\nabla f(x) = \frac1\delta A^\top A x + c - A^\top\hat\pi - \frac1\delta A^\top b
= c - A^\top\Big(\hat\pi + \frac{b - Ax}{\delta}\Big)
= \boxed{\;c - A^\top\pi(x)\;}
\tag{4.3}
$$

So **$g_j = c_j - a_j^\top\pi$ is the reduced cost of column $j$ without the convexity
dual**, at every iterate, not only at the optimum. The solver therefore stores the
$m$-vector $\pi$ instead of the $n$-vector $g$, and computes $g_j$ only where it is needed
(`gradient(st, j)`, cost $O(\bar\nu)$).

### 4.5 The objective value

With $\pi - \hat\pi = r/\delta$:
$\hat\pi^\top r + \frac{1}{2\delta}\lVert r\rVert^2 = \frac\delta2\lVert\pi\rVert^2 - \frac\delta2\lVert\hat\pi\rVert^2$,
so the stabilized master objective reported by every stabilized master is

$$
\Phi(x) = c^\top x + \frac\delta2\lVert\pi\rVert^2 - \frac\delta2\lVert\hat\pi\rVert^2
= f(x) + \frac1{2\delta}\lVert b\rVert^2 + \hat\pi^\top b
\tag{4.4}
$$

(`stabilized_objective`). It needs only $c^\top x$ and the stored $\pi$.

---

## 5. Optimality conditions of the eliminated QP

### 5.1 KKT conditions

With multipliers $\mu\in\mathbb R^P$ for $Cx = \mathbf 1$ and $\lambda\in\mathbb R^n$ for
$x\ge 0$, a point $x$ is optimal for (QP) if and only if (the QP is convex)

$$
\begin{aligned}
g := \nabla f(x) &= C^\top\mu + \lambda &&\text{(stationarity)}\\
Cx &= \mathbf 1,\quad x\ge 0 &&\text{(primal feasibility)}\\
\lambda &\ge 0 &&\text{(dual feasibility)}\\
\lambda_j x_j &= 0 &&\text{(complementarity)}.
\end{aligned}
$$

Componentwise, stationarity reads $g_j = \mu_{\psi(j)} + \lambda_j$, i.e.

$$
\lambda_j = g_j - \mu_{\psi(j)} = c_j - a_j^\top\pi - \mu_{\psi(j)} = \bar c_j .
\tag{5.1}
$$

**The bound multipliers of the QP are the column generation reduced costs.** Optimality of
(QP) means: every generated column has $\bar c_j\ge 0$, and every column with $x_j>0$ has
$\bar c_j = 0$. These are the same conditions that pricing tests for the columns that are
*not* generated yet.

### 5.2 The equality-constrained subproblem

An active-set method fixes a **working set** $W$ of indices with $x_j = 0$ and solves

$$
\min_x\; f(x)\quad\text{s.t.}\quad Cx = \mathbf 1,\quad x_j = 0\;\;(j\in W).
\tag{EQP$_W$}
$$

Its KKT system (thesis eq. 4.17), for a step $p$ from a feasible $x$, is

$$
\begin{pmatrix} G & C^\top & E_W\\ C & 0 & 0\\ E_W^\top & 0 & 0\end{pmatrix}
\begin{pmatrix} p\\ -\mu\\ -\lambda_W\end{pmatrix}
=
\begin{pmatrix} -g\\ 0\\ 0\end{pmatrix},
\qquad E_W = [e_j]_{j\in W},
$$

where the zero right-hand sides express $Cp = 0$ and $p_W = 0$, so feasibility is kept.
This system has dimension $n + P + |W|$, is singular in its $G$-block and grows with every
column. **It is never formed.** Section 6 shows how it reduces to a small positive definite
system.

### 5.3 Tolerances

The QPAS accepts $x$ when $\lambda_j\ge -\varepsilon$ for all $j\in W$, with

$$
\varepsilon = \texttt{dual\_tolerance}\cdot\max\big(1,\max_j|c_j|\big) = 10^{-10}\max(1,\max_j|c_j|) .
$$

Pricing requires $\bar c < -10^{-6}$ to add a column. Since $\varepsilon\ll 10^{-6}$, a
column already in the master can never be "found" again by pricing.

---

## 6. The active-set framework: ordering, nullspace basis, reduced system

### 6.1 The ordering $[S, W, N]$ (thesis Thm. 4.2.3)

Every registered column is in exactly one of three sets:

* **$S$**: one **representative** per block, $S[b] = \mathrm{rep}$ of block $b$. The solver
  maintains $x_{S[b]} > 0$ (in practice, as the column with large weight).
* **$W$**: the **working set**, columns fixed at $x_j = 0$.
* **$N$**: the **nullspace set**, the remaining free columns.

`ColumnOrdering` stores `rep[b]`, the index sets `W` and `N` (O(1) insert/delete), and
`set_of[j]` for O(1) membership tests.

### 6.2 The nullspace basis is implicit (thesis Cor. 4.3.4)

The feasible directions of (EQP$_W$) are
$\{p: Cp = 0,\; p_W = 0\}$, meaning that every block sums to zero and $W$ stays at zero. A basis
of this space is

$$
z_j = e_j - e_{\mathrm{rep}(j)}, \qquad j\in N .
\tag{6.1}
$$

*Proof.* Each $z_j$ has block sum $1 - 1 = 0$ and no entry in $W$. The $z_j$ are linearly
independent because $z_j$ is the only basis vector with a nonzero in position $j\in N$.
The dimension is $n - P - |W| = |N|$ ($P$ block constraints, $|W|$ bound constraints, all
independent because each block keeps its representative). $\square$

So $Z = [z_j]_{j\in N}$ is determined by the ordering alone. It is **never stored**, and
neither is

$$
AZ = [u_j]_{j\in N}, \qquad u_j = A z_j = a_j - a_{\mathrm{rep}(j)} .
\tag{6.2}
$$

A direction $p = Zy$ has the components

$$
p_j = y_j\;(j\in N), \qquad p_{S[b]} = -\sum_{j\in N,\,\psi(j) = b} y_j, \qquad p_W = 0,
\tag{6.3}
$$

so its support is $N\cup S$.

### 6.3 Reduced gradient and reduced Hessian

$$
(Z^\top g)_j = g_j - g_{\mathrm{rep}(j)}, \qquad
Z^\top G Z = \frac1\delta (AZ)^\top(AZ) = \frac1\delta H, \qquad
H_{kj} = u_k^\top u_j .
\tag{6.4}
$$

$H$ is the **unscaled reduced Hessian** (thesis eq. 4.31). It is positive definite if and
only if the $u_j$ ($j\in N$) are linearly independent, which forces

$$
|N| \le \operatorname{rank}(A) \le m .
$$

The linear system the QPAS solves never exceeds $m\times m$, however many columns are
generated.

### 6.4 The reduced KKT system

With $p = Zy$, the first block row of the KKT system, multiplied by $Z^\top$ (which
annihilates $C^\top\mu$ and $E_W\lambda_W$), gives

$$
Z^\top G Z\, y = -Z^\top g
\quad\Longleftrightarrow\quad
\boxed{\;H\,y = -\delta\, Z^\top g\;}
\tag{6.5}
$$

(thesis eq. 4.32). $\delta$ appears only on the right-hand side, so **the factor of $H$
does not depend on $\delta$**.

$y$ minimizes $f$ on the affine subspace $x + \operatorname{span}(Z)$:
$f(x + Zy) = f(x) + (Z^\top g)^\top y + \frac{1}{2\delta}y^\top H y$, whose minimizer is
(6.5).

### 6.5 Multipliers at a subspace minimizer (thesis Thm. 4.3.5)

$x$ is a **subspace minimizer** when $p = 0$, i.e. $Z^\top g = 0$:

$$
g_j = g_{\mathrm{rep}(j)} \qquad\text{for all } j\in N .
$$

The remaining KKT rows then give, for each block $b$,

$$
\mu_b = g_{S[b]} = c_{S[b]} - a_{S[b]}^\top\pi ,
\tag{6.6}
$$

$$
\lambda_j = g_j - \mu_{\psi(j)} \quad (j\in W) .
\tag{6.7}
$$

So every column in $N\cup S$ has reduced cost exactly $0$ (by (6.6) and $Z^\top g = 0$), and
every column in $W$ has reduced cost $\lambda_j$. The solver computes $\mu$ by (6.6)
(`update_convexity_duals!`) and $\lambda$ by (6.7) (`most_negative_multiplier`).

---

## 7. One QPAS solve, operation by operation

`qpas_solve!` is thesis Alg. 4.3 with efficient working-set updates. Every solve starts from
the iterate, ordering and factor left by the previous solve (§9).

### 7.0 Start of a solve: recompute the duals

$$
\pi \leftarrow \hat\pi + \frac{1}{\delta}\Big(b - \sum_{j: x_j\ne 0} x_j a_j\Big)
\qquad(\texttt{refresh\_duals!},\; O(\operatorname{nnz}(A)))
$$

This picks up a new center and new columns, and removes any drift accumulated by the
incremental updates in §7.3.

### 7.1 Direction

If the iterate is not known to be a subspace minimizer:

1. $\mu_b = g_{S[b]}$ for every block (6.6). This costs $P$ column dot products.
2. Right-hand side on $N$:
   $\;\mathrm{rhs}_k = -\delta\,(g_k - \mu_{\psi(k)}) = -\delta (Z^\top g)_k,\;k\in N$.
3. Solve $\bar H y = \mathrm{rhs}$ with the factor, restricted to the pattern $N$ (§8.4).
4. Expand $p = Zy$ by (6.3): $p_k = y_k$, $p_{\mathrm{rep}(k)} \mathrel{-}= y_k$.
5. $\lVert p\rVert_\infty$ over $N\cup S$.

If $\lVert p\rVert_\infty\le 10^{-10}$ (`step_tolerance`), the iterate is a subspace
minimizer: go to §7.2. Otherwise go to §7.3.

Cost: $O((|N| + P)\bar\nu)$ for the gradient entries, plus one sparse triangular solve
pair. It does not depend on $n$.

### 7.2 Subspace minimizer: multiplier check

$$
\lambda_{\min} = \min_{j\in W}\big(g_j - \mu_{\psi(j)}\big), \qquad r = \arg\min .
$$

* If $W = \emptyset$ or $\lambda_{\min}\ge -\varepsilon$: **the solve is optimal** (KKT of
  §5.1 hold). The solver returns $x$, $\pi$, $\mu$.
* Otherwise release $r$ from $W$ (**reduction**, §7.5). The new iterate is no longer a
  subspace minimizer, so the next iteration computes a direction.

This check is the only operation whose cost grows with $n$: $O(\operatorname{nnz}(A))$.

### 7.3 Ratio test and step

The step keeps $x\ge 0$. Only components with $p_k<0$ can block:

$$
\alpha = \min\Big(1,\;\min_{k\in N\cup S,\; p_k < -10^{-14}} \frac{x_k}{-p_k}\Big),
\qquad q = \text{the minimizing index, or } 0 \text{ if } \alpha = 1 .
$$

Then

$$
x \leftarrow \max(x + \alpha p,\,0) \;\text{on } N\cup S,
\qquad
\pi \leftarrow \pi - \frac\alpha\delta\,Ap,
\qquad
Ap = \sum_{k\in N\cup S} p_k\,a_k .
$$

The $\pi$ update follows from (4.2), since $\pi$ is affine in $x$. It costs
$O((|N| + P)\bar\nu + m)$, compared with $O(\operatorname{nnz}(A))$ for updating a stored
$g$.

* **$\alpha = 1$ (full step):** the new iterate minimizes $f$ on $x+\operatorname{span}(Z)$
  (§6.4), so the next direction is zero. The solver skips the direction solve and goes
  directly to the multiplier check (§7.2).
* **$\alpha<1$ (blocked step):** $x_q$ has hit zero, so $q$ joins the working set
  (**expansion**, §7.4).

### 7.4 Expansion: $q$ enters $W$ (thesis §4.4.1)

**Case $q\in N$.** The basis vector $z_q$ disappears, and with it row and column $q$ of
$H$:

$$
\texttt{rowdel}(q), \qquad N\leftarrow N\setminus\{q\},\qquad W\leftarrow W\cup\{q\},\qquad x_q = 0 .
$$

One factor update. No other $z_j$ changes.

**Case $q\in S$ (the representative of block $b$ blocks).** Block $b$ needs a new
representative. Let $\nu = \{j\in N: \psi(j) = b\}$, which is nonempty because $x_q$ could only
decrease if some $y_j\ne 0$, $j\in\nu$. The code chooses

$$
r = \arg\max_{j\in\nu} x_j
$$

(the column furthest from its bound, the least likely to block next). Every $z_j$,
$j\in\nu\setminus\{r\}$, changes from $e_j - e_q$ to $e_j - e_r$, so

$$
u_j' = a_j - a_r = u_j - u_r .
$$

By thesis Thm. 4.4.4 the new Hessian entries are

$$
H'_{kj} = H_{kj} - H_{kr}\;(k\notin\nu),\qquad
H'_{ij} = H_{ij} - H_{ir} - H_{rj} + H_{rr}\;(i,j\in\nu\setminus\{r\}).
$$

The old column $H_{:,r}$ is not stored, so the code instead rebuilds the affected rows
directly from $u_j'$, which costs the same:

1. `rowdel(j)` for every $j\in\nu$ (including $r$),
2. make $r$ the representative ($r$ moves from $N$ to $S$),
3. `rowadd(j)` with the freshly assembled column of $u_j' = a_j - a_r$ for every
   $j\in\nu\setminus\{r\}$,
4. move $q$ to $W$, $x_q = 0$.

That is $2|\nu| - 1$ factor updates and $|\nu| - 1$ Hessian columns.

**After every expansion** the factor is rebuilt (§8.5) if the refactorization interval is
due or if a pivot has degraded below `singular_tolerance` (`_check_factor!`).

### 7.5 Reduction: $r$ leaves $W$ (thesis §4.4.2)

$r$ has $\lambda_r<0$. Moving it into $N$ adds the basis vector $z_r$, so $H$ grows by one
row and column:

$$
h_k = u_k^\top u_r\;(k\in N),\qquad H_{rr} = \lVert u_r\rVert^2,\qquad
H^+ = \begin{pmatrix} H & h\\ h^\top & H_{rr}\end{pmatrix}.
$$

**Assembling the Hessian column** (`hessian_column!`). Scatter $u = a_r - a_{\mathrm{rep}(r)}$
into a dense $m$-vector, then

$$
h_k = a_k^\top u - a_{\mathrm{rep}(k)}^\top u ,
$$

where $a_{S[b]}^\top u$ is computed once per block and cached. Cost
$O((|N| + P)\bar\nu)$. Only nonzero $h_k$ are passed on.

**Update and singularity check.** `rowadd(r, [h; H_rr])`. The block determinant identity

$$
\det H^+ = \det H\cdot s,\qquad s = H_{rr} - h^\top H^{-1}h
$$

shows that $H^+$ is singular exactly when $s = 0$, i.e. when
$u_r\in\operatorname{span}\{u_k: k\in N\}$. The solver keeps every scaled pivot
$D_{kk}/\lVert u_k\rVert^2$ of the factor above `singular_tolerance`, so a vanishing $s$ shows
up as a small pivot right after the update. The code checks

$$
\min_{k\in N\cup\{r\}} \frac{D_{kk}}{\lVert a_k - a_{\mathrm{rep}(k)}\rVert^2} > 10^{-10}
$$

(`min_pivot_ratio`, with the scale $\lVert u_k\rVert^2$ stored in `hdiag` at activation). With
equality rows only this scale is $H_{kk}$; with inequality rows it is the norm over all rows,
not the diagonal of $H_{\mathrm{eff}}$, so that a column whose masked part vanishes is still
detected.

* **Independent (the normal case):** the reduction is complete, and the next iteration
  computes a direction. Because $\lambda_r<0$, the new direction has $p_r > 0$ (thesis
  §4.4.2), so the step is a descent step and $x_r$ becomes positive.
* **Dependent ($s\approx 0$):** undo by moving $r$ back to $W$ and refactoring (§8.5),
  which also discards any large entries created by the tiny pivot. Then take a
  **zero-curvature step**:

  $$
  t = H^{-1}h, \qquad d = z_r - \sum_{k\in N} t_k z_k .
  $$

  Then $Ad = u_r - \sum t_k u_k = 0$ (no curvature), $Cd = 0$, and
  $g^\top d = g^\top z_r = g_r - g_{\mathrm{rep}(r)} = \lambda_r < 0$, because $Z^\top g = 0$ at a
  subspace minimizer. So $f(x + \alpha d) = f(x) + \alpha\lambda_r$ decreases linearly.
  The block of $r$ sums to zero in $d$ and $d_r = 1$, so some component is negative, and the ratio
  test (with $\alpha_{\max}=\infty$, including index $r$) returns a finite $\alpha$ and a
  blocking index $q$. The code steps there (the $\pi$ update is exactly zero, because
  $Ad = 0$) and adds $q$ to $W$. After that, $u_r$ is independent of the remaining columns
  (proof in the notes, Prop. 5), and the reduction is retried. If $q$ is $r$'s own
  representative, $r$ simply becomes the new representative, which completes the reduction
  at once.

After a successful reduction the refactorization interval is checked (`_check_refactor_interval!`).

### 7.6 Loop summary

```
refresh π from x and π̂
at_min ← false
repeat
    if not at_min:
        compute p = Z y with H y = −δ Zᵀg               (§7.1)
        if ‖p‖∞ ≤ tol: at_min ← true
    if at_min:
        λ_min, r ← min_{j∈W} g_j − μ_ψ(j)               (§7.2)
        if λ_min ≥ −ε: return OPTIMAL
        release r into N (rowadd + pivot check)         (§7.5)
        at_min ← false
    else:
        α, q ← ratio test; x += αp; π −= (α/δ)Ap        (§7.3)
        if q = 0: at_min ← true                         (full step)
        else: add q to W (rowdel or representative swap) (§7.4)
```

Each iteration either strictly decreases $f$ (a step with $\alpha>0$), adds a constraint to $W$
(at most $|N|$ times in a row before a subspace minimizer is reached, since $N$ shrinks), or
releases one with negative multiplier. This is the standard finite-termination argument of
primal active-set methods, under nondegeneracy.

---

## 8. The KKT factor: a fixed-size LDLᵀ updated by row additions and deletions

### 8.1 Fixed size and identity embedding (thesis §4.5.1)

CHOLMOD holds an $n_{\max}\times n_{\max}$ simplicial $LDL^\top$ factorization of

$$
\bar H_{kj} = \begin{cases} u_k^\top u_j & k, j\in N,\\ 1 & k = j\notin N,\\ 0 & \text{otherwise.}\end{cases}
$$

Index $k$ of the factor *is* column $k$ of the pool, because the ordering is natural and
unpermuted. A new column never resizes anything. The factor starts as $L = D = I$, built by
converting a symbolic factor to a numeric one (`_identity_factor`). From then on it only
changes through `rowadd` and `rowdel`.

**Decoupling.** For an inactive $k$, row and column $k$ of $L$ are $e_k$ and $D_{kk} = 1$.
So for a right-hand side supported on $N$, $\bar H^{-1}$ acts as $H^{-1}$ on $N$ and as the
identity elsewhere.

### 8.2 `rowadd(k, h)`: an index enters $N$

Partition the indices into $1 = \{<k\}$, $k$, $3 = \{>k\}$. Before the update row $k$ is
$e_k$:

$$
L = \begin{pmatrix} L_{11} & & \\ 0 & 1 & \\ L_{31} & 0 & L_{33}\end{pmatrix},\quad
D = \operatorname{diag}(D_1, 1, D_3).
$$

The new matrix has row/column $k$ equal to $(h_1, h_{kk}, h_3)$. The updated factor is

$$
\begin{aligned}
\ell_{k1} &= D_1^{-1}L_{11}^{-1}h_1 &&\text{(triangular solve with the leading block)}\\
d_k &= h_{kk} - \ell_{k1}^\top D_1\,\ell_{k1} &&\text{(new pivot)}\\
\ell_{3k} &= \big(h_3 - L_{31}D_1\ell_{k1}\big)/d_k &&\text{(new column below $k$)}\\
L_{33}'D_3'L_{33}'^\top &= L_{33}D_3L_{33}^\top - d_k\,\ell_{3k}\ell_{3k}^\top &&\text{(rank-one downdate of the trailing block).}
\end{aligned}
$$

The pivot $d_k$ is exactly the Schur complement $s$ of §7.5 when $k$ is the last active
index. In general $\prod_{j\in N\cup\{k\}} D_{jj} = \det H^+$, which is why the pivot check
detects dependence.

**If $k$ is larger than every active index**, block 3 has no coupling to $N$, the
downdate is empty, and `rowadd` costs one triangular solve with the active part, about
$|N|^2$ flops.

### 8.3 `rowdel(k)`: an index leaves $N$

This is the reverse operation. Row/column $k$ is reset to $e_k$:

$$
L_{33}'D_3'L_{33}'^\top = L_{33}D_3L_{33}^\top + d_k\,\ell_{3k}\ell_{3k}^\top
\quad\text{(rank-one update)},\qquad \ell_{k1} = 0,\;\ell_{3k} = 0,\; d_k = 1 .
$$

CHOLMOD zeroes the entries numerically but keeps their positions in the sparsity pattern.
§8.5 explains why that matters.

### 8.4 Sparse right-hand-side solves

The right-hand sides $-\delta Z^\top g$ (direction) and $h$ (singular step) are supported
on $N$. `cholmod_l_solve2` with $B_{\text{set}} = N$ works only on
$\operatorname{reach}_L(N)$, the indices reachable from $N$ in the graph of $L$. By the
decoupling property, the result on $N$ equals $H^{-1}$ applied to the right-hand side.
The cost is proportional to the entries of $L$ in those columns and does not depend on
$n_{\max}$ (`sparse_solves = true`).

### 8.5 Periodic refactorization

A simplicial pattern only grows, so after many `rowdel`s the stored pattern is the union of
all past factors. These structural zeros enlarge $\operatorname{reach}_L(N)$ and slow every
solve and update. Every $\tau$ row updates ($\tau = 150$, `refactor_interval`), and whenever
a pivot has degraded, the factor is rebuilt from scratch (`refactor!`):

1. reset to $L = D = I$,
2. for $j\in N$ **in increasing order**: assemble column $j$ of $H$ against the indices
   already re-added, and `rowadd(j)`.

Each insertion then appends a last row, so it costs one triangular solve and no downdate.
The whole rebuild costs about $|N|^3/3$ flops (a dense Cholesky factorization) plus $|N|$
Hessian columns.

### 8.6 Invariant

Between operations the solver maintains:

$$
\bar H = L D L^\top \text{ on } N,\qquad
D_{kk} > 10^{-10}\,\lVert a_k - a_{\mathrm{rep}(k)}\rVert^2\;\;(k\in N),\qquad
L_{k,:} = e_k^\top,\; D_{kk} = 1\;\;(k\notin N).
$$

In debug mode `verify_kkt` checks all of this after every iteration, against a from-scratch
$(AZ)^\top(AZ)$.

---

## 9. Hot start: what survives from one column generation iteration to the next

`QPASMaster` keeps one `QPASState` for the whole run. Between two master solves only two
things can change, and neither invalidates anything expensive.

### 9.1 New columns (`register_columns!`)

A new column $j$ enters **$W$ with $x_j = 0$**.

* $x$ stays feasible: $Cx = \mathbf 1$ still holds, since the new entry is zero.
* $S$ and $N$ are unchanged, so $Z$, $AZ$, $H$ **and the factor are unchanged**. No
  factor work at all.
* $Ax$ is unchanged, so $\pi$ is unchanged. The objective value is unchanged.
* The only change is a new multiplier $\lambda_j = \bar c_j$, which is negative: pricing
  just found it improving. The previous optimum is therefore a subspace minimizer
  with exactly the new columns' multipliers negative. The next solve typically starts with
  the reduction of a new column.

### 9.2 A new center $\hat\pi\to\hat\pi'$ (`set_center!`)

Write $\Delta = \hat\pi' - \hat\pi$. Only $\tilde c$ changes, by $-A^\top\Delta$.

* $x$, the ordering and the factor are **unchanged**, since $H$ does not involve $\hat\pi$.
* By (4.2), $\pi$ shifts by exactly $\Delta$ at the same $x$, so every gradient entry
  shifts: $g_j' = g_j - a_j^\top\Delta$.
* The reduced gradient becomes $(Z^\top g')_j = (Z^\top g)_j - u_j^\top\Delta$. In general
  this is nonzero, so the iterate is no longer a subspace minimizer, and the next solve
  starts with a direction computation on the inherited factor.

The new $\pi$ is computed from scratch at the start of the next solve (§7.0), in
$O(\operatorname{nnz}(A))$.

### 9.3 A new $\delta$ (not used by the driver, but free)

Only the right-hand side $-\delta Z^\top g$ and the $\pi$ formula contain $\delta$. The
factor of the *unscaled* $H$ is unaffected.

### 9.4 Initial state

Every pilot's cheapest path (pricing at $\pi = 0$) gives the first column of each block.
That column becomes the representative with $x = 1$. $N = \emptyset$, the factor is the
identity, and $W$ holds every further initial column. This $x$ is feasible for (QP).

---

## 10. The column generation driver

`solve_column_generation(instance, QPASMaster())` runs the following.

### 10.1 Setup

1. Price every pilot at $\pi = 0$ and add its cheapest path. These are the $P$ initial
   columns.
2. `initialize!`: build the `QPASState` (§9.4) with capacity $n_{\max}$.
3. Center $\hat\pi = 0$ (or `initial_center`), `set_center!`.
4. For `center_update = best_bound`, also compute the center value
   $\theta(\hat\pi)$ by pricing every pilot.

### 10.2 One iteration

**(a) Master solve.** `solve!` calls `qpas_solve!` and returns $x$, $\pi$, $\mu$ and
$\Phi(x)$ (4.4). If the status is not optimal, the run stops with `master_failure`.

**(b) Pricing with $\pi$.** Visit the pilots in random order. For pilot $k$, compute the
shortest-path value $v_k = \min_{\xi}(c_k - A^\top_{\text{task}}\pi)^\top\xi$, accumulate
$\theta \mathrel{+}= v_k$, and if $v_k - \mu_k < -10^{-6}$ add the column (unless it is a
duplicate or the pool is full). With `first_improving`, stop after the first added column.
If all $P$ pilots were priced, $\theta = b^\top\pi + \sum_k v_k$ is a valid Lagrangian
bound. Otherwise the bound is not recorded (NaN).

Note that the duals passed to pricing are $\pi = \hat\pi + w$, the **stabilized** duals of
the QP, and $\mu$ from (6.6).

**(c) Convergence and center decision.**

| pricing result | condition | action |
|---|---|---|
| no new column, pool not full | $\lVert\pi-\hat\pi\rVert_\infty\le\tau_c\max(1,\lVert\pi\rVert_\infty)$ | **converged**, stop |
| no new column, pool not full | otherwise | **center-only iteration**: $\hat\pi\leftarrow\pi$ (§11) |
| new column(s) | `every_iteration` | add columns and $\hat\pi\leftarrow\pi$ |
| new column(s) | `best_bound` | add columns; $\hat\pi\leftarrow\pi$ only if $\theta(\pi)$ > center value |
| no new column, pool full | | stop with `column_limit` |

$\tau_c = 10^{-6}$ (`center_tolerance`). `best_bound` forces full pricing every iteration,
so that $\theta$ is always available. This is the bundle-method rule (thesis §2.3.1). The
default `every_iteration` is the pure proximal point method.

**(d) Pass the changes to the master.** `add_columns!` registers the new columns
(§9.1) and `set_center!` stores the new center (§9.2). Nothing is solved yet.

**(e) Statistics.** Record $c^\top x$, $\Phi$, the bound, the infeasibility
$\lVert b - Ax\rVert_\infty$, $\lVert\pi - \pi_{\text{prev}}\rVert_\infty$, the number of
columns and the inner iterations.

**(f) Time limit.** Stop with `time_limit` if the wall clock exceeds the limit.

### 10.3 Two kinds of iterations

With the default `every_iteration`, *every* iteration moves the center. The iterations
differ only in whether a column was added:

| | column-adding iteration | center-only iteration |
|---|---|---|
| pricing | found an improving column (often after visiting only part of the pilots) | complete round, nothing improving |
| change to the QP | new $j\in W$ with $\lambda_j<0$, **and** new center | new center only |
| what the next solve starts with | a direction for the new center, then reductions of the new columns | a direction for the new center |
| factor work | the reductions (`rowadd`) and the expansions they cause | usually none (§11.3) |

---

## 11. Center-only iterations (no new column, only a new center)

### 11.1 Why they exist: exact termination

A stabilized master stops only when two things hold simultaneously:

1. a complete pricing round finds no column with $\bar c < -\varepsilon_{\text{rc}}$, and
2. $\pi = \hat\pi$, i.e. $w = 0$, i.e. $Ax = b$.

**Proposition (exact stopping).** If (QP) is solved to optimality, $\pi = \hat\pi$, and a
complete pricing round finds nothing, then $x$ is optimal for the full LP master.

*Proof.* $\pi = \hat\pi$ gives $Ax = b$, so $x$ is primal feasible for the unstabilized
master. The QP optimality conditions (5.1) give $\bar c_j\ge 0$ on the generated columns,
with equality where $x_j > 0$. Pricing extends $\bar c\ge 0$ to every column. So $(\pi,\mu)$ is
dual feasible, with complementary slackness. $\square$

If condition 1 holds but condition 2 does not, the restricted master has the right columns
for now, but the proximal term still holds $\pi$ away from the restricted dual optimum. The
driver then moves the center to the current $\pi$ and re-solves **without adding a
column**. That is one proximal-point step on the current restricted model,
$\hat\pi^{(t+1)} = \pi^{(t)}$, applied to (D$_\delta$). The proximal point method converges
on the restricted dual, and along the way pricing may find new columns again, because
$\pi$ moves.

This replaces the heuristic "stop after $k$ non-improving iterations". The run stops only
when the restricted proximal fixed point $\pi = \hat\pi$ is reached *and* certified by
pricing.

### 11.2 What the QPAS does in a center-only iteration

Let $x$ be the optimum of the previous solve with working set $W$, center $\hat\pi$, duals
$\pi$ and residual $r = b - Ax = \delta(\pi - \hat\pi)$. The driver sets

$$
\hat\pi' = \pi, \qquad \Delta = \hat\pi' - \hat\pi = \pi - \hat\pi = r/\delta .
$$

The next solve does the following:

1. **Refresh** (§7.0): $\pi_{\text{start}} = \hat\pi' + r/\delta = \pi + \Delta$.
2. **Reduced gradient.** At the previous optimum $Z^\top g = 0$, so
   $$
   Z^\top g_{\text{start}} = -(AZ)^\top\Delta .
   $$
3. **Direction.** $Hy = -\delta Z^\top g_{\text{start}} = \delta (AZ)^\top\Delta$, so
   $$
   y = \delta H^{-1}(AZ)^\top\Delta, \qquad
   Ap = AZy = \delta\,\Pi\,\Delta, \qquad
   \Pi = AZ\big((AZ)^\top AZ\big)^{-1}(AZ)^\top ,
   $$
   where $\Pi$ is the orthogonal projector onto $\operatorname{range}(AZ)$.
4. **Full step** (if no variable blocks):
   $$
   \pi_{\text{new}} = \pi_{\text{start}} - \frac1\delta Ap = \pi + (I - \Pi)\Delta,
   \qquad
   r_{\text{new}} = \delta(\pi_{\text{new}} - \hat\pi') = (I - \Pi)\,r .
   $$
5. **Multiplier check.** If all $\lambda_j\ge -\varepsilon$, the solve is finished after
   one direction and one check.

**Consequence: the dual moves along a ray.** Assume the working set stays the same over
several center-only iterations. After the first one, $r\perp\operatorname{range}(AZ)$, so
in the following ones $(AZ)^\top\Delta = 0$, the direction is $p = 0$, and $x$ **does not
move at all**. Each center-only iteration then simply does

$$
\pi \;\leftarrow\; \pi + \frac{r}{\delta}, \qquad r \text{ fixed},
$$

so the dual walks along the fixed direction $r$ (the gradient of the restricted dual
function, $b - Ax$) with step length $\lVert r\rVert/\delta$. Along this walk the
working-set multipliers and the pricing reduced costs change linearly:

$$
\lambda_j \;\leftarrow\; \lambda_j - \frac{u_j^\top r}{\delta}\quad(j\in W),
\qquad
\bar c(\xi) \;\leftarrow\; \bar c(\xi) - \frac{(A_{\text{task}}\xi - a_{\mathrm{rep}})^\top r}{\delta}.
$$

The walk ends after about $\delta\lambda_j/(u_j^\top r)$ iterations, when one of these
happens:

* some $\lambda_j$, $j\in W$, crosses $-\varepsilon$. The QPAS then performs a reduction,
  $x$ moves, and $r$ shrinks or changes direction.
* pricing finds an improving column. A column-adding iteration follows.
* $r = 0$: the walk has stopped at $\pi = \hat\pi$, and if pricing confirms it the run has
  converged. (With $r\ne 0$ the walk cannot continue forever, because the full LP is
  feasible and bounded, so the full dual cannot increase without limit along $r$.)

This explains the long tail. The proximal point method on a polyhedral function advances
with step $\lVert r\rVert/\delta$ along each face. When $r$ is already small, many
iterations are needed to cross a face. A smaller $\delta$ would take larger steps in this
phase, and by §9.3 changing it costs the QPAS no factorization work.

### 11.3 Cost of a center-only iteration

With an unchanged working set, the QPAS solve consists of

* the refresh of $\pi$, $O(\operatorname{nnz}(A))$,
* one direction (one sparse solve pair, $O((|N|+P)\bar\nu)$),
* possibly one full step,
* one multiplier check, $O(\operatorname{nnz}(A))$.

**No factor update** happens unless a multiplier turns negative. The HiGHS masters, by
contrast, must re-pass all costs (the eliminated form) or the $w$ costs (the explicit form)
and re-run their QP solver, which is why the tail costs them far more.

### 11.4 Measured on I2 (this PC, default settings)

| | column-adding iterations | center-only iterations |
|---|---|---|
| count | 1 261 | 10 713 |
| average QPAS inner iterations | 34.2 | 1.78 |
| total master time | 1.35 s | 1.09 s |

The two kinds are **interleaved**, not separated into phases. The first center-only
iteration is iteration 876, and the last column is added at iteration 10 571 (of 11 974).
The center moves, and pricing then finds new columns again. QPAS totals for the run:
62 209 active-set iterations, 34 807 steps (19 601 full), 15 209 expansions (215
representative swaps), 15 428 reductions, 3 singular-Hessian steps, 60 401 CHOLMOD row
updates, 221 refactorizations, final $|N| = 219$ for $m = 225$.

---

## 12. Termination and the accuracy it guarantees

### 12.1 What the stopping test implies

The run stops when a complete pricing round finds nothing and

$$
\lVert\pi - \hat\pi\rVert_\infty \le \tau_c\max(1,\lVert\pi\rVert_\infty).
$$

Because $\pi - \hat\pi = (b - Ax)/\delta$, this is a bound on the coupling infeasibility:

$$
\boxed{\;\lVert b - Ax\rVert_\infty \;\le\; \delta\,\tau_c\,\max(1,\lVert\pi\rVert_\infty)\;}
$$

The tolerance is relative to the size of the duals and scales with $\delta$. It is not an
absolute $10^{-6}$ on feasibility.

### 12.2 Which number to read

At termination:

* The **Lagrangian bound** $\theta(\pi)$ is a valid lower bound on $z_{\text{LP}}$, and the
  stabilized objective $\Phi$ is very close to it.
* The **operational cost** $c^\top x$ is the cost of a slightly infeasible $x$. It
  differs from $z_{\text{LP}}$ by roughly $\pi^\top(b - Ax)$, which can be of order
  $\lVert\pi\rVert_1\lVert r\rVert_\infty$.

On I2 (this PC): $\lVert r\rVert_\infty = 5.8\cdot 10^{-5}$, consistent with the bound
above for $\lVert\pi\rVert_\infty\approx 58$. $\Phi = \theta = 1126.974681$, against
$z_{\text{LP}} = 1126.974867$ (relative $1.7\cdot 10^{-7}$). But
$c^\top x = 1126.949902$ is $0.025$ too low, because $x$ is slightly infeasible. For a
tighter $c^\top x$, decrease `center_tolerance`. This costs extra center-only iterations,
which §11.3 shows are cheap for the QPAS.

---

## 13. The complete algorithm in pseudocode

```text
INPUT  instance (DAG, pilot edge costs, task matrix, b), δ, tolerances
OUTPUT x, π, μ, bound

── setup ──────────────────────────────────────────────────────────────────────────────
for k = 1..P:  ξ ← shortest path for pilot k at π = 0;  add column (a = A_task ξ, c = c_kᵀξ)
state: S[k] ← that column, x_S = 1;  W ← {}; N ← {};  factor L = D = I (n_max × n_max)
π̂ ← 0

── column generation ──────────────────────────────────────────────────────────────────
loop
    ── master: QPAS on  min 1/(2δ)‖Ax‖² + c̃ᵀx,  Cx = 1, x ≥ 0 ──────────────────────
    π ← π̂ + (b − Ax)/δ                                       O(nnz A)
    at_min ← false
    loop
        if not at_min:
            μ_b ← c_rep(b) − a_rep(b)ᵀπ                       for all blocks
            rhs_k ← −δ (c_k − a_kᵀπ − μ_ψ(k))                 k ∈ N
            y ← L⁻ᵀ D⁻¹ L⁻¹ rhs  on reach(N)                  sparse solve
            p_k ← y_k;  p_rep(k) −= y_k                       k ∈ N
            if ‖p‖∞ ≤ 1e−10: at_min ← true
        if at_min:
            μ_b ← c_rep(b) − a_rep(b)ᵀπ
            r ← argmin_{j∈W} λ_j,  λ_j = c_j − a_jᵀπ − μ_ψ(j)  O(nnz A)
            if W = {} or λ_r ≥ −ε: break                      ← master optimal
            u ← a_r − a_rep(r)
            h_k ← (a_k − a_rep(k))ᵀu  (k ∈ N),  h_rr ← ‖u‖²
            rowadd(r, h);  N ← N ∪ {r}
            if min_{k∈N} D_kk / ‖u_k‖² ≤ 1e−10:               ← u dependent on AZ
                W ← W ∪ {r}; refactor
                t ← H⁻¹h;  d ← z_r − Σ_k t_k z_k              Ad = 0, gᵀd = λ_r < 0
                α, q ← ratio test along d (α_max = ∞);  x += αd
                add q to W (as in the expansion below), retry the release of r
            at_min ← false
        else:
            α ← min(1, min_{p_k<0} x_k / (−p_k));  q ← blocking index or none
            x ← x + αp;   π ← π − (α/δ) Σ_k p_k a_k
            if q = none: at_min ← true                        ← full step
            elif q ∈ N:  rowdel(q); N ← N∖{q}; W ← W ∪ {q}; x_q ← 0
            else (q = rep of block b):
                ν ← {j ∈ N : ψ(j) = b};  r ← argmax_{j∈ν} x_j
                rowdel(j) for j ∈ ν;  S[b] ← r
                rowadd(j, column of (a_j − a_r)) for j ∈ ν∖{r}
                W ← W ∪ {q}; x_q ← 0
            if #updates since refactor ≥ 150 or a pivot decayed: refactor
    return x, π, μ, Φ = cᵀx + δ/2‖π‖² − δ/2‖π̂‖²

    ── pricing ─────────────────────────────────────────────────────────────────────
    θ ← bᵀπ;  new ← {}
    for k in random order:
        v_k ← shortest path value with edge weights c_ek − (A_taskᵀπ)_e;  θ += v_k
        if v_k − μ_k < −1e−6 and not a duplicate: new ← new ∪ {column}; break (first_improving)

    ── decision ────────────────────────────────────────────────────────────────────
    if new = {}:
        if ‖π − π̂‖∞ ≤ 1e−6 · max(1, ‖π‖∞): STOP (optimal; bound θ)
        else: π̂ ← π                                         ← center-only iteration
    else:
        new columns → W with x = 0 (factor untouched)
        π̂ ← π   (every_iteration)   or   if θ > θ(π̂): π̂ ← π   (best_bound)
    stop on time limit / iteration limit / full pool
end loop
```

---

## 14. Cost of every operation and map to the code

| operation | formula | cost | grows with $n$? | code |
|---|---|---|---|---|
| refresh duals | $\pi = \hat\pi + (b - Ax)/\delta$ | $O(\operatorname{nnz}(A))$ | yes | `refresh_duals!` |
| gradient entry | $g_j = c_j - a_j^\top\pi$ | $O(\bar\nu)$ | no | `gradient` |
| convexity duals | $\mu_b = g_{S[b]}$ | $O(P\bar\nu)$ | no | `update_convexity_duals!` |
| direction | $Hy = -\delta Z^\top g$, $p = Zy$ | $O((\lvert N\rvert+P)\bar\nu)$ + sparse solve | no | `compute_direction!` |
| ratio test + step | $\alpha$, $x \mathrel{+}= \alpha p$, $\pi \mathrel{-}= \frac\alpha\delta Ap$ | $O((\lvert N\rvert+P)\bar\nu + m)$ | no | `ratio_test`, `take_step!` |
| multiplier check | $\lambda_j = g_j - \mu_{\psi(j)}$, $j\in W$ | $O(\operatorname{nnz}(A))$ | yes | `most_negative_multiplier` |
| Hessian column | $h_k = (a_k - a_{\mathrm{rep}(k)})^\top u_r$ | $O((\lvert N\rvert+P)\bar\nu)$ | no | `hessian_column!` |
| reduction | `rowadd` + pivot check | $O(\lvert N\rvert^2)$ at most | no | `reduce_working_set!` |
| expansion, $q\in N$ | `rowdel` | $O(\operatorname{nnz}L)$ at most | no | `expand_working_set!` |
| expansion, $q\in S$ | $2\lvert\nu\rvert - 1$ updates + $\lvert\nu\rvert-1$ columns | | no | `_swap_representative!` |
| singular step | $t = H^{-1}h$, step along $d$ | one solve + ratio test | no | `_singular_step!` |
| refactorization | rebuild in increasing index order | $\approx\lvert N\rvert^3/3$ + $\lvert N\rvert$ columns | no | `refactor!` |
| new column | $j\to W$, $x_j = 0$ | $O(1)$ | no | `register_columns!` |
| new center | store $\hat\pi$ | $O(m)$ (+ refresh at next solve) | — | `set_center!` |
| pricing (one pilot) | DAG shortest path | $O(\lvert V\rvert + \lvert E\rvert)$ | no | `price_block!` |

| file | contents |
|---|---|
| `src/cg/column_generation.jl` | driver: setup, pricing loop, convergence and center decision, Lagrangian bound |
| `src/cg/options.jl` | `CGOptions`, `StabilizationOptions` (δ, center rule, tolerances) |
| `src/master/qpas_master.jl` | adapter between the driver and the QPAS state |
| `src/qpas/hotstart_state.jl` | `QPASState`, $\pi$/gradient identities, registering columns, center |
| `src/qpas/ordering.jl` | the partition $[S, W, N]$ |
| `src/qpas/nullspace.jl` | explicit $Z$ and $H$ for tests/debugging |
| `src/qpas/kkt_system.jl` | Hessian column assembly, `rowadd`/`rowdel` wrappers, refactorization, verification |
| `src/qpas/solver.jl` | the active-set loop: direction, multipliers, step, expansion, reduction, singular steps |
| `src/qpas/options.jl` | QPAS tolerances, refactorization interval, counters |
| `src/backends/cholmod/ldlt_factor.jl` | fixed-size CHOLMOD $LDL^\top$: identity start, `rowadd!`, `rowdel!`, sparse `solve!` |
| `src/pricing/shortest_path.jl` | DAG shortest-path pricing |
