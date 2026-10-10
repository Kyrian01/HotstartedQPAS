# Regularized Column Generation with Inequality Coupling Constraints

*An extension of Chapter 4 of "Hot-Started Active-Set Method for Regularized Column
Generation". Sections 1–5 below follow Sections 4.1–4.5 of the thesis one to one. References
of the form "Theorem 4.2.3" or "(4.17)" point to the thesis; numbers of the form
"Theorem 2.3" or "(2.5)" point to this document.*

In Chapter 4 all linking constraints are equalities, $AX\lambda = b$. In many applications some
of them are inequalities. In crew scheduling, for example, a task may be covered *at least* once,
with deadheading allowed. This chapter extends the regularized column generation framework
and the hot-started QPAS to linking constraints of the form

$$
A_E X\lambda = b_E,\qquad A_I X\lambda\ \ge\ b_I .
$$

The approach is to introduce surplus variables, which turn the inequalities into equalities,
and to collect weights and surpluses in one vector. The regularized master problem then has
the same form as in Chapter 4. Only the convexity matrix gains zero columns, which breaks the
column generation properties (Definition 4.2.1). We show that the reordering theorem still
holds after splitting the surpluses into an active and a free set. The nullspace basis becomes
block diagonal, and the reduced KKT system splits into an explicit surplus part and a column
part.

The column part is the main result: its reduced Hessian is exactly the reduced Hessian of
Chapter 4, built from the equality rows together with the inequality rows whose surplus is in
the working set. As a result, all results of Sections 4.3–4.5 carry over, and a surplus that
changes status causes a rank-one update of the factorization.

---

## 1. Regularized Column Generation Framework

### 1.1 The master problem with inequality linking constraints

We consider the multi-block linear program (4.1) with the linking constraints split into
$m_E$ equality rows and $m_I$ inequality rows, $m = m_E + m_I$:

$$
\begin{aligned}
\min_x\;\; & c^\top x\\
\text{s.t.}\;\; & A_E x = b_E,\\
& A_I x \ge b_I,\\
& Fx = d,\quad l\le x\le u .
\end{aligned}
\tag{1.1}
$$

The Dantzig–Wolfe reformulation of Section 1.4 applies unchanged, since it only acts on the
easy constraints. This gives the **master problem**

$$
\begin{aligned}
\min_\lambda\;\; & c^\top X\lambda\\
\text{s.t.}\;\; & A_EX\lambda = b_E, &&(1.2\text{b})\\
& A_IX\lambda \ge b_I, &&(1.2\text{c})\\
& C\lambda = \mathbf 1,\quad \lambda\ge 0. &&(1.2\text{d})
\end{aligned}
\tag{1.2}
$$

Its dual (Section 2.2) has multipliers $\pi_E\in\mathbb R^{m_E}$ (free), $\pi_I\in\mathbb R^{m_I}$
and $\pi_0\in\mathbb R^p$. The only difference from the equality case is the **sign condition**

$$
\pi_I\ \ge\ 0 .
\tag{1.3}
$$

### 1.2 Surplus variables and the unified variable

We introduce a surplus vector $s\in\mathbb R^{m_I}$, $s\ge 0$, and write (1.2c) as
$A_IX\lambda - s = b_I$. Collecting weights and surpluses in

$$
y := \begin{bmatrix}\lambda\\ s\end{bmatrix},
\qquad
\bar A := \begin{bmatrix} A_EX & 0\\ A_IX & -I\end{bmatrix},
\qquad
\bar C := \begin{bmatrix} C & 0\end{bmatrix},
\qquad
\bar c := \begin{bmatrix} X^\top c\\ 0\end{bmatrix},
\qquad
b := \begin{bmatrix} b_E\\ b_I\end{bmatrix},
$$

the master problem (1.2) becomes

$$
\min_y\;\bar c^\top y\quad\text{s.t.}\quad \bar Ay = b,\quad \bar Cy = \mathbf 1,\quad y\ge 0 .
\tag{1.4}
$$

This is exactly the form of (4.2), with $(AX, C, c^\top X)$ replaced by
$(\bar A, \bar C, \bar c)$.

### 1.3 The regularized master problem

Applying the proximal stabilization of Section 2.3 to (1.4) gives the
**regularized master problem**

$$
\begin{aligned}
\min_{y,w}\;\; & \bar c^\top y + \frac\delta2\lVert w + \hat\pi\rVert^2 - \frac\delta2\lVert\hat\pi\rVert^2\\
\text{s.t.}\;\; & \bar Ay + \delta w = b,\qquad \bar Cy = \mathbf 1,\qquad y\ge 0,
\end{aligned}
\tag{1.5}
$$

with stability center $\hat\pi = [\hat\pi_E;\hat\pi_I]\in\mathbb R^m$. As in (4.4), the multiplier
of $\bar Ay + \delta w = b$ is

$$
\pi = \hat\pi + w = \hat\pi + \frac1\delta\big(b - \bar Ay\big),
\qquad\text{i.e.}\qquad
\begin{aligned}
\pi_E &= \hat\pi_E + \tfrac1\delta\big(b_E - A_EX\lambda\big),\\
\pi_I &= \hat\pi_I + \tfrac1\delta\big(b_I - A_IX\lambda + s\big).
\end{aligned}
\tag{1.6}
$$

A surplus raises the dual of its row.

**Proposition 1.1.** The dual of (1.5) is

$$
\max_{\pi,\pi_0}\; b^\top\pi + \mathbf 1^\top\pi_0 - \frac\delta2\lVert\pi - \hat\pi\rVert^2
\quad\text{s.t.}\quad X^\top c - (AX)^\top\pi - C^\top\pi_0\ \ge\ 0,\qquad \pi_I\ \ge\ 0 ,
$$

where $A = [A_E; A_I]$.

*Proof.* As in Section 2.3, eliminating $w$ from the Lagrangian leaves
$-\frac\delta2\lVert\pi - \hat\pi\rVert^2$. Minimizing over $y\ge 0$ requires
$\bar c - \bar A^\top\pi - \bar C^\top\pi_0\ge 0$. The first block row of this condition is the
reduced-cost condition. The second reads $0 - (-I)\pi_I - 0\ge 0$, i.e. $\pi_I\ge 0$. $\square$

So the sign condition (1.3) is part of the stabilized dual automatically, through the surplus
columns. The center $\hat\pi$ itself may violate it.

Since all linking rows of (1.5) are now equalities, $w$ can be eliminated exactly as in (4.5).
This gives the equivalent problem

$$
\min_y\;\Big(\bar c - \bar A^\top\big(\tfrac b\delta + \hat\pi\big)\Big)^\top y + \frac1{2\delta}\lVert\bar Ay\rVert^2
\quad\text{s.t.}\quad \bar Cy = \mathbf 1,\quad y\ge 0 ,
\tag{1.7}
$$

up to constant terms. Algorithm 4.1 applies without change. Pricing uses $\pi = [\pi_E;\pi_I]$
from (1.6), and the surpluses are never priced: they form a fixed set of $m_I$ variables that
are present from the start.

---

## 2. QPAS for the Regularized Master Problem

### 2.1 Simplified notation

As in Section 4.2, we write $x\in\mathbb R^n$ for the column weights and set

$$
A_E\leftarrow A_EX\in\mathbb R^{m_E\times n},\qquad A_I\leftarrow A_IX\in\mathbb R^{m_I\times n},\qquad
c\leftarrow X^\top c\in\mathbb R^n .
$$

Unlike Chapter 4, we keep $c$ for the **original column costs**, because they are needed in
the multiplier formulas below. The linear term of (1.7) is denoted $\tilde c$:

$$
\tilde c = \begin{bmatrix}\tilde c_x\\ \tilde c_s\end{bmatrix}
= \begin{bmatrix} c - A_E^\top\big(\tfrac{b_E}\delta + \hat\pi_E\big) - A_I^\top\big(\tfrac{b_I}\delta + \hat\pi_I\big)\\[3pt]
\tfrac{b_I}{\delta} + \hat\pi_I\end{bmatrix}.
$$

With $y = [x; s]\in\mathbb R^{n+m_I}$, the regularized master problem reads

$$
\begin{aligned}
\min_{y}\;\; & \frac1{2\delta}\,y^\top\bar A^\top\bar A\,y + \tilde c^\top y &&(2.1\text{a})\\
\text{s.t.}\;\; & \bar Cy = \mathbf 1, &&(2.1\text{b})\\
& y\ge 0, &&(2.1\text{c})
\end{aligned}
\tag{2.1}
$$

with

$$
\bar A = \begin{bmatrix} A_E & 0\\ A_I & -I\end{bmatrix},\qquad
\bar C = \begin{bmatrix} C & 0\end{bmatrix},\qquad
\bar A^\top\bar A = \begin{bmatrix} A_E^\top A_E + A_I^\top A_I & -A_I^\top\\ -A_I & I\end{bmatrix}.
\tag{2.2}
$$

Problem (2.1) has exactly the structure of (4.6).

### 2.2 The gradient

**Lemma 2.1 (gradient–dual relation).** Let $\pi = [\pi_E;\pi_I]$ be given by (1.6) at the
iterate $y = [x;s]$. Then the gradient $g = \frac1\delta\bar A^\top\bar Ay + \tilde c$ of
(2.1a) is

$$
g = \begin{bmatrix} g_x\\ g_s\end{bmatrix}
= \begin{bmatrix} c - A_E^\top\pi_E - A_I^\top\pi_I\\ \pi_I\end{bmatrix}.
\tag{2.3}
$$

*Proof.* $g = \bar c + \frac1\delta\bar A^\top\bar Ay - \bar A^\top\big(\tfrac b\delta + \hat\pi\big)
= \bar c - \bar A^\top\big(\hat\pi + \tfrac1\delta(b - \bar Ay)\big) = \bar c - \bar A^\top\pi$.
The two block rows of $\bar A^\top$ are $[A_E^\top\ A_I^\top]$ and $[0\ -I]$. $\square$

The gradient with respect to a column weight is the reduced cost of that column without the
convexity dual, exactly as in the equality case. The gradient with respect to a surplus
is the dual of its inequality row.

### 2.3 Constraint matrix and working sets

The working set $\mathcal W\subseteq\{1,\dots,n+m_I\}$ now contains column indices *and*
surplus indices. We write

$$
\mathcal W = \mathcal W_x\cup\mathcal W_s,
$$

where $\mathcal W_x\subseteq\{1,\dots,n\}$ are the columns fixed at $x_j = 0$, and
$\mathcal W_s\subseteq\{1,\dots,m_I\}$ are the **active surpluses**, fixed at $s_i = 0$.
As in (4.9), let $W_x\in\mathbb R^{|\mathcal W_x|\times n}$ and
$W_s\in\mathbb R^{|\mathcal W_s|\times m_I}$ be the corresponding selection matrices. The
constraint matrix associated with $\mathcal W$ is

$$
D_{\mathcal W} = \begin{bmatrix} C & 0\\ W_x & 0\\ 0 & W_s\end{bmatrix}.
\tag{2.4}
$$

**The column generation properties are lost.** The matrix $\bar C = [C\ \ 0]$ has $m_I$ zero
columns, so property (ii) of Definition 4.2.1 fails for $\bar C$. The surpluses belong to no
block. The matrix $C$ of the column weights, however, still satisfies all four properties. The
reordering theorem therefore extends by treating the two kinds of variables separately.

**Definition 2.2 (extended column generation ordering).** Let $\mathcal W = \mathcal W_x\cup\mathcal W_s$
be a working set. The **extended ordering** is the partition of the index set into five sets:

| set | contents |
|---|---|
| $\mathcal S$ | the list of representatives, one column per block (Theorem 4.2.3) |
| $\mathcal W_x$ | columns in the working set |
| $\mathcal N_x$ | the remaining free columns (nullspace columns) |
| $\mathcal W_s$ | surpluses in the working set (active surpluses) |
| $\mathcal N_s$ | the remaining surpluses (free surpluses) |

We write $\mathcal N_s$ rather than "free set" because, as Section 3.1 shows, every free
surplus is a nullspace index: unlike a column, it needs no representative.

**Theorem 2.3 (extended reordering theorem).** Let $\mathcal W = \mathcal W_x\cup\mathcal W_s$
be a working set such that every block has at least one column outside $\mathcal W_x$. Then
there is a permutation matrix $Q$, ordering the columns as
$[\mathcal S, \mathcal W_x, \mathcal N_x \mid \mathcal W_s, \mathcal N_s]$, with

$$
D_{\mathcal W}\,Q =
\begin{bmatrix}
I_p & C_{\mathcal W} & C_{\mathcal N} & 0 & 0\\
0 & I_{|\mathcal W_x|} & 0 & 0 & 0\\
0 & 0 & 0 & I_{|\mathcal W_s|} & 0
\end{bmatrix}.
\tag{2.5}
$$

*Proof.* $Q = \mathrm{diag}(Q_x, Q_s)$ with $Q_x$ the permutation of Theorem 4.2.3 applied
to the column weights, which is possible because $C$ satisfies the column generation properties.
$Q_s$ orders the surpluses as $[\mathcal W_s, \mathcal N_s]$. The first two block rows of (2.4)
then take the form (4.12), with two zero blocks appended. The last block row selects
$\mathcal W_s$, giving $[0\ \ I]$ on the surplus columns. $\square$

**Theorem 2.4.** $D_{\mathcal W}$ has full row rank, $\operatorname{rank}(D_{\mathcal W}) = p + |\mathcal W_x| + |\mathcal W_s|$.

*Proof.* In (2.5) the pivot blocks $I_p$, $I_{|\mathcal W_x|}$, $I_{|\mathcal W_s|}$ are in
distinct rows and columns. $\square$

So the first assumption of the nullspace method (Remark 4.3.1) holds for every working set.

### 2.4 Reduced equality-constrained problem

For a working set $\mathcal W$ and the current iterate $y_k$, the search direction
$p = [p_x; p_s]$ solves (cf. (4.16))

$$
\min_p\;\frac1{2\delta}p^\top\bar A^\top\bar A\,p + g^\top p
\quad\text{s.t.}\quad Cp_x = 0,\quad W_xp_x = 0,\quad W_sp_s = 0,
\tag{2.6}
$$

and the KKT system (4.17) becomes

$$
\begin{bmatrix} \frac1\delta\bar A^\top\bar A & D_{\mathcal W}^\top\\ D_{\mathcal W} & 0\end{bmatrix}
\begin{bmatrix} p\\ -\mu\\ -\lambda_x\\ -\lambda_s\end{bmatrix}
=
\begin{bmatrix} -g\\ 0\\ 0\\ 0\end{bmatrix},
\tag{2.7}
$$

with $\mu\in\mathbb R^p$ the multipliers of the convexity constraints, and
$\lambda_x\in\mathbb R^{|\mathcal W_x|}$, $\lambda_s\in\mathbb R^{|\mathcal W_s|}$ those of the
working-set constraints. Algorithm 4.2 applies verbatim with $x$ replaced by $y$. In
particular, the ratio test runs over all free indices of $y$, i.e. over
$\mathcal S\cup\mathcal N_x\cup\mathcal N_s$:

$$
\alpha = \min\Big\{1,\ \min_{i\notin\mathcal W,\ p_i<0}\Big(-\frac{y_i}{p_i}\Big)\Big\}.
$$

---

## 3. Nullspace Method for Solving the KKT System

### 3.1 Nullspace basis of the constraint matrix

By Theorem 2.4 and the rank–nullity theorem,

$$
\dim\operatorname{Null}(D_{\mathcal W}) = n + m_I - p - |\mathcal W_x| - |\mathcal W_s| = |\mathcal N_x| + |\mathcal N_s| .
$$

**Theorem 3.1 (extended nullspace basis).** A basis of $\operatorname{Null}(D_{\mathcal W})$ is

$$
Z = \begin{bmatrix} Z_x & 0\\ 0 & Z_s\end{bmatrix}\in\mathbb R^{(n+m_I)\times(|\mathcal N_x| + |\mathcal N_s|)},
\tag{3.1}
$$

where $Z_x\in\mathbb R^{n\times|\mathcal N_x|}$ is the basis of Theorem 4.3.3 for the column
weights, and $Z_s\in\mathbb R^{m_I\times|\mathcal N_s|}$ is the selection matrix of the free
surpluses. In terms of columns,

$$
z_j = e_j - e_{\mathrm{rep}(j)}\quad(j\in\mathcal N_x),
\qquad\qquad
z_i = e_i\quad(i\in\mathcal N_s).
\tag{3.2}
$$

*Proof.* With (2.4): $CZ_x = 0$ and $W_xZ_x = 0$ by Theorem 4.3.3, and $W_sZ_s = 0$ because
$\mathcal W_s\cap\mathcal N_s = \emptyset$. So $D_{\mathcal W}Z = 0$. The columns are
linearly independent, and their number equals the dimension of the nullspace. $\square$

The first kind of nullspace vector shifts weight from a column to its representative, as
explained after Corollary 4.3.4. The second kind simply changes a free surplus. A surplus
appears in no convexity constraint, so it does not need a partner to keep (2.1b) satisfied.

### 3.2 Solving the reduced KKT system

With $p = Z\eta$, $\eta = [\eta_x; \eta_s]$ (we write $\eta$ for the reduced coordinates, called $y$ in Chapter 4, because $y$ now denotes $[x;s]$), the KKT system reduces as in (4.30) to

$$
\frac1\delta\,Z^\top\bar A^\top\bar AZ\;\eta = -Z^\top g .
\tag{3.3}
$$

To write this system out, we introduce the products of the linking matrices with the
column basis:

$$
B_E := A_EZ_x\in\mathbb R^{m_E\times|\mathcal N_x|},\qquad
B_I := A_IZ_x\in\mathbb R^{m_I\times|\mathcal N_x|},
$$

and we split the rows of $B_I$ according to the status of the surplus of each row:

$$
B_{\mathcal W} := W_sB_I\quad(\text{rows with an active surplus}),\qquad
B_{\mathcal N} := Z_s^\top B_I\quad(\text{rows with a free surplus}).
$$

In the same way, $\pi_{\mathcal W} := W_s\pi_I$ and $\pi_{\mathcal N} := Z_s^\top\pi_I$
denote the duals of these two groups of inequality rows. Since every inequality row belongs
to exactly one group,

$$
B_I^\top B_I = B_{\mathcal W}^\top B_{\mathcal W} + B_{\mathcal N}^\top B_{\mathcal N}.
\tag{3.4}
$$

**The blocks of (3.3).** From (2.2) and (3.1),

$$
\bar AZ = \begin{bmatrix} B_E & 0\\ B_I & -Z_s\end{bmatrix},
\qquad
\bar H := Z^\top\bar A^\top\bar AZ
= \begin{bmatrix} B_E^\top B_E + B_I^\top B_I & -B_{\mathcal N}^\top\\ -B_{\mathcal N} & I\end{bmatrix}.
\tag{3.5}
$$

The upper-left block is the reduced Hessian $H = (AZ_x)^\top(AZ_x)$ of (4.31) with
$A = [A_E; A_I]$. The reduced gradient follows from (2.3):

$$
Z^\top g = \begin{bmatrix} Z_x^\top g_x\\ \pi_{\mathcal N}\end{bmatrix}.
$$

**Theorem 3.2 (splitting of the reduced KKT system).** The reduced system (3.3) is
equivalent to

$$
\boxed{\;H_{\mathrm{eff}}\;\eta_x = -\delta\,Z_x^\top\big(c - A_E^\top\pi_E - A_I^\top W_s^\top\pi_{\mathcal W}\big)\;}
\tag{3.6}
$$

$$
\boxed{\;\eta_s = B_{\mathcal N}\,\eta_x - \delta\,\pi_{\mathcal N}\;}
\tag{3.7}
$$

with the **effective reduced Hessian**

$$
\boxed{\;H_{\mathrm{eff}} := H - B_{\mathcal N}^\top B_{\mathcal N} = B_E^\top B_E + B_{\mathcal W}^\top B_{\mathcal W}\;}
\tag{3.8}
$$

*Proof.* Multiply (3.3) by $\delta$ and write it in the blocks of (3.5):

$$
\begin{aligned}
H\,\eta_x - B_{\mathcal N}^\top \eta_s &= -\delta\,Z_x^\top g_x,\\
-B_{\mathcal N}\,\eta_x + \eta_s &= -\delta\,\pi_{\mathcal N}.
\end{aligned}
$$

The second equation is (3.7). Substituting it into the first gives
$(H - B_{\mathcal N}^\top B_{\mathcal N})\,\eta_x = -\delta\big(Z_x^\top g_x + B_{\mathcal N}^\top\pi_{\mathcal N}\big)$.
The second expression for $H_{\mathrm{eff}}$ follows from (3.4). For the right-hand side, insert
$g_x$ from (2.3) and split $\pi_I$ into its two groups:
$Z_x^\top A_I^\top\pi_I = B_{\mathcal W}^\top\pi_{\mathcal W} + B_{\mathcal N}^\top\pi_{\mathcal N}$.
The term $B_{\mathcal N}^\top\pi_{\mathcal N}$ cancels, leaving (3.6). $\square$

$H_{\mathrm{eff}}$ is the Schur complement of the identity block in $\bar H$. Only an
$|\mathcal N_x|\times|\mathcal N_x|$ system has to be solved. The surplus part of the direction
follows explicitly from (3.7), in components

$$
(p_s)_i = (A_Ip_x)_i - \delta\,(\pi_I)_i\quad(i\in\mathcal N_s),\qquad (p_s)_i = 0\quad(i\in\mathcal W_s).
$$

The following corollary is the central observation of this chapter.

**Corollary 3.3 (the effective coupling matrix).** Define the **effective coupling matrix**
and its dual vector

$$
A_{\mathrm{eff}} := \begin{bmatrix} A_E\\ W_sA_I\end{bmatrix},\qquad
\pi_{\mathrm{eff}} := \begin{bmatrix}\pi_E\\ \pi_{\mathcal W}\end{bmatrix},
$$

i.e. the equality rows together with the inequality rows whose surplus is active. Then

$$
H_{\mathrm{eff}} = (A_{\mathrm{eff}}Z_x)^\top(A_{\mathrm{eff}}Z_x),
\qquad
H_{\mathrm{eff}}\,\eta_x = -\delta\,Z_x^\top\big(c - A_{\mathrm{eff}}^\top\pi_{\mathrm{eff}}\big).
$$

This is exactly the reduced KKT system (4.32) of Chapter 4 for the master problem whose
linking constraints are the rows of $A_{\mathrm{eff}}$.

*Proof.* $A_{\mathrm{eff}}Z_x = [B_E; B_{\mathcal W}]$, and (3.8) and (3.6) are the claimed
expressions. $\pi_{\mathrm{eff}}$ is given by (1.6) restricted to these rows, with $s_i = 0$ for
$i\in\mathcal W_s$, so it is the dual vector (4.4) of that master problem. $\square$

**Interpretation.** An inequality row with a free surplus does not constrain the column
weights: any change of its residual is absorbed by the surplus. Such a row therefore
disappears from the column part of the problem. An inequality row whose surplus is active
behaves like an equality row. The QPAS for the inequality case is thus the QPAS of
Chapter 4, applied to a coupling matrix whose rows change with the surplus working set.

**Corollary 3.4 (positive definiteness).** $\bar H$ is positive definite if and only if
$H_{\mathrm{eff}}$ is, i.e. if and only if $A_{\mathrm{eff}}Z_x$ has full column rank. In
particular,

$$
|\mathcal N_x|\ \le\ \operatorname{rank}(A_{\mathrm{eff}})\ \le\ m_E + |\mathcal W_s| .
$$

*Proof.* A symmetric matrix with a positive definite diagonal block ($I$) is positive definite
if and only if the Schur complement of that block is. $\square$

**Remark 3.5.** If there are no inequality rows ($m_I = 0$), then $A_{\mathrm{eff}} = A_E$,
$H_{\mathrm{eff}} = H$, and Theorem 3.2 reduces to (4.32). If all surpluses are active
($\mathcal N_s = \emptyset$), every inequality row is treated as an equality, and again
$H_{\mathrm{eff}} = H$.

### 3.3 Solving for the Lagrange multipliers

As in (4.34) let $h := \frac1\delta\bar A^\top\bar Ap + g$, partitioned as $h = [h_x; h_s]$.
The first block row of (2.7) reads $D_{\mathcal W}^\top[\mu;\lambda_x;\lambda_s] = h$.

**Theorem 3.6 (extension of Theorem 4.3.5).** With the extended ordering of Theorem 2.3, the
Lagrange multipliers are

$$
\mu = h_{\mathcal S},\qquad
\lambda_x = h_{\mathcal W_x} - C_{\mathcal W}^\top\mu,\qquad
\lambda_s = h_{\mathcal W_s},
\tag{3.9}
$$

and in addition $h_{\mathcal N_s} = 0$.

*Proof.* With (2.5), the system $Q^\top D_{\mathcal W}^\top[\mu;\lambda_x;\lambda_s] = Q^\top h$ reads

$$
\begin{bmatrix}
I_p & 0 & 0\\
C_{\mathcal W}^\top & I & 0\\
C_{\mathcal N}^\top & 0 & 0\\
0 & 0 & I\\
0 & 0 & 0
\end{bmatrix}
\begin{bmatrix}\mu\\ \lambda_x\\ \lambda_s\end{bmatrix}
=
\begin{bmatrix} h_{\mathcal S}\\ h_{\mathcal W_x}\\ h_{\mathcal N_x}\\ h_{\mathcal W_s}\\ h_{\mathcal N_s}\end{bmatrix}.
$$

The first, second and fourth block rows give (3.9), and the last gives $h_{\mathcal N_s} = 0$. $\square$

**Corollary 3.7 (multipliers at a subspace minimizer).** If $p = 0$, then $h = g$, and by Lemma 2.1

$$
\begin{aligned}
\mu_b &= c_{S[b]} - a_{S[b]}^\top\pi && (b = 1,\dots,p),\\
(\lambda_x)_j &= c_j - a_j^\top\pi - \mu_{\psi(j)} && (j\in\mathcal W_x),\\
(\lambda_s)_i &= (\pi_I)_i && (i\in\mathcal W_s),\\
(\pi_I)_i &= 0 && (i\in\mathcal N_s),
\end{aligned}
$$

where $a_j$ denotes column $j$ of $A = [A_E;A_I]$.

The optimality test of Algorithm 4.2, $\lambda\ge 0$, therefore checks two things:

* **column multipliers:** $(\lambda_x)_j$ is the reduced cost of column $j$, the same quantity
  that the pricing problem evaluates for columns that have not been generated yet. Exactly as in
  the equality case, the QPAS and the pricing problem test the same condition, inside and
  outside the restricted master problem.
* **surplus multipliers:** $(\lambda_s)_i$ is the dual of inequality row $i$, so
  $\lambda_s\ge 0$ is the sign condition (1.3) on the rows with an active surplus. On the rows
  with a free surplus, the dual is zero. A negative $(\lambda_s)_i$ means that row $i$ is
  over-covered while its dual is negative. Releasing the surplus moves the dual back to zero.

**Remark 3.8.** At a subspace minimizer, $\pi_I\ge 0$ holds on the whole inequality block,
and $(\pi_I)_i = 0$ whenever $s_i > 0$. These are dual feasibility and complementary slackness
of the inequality rows of (1.2). Consequently, the stopping criterion of the column generation
(an optimal master, $\pi = \hat\pi$, and no column with negative reduced cost) certifies optimality
of (1.2), just as in the equality case.

### 3.4 Summary

Each QPAS iteration solves the KKT system as follows.

1. **Determine the ordering** $[\mathcal S,\mathcal W_x,\mathcal N_x\mid\mathcal W_s,\mathcal N_s]$.
   The column sets are obtained as in Section 4.3.4, and the surpluses are split according to
   the working set.
2. **Nullspace basis:** $z_j = e_j - e_{\mathrm{rep}(j)}$ for $j\in\mathcal N_x$, and $z_i = e_i$ for $i\in\mathcal N_s$.
3. **Column part:** form the effective coupling matrix $A_{\mathrm{eff}} = [A_E;W_sA_I]$, the
   effective reduced Hessian $H_{\mathrm{eff}} = (A_{\mathrm{eff}}Z_x)^\top(A_{\mathrm{eff}}Z_x)$,
   and its $LDL^\top$ factorization. Solve
   $H_{\mathrm{eff}}\eta_x = -\delta Z_x^\top(c - A_{\mathrm{eff}}^\top\pi_{\mathrm{eff}})$ and set
   $p_x = Z_x\eta_x$.
4. **Surplus part:** $(p_s)_i = (A_Ip_x)_i - \delta(\pi_I)_i$ for $i\in\mathcal N_s$, and zero otherwise.
5. **Multipliers** (when $p = 0$): $\mu = g_{\mathcal S}$, $\lambda_x$ = reduced costs on
   $\mathcal W_x$, $\lambda_s = \pi_I$ on $\mathcal W_s$.

---

## 4. Efficient Updates for the KKT System

As in Section 4.4, we use a fixed-length representation. The column part keeps the dimension
$n$ (and $n_{\max}$ in practice, Section 4.5.1), and the surplus part has the fixed dimension
$m_I$. An important consequence of Theorem 3.2 is that **the matrix to be factorized, $H_{\mathrm{eff}}$,
lives on the column indices only.** Its fixed size is the same as in Chapter 4, whatever the
number of inequality rows.

Each QPAS iteration adds or removes one index of $\mathcal W$. There are now five cases:
the three expansions and two reductions below.

### 4.1 Expanding the working set

Let $q$ be the blocking index of the ratio test.

**(i) $q\in\mathcal N_x$** and **(ii) $q\in\mathcal S$.** The blocking index is a column.
The surplus sets, and hence $A_{\mathrm{eff}}$, are unchanged. Lemma 4.4.1 and
Theorems 4.4.2 and 4.4.4 apply word for word, with $A$ replaced by $A_{\mathrm{eff}}$ and
$H$ by $H_{\mathrm{eff}}$. In particular, in case (ii) the vector $u = A_{\mathrm{eff}}z_r$ is
used in (4.45)–(4.46). The surplus basis $Z_s$ is unchanged.

**(iii) $q = i\in\mathcal N_s$.** A free surplus reaches zero. The surplus sets change as

$$
\mathcal W_s^{k+1} = \mathcal W_s^k\cup\{i\},\qquad \mathcal N_s^{k+1} = \mathcal N_s^k\setminus\{i\},
$$

and the column sets are unchanged.

**Theorem 4.1.** Let $a_i^I$ denote row $i$ of $A_I$, written as a column vector, and define

$$
v_i := Z_x^\top a_i^I\in\mathbb R^{|\mathcal N_x|},\qquad (v_i)_j = (A_I)_{ij} - (A_I)_{i,\mathrm{rep}(j)}\quad(j\in\mathcal N_x).
$$

Then, in case (iii),

$$
Z_s^{k+1} = Z_s^k\ \text{without the column } e_i,\qquad
A_{\mathrm{eff}}^{k+1} = \begin{bmatrix} A_{\mathrm{eff}}^k\\ (a_i^I)^\top\end{bmatrix},\qquad
\boxed{\;H_{\mathrm{eff}}^{k+1} = H_{\mathrm{eff}}^k + v_iv_i^\top\;}
\tag{4.1}
$$

*Proof.* Removing $i$ from $\mathcal N_s$ removes the nullspace vector $e_i$ from $Z_s$.
Adding $i$ to $\mathcal W_s$ adds row $i$ of $A_I$ to $A_{\mathrm{eff}}$. Hence
$A_{\mathrm{eff}}^{k+1}Z_x = [A_{\mathrm{eff}}^kZ_x;\ v_i^\top]$, and
$H_{\mathrm{eff}}^{k+1} = (A_{\mathrm{eff}}^kZ_x)^\top(A_{\mathrm{eff}}^kZ_x) + v_iv_i^\top$. $\square$

The vector $v_i$ is nonzero only for the free columns $j$ for which exactly one of the two
paths, column $j$'s or its representative's, covers task $i$.

### 4.2 Reducing the working set

Let $r\in\mathcal W$ have a negative multiplier. By Corollary 3.7 the column multipliers and
the surplus multipliers are both measured in units of cost, so the most negative multiplier is
selected over $\mathcal W_x\cup\mathcal W_s$ as in Algorithm 4.3.

**(iv) $r\in\mathcal W_x$.** A column is released. The surplus sets are unchanged, and
Theorem 4.4.5 applies with $A$ replaced by $A_{\mathrm{eff}}$:

$$
Z_x^{k+1} = [Z_x^k\ \ z_r],\qquad u = A_{\mathrm{eff}}z_r,\qquad
H_{\mathrm{eff}}^{k+1} = \begin{bmatrix} H_{\mathrm{eff}}^k & (A_{\mathrm{eff}}Z_x^k)^\top u\\ u^\top A_{\mathrm{eff}}Z_x^k & u^\top u\end{bmatrix}.
$$

**(v) $r = i\in\mathcal W_s$.** An active surplus with $(\pi_I)_i < 0$ is released:

$$
\mathcal W_s^{k+1} = \mathcal W_s^k\setminus\{i\},\qquad \mathcal N_s^{k+1} = \mathcal N_s^k\cup\{i\}.
$$

**Theorem 4.2.** In case (v),

$$
Z_s^{k+1} = [Z_s^k\ \ e_i],\qquad
A_{\mathrm{eff}}^{k+1} = A_{\mathrm{eff}}^k\ \text{without the row } (a_i^I)^\top,\qquad
\boxed{\;H_{\mathrm{eff}}^{k+1} = H_{\mathrm{eff}}^k - v_iv_i^\top\;}
\tag{4.2}
$$

with $v_i$ as in Theorem 4.1.

*Proof.* This is the reverse of Theorem 4.1. $\square$

**Lemma 4.3.** After case (v), the next search direction satisfies $(p_s)_i > 0$. The released
surplus moves away from its bound.

*Proof.* Before the reduction $y_k$ is a subspace minimizer, so $Z^\top g = 0$. After the
reduction the only nonzero entry of the reduced gradient is the new one,
$(Z^\top g)_i = (\pi_I)_i < 0$. With (3.3),
$(p_s)_i = \eta_{s,i} = -\delta\,(\pi_I)_i\,(\bar H^{-1})_{ii} > 0$, since $\bar H^{-1}$ is positive
definite. $\square$

### 4.3 Updating the $LDL^\top$ factorization

All five cases change $H_{\mathrm{eff}}$ in a low-rank way, and each has a CHOLMOD counterpart:

| case | change of $H_{\mathrm{eff}}$ | CHOLMOD routine |
|---|---|---|
| (i) $q\in\mathcal N_x$ | row/column $q$ removed | `cholmod_l_rowdel` |
| (ii) $q\in\mathcal S$ | rows/columns of $\nu$ rewritten (4.58) | `cholmod_l_rowdel` + `cholmod_l_rowadd` |
| (iii) $q\in\mathcal N_s$ | $+\,v_iv_i^\top$ | `cholmod_l_updown` (update) |
| (iv) $r\in\mathcal W_x$ | row/column $r$ added | `cholmod_l_rowadd` |
| (v) $r\in\mathcal W_s$ | $-\,v_iv_i^\top$ | `cholmod_l_updown` (downdate) |

`cholmod_l_updown` belongs to the same MODIFY module as `rowadd` and `rowdel`, and it
updates an existing $LDL^\top$ factorization by a sparse rank-$k$ term. Cases (i), (ii) and (iv)
are identical to Chapter 4, except that the Hessian entries are computed with
$A_{\mathrm{eff}}$, i.e. with the rows of the free surpluses left out. Concretely, the new row in
case (iv) has the entries

$$
(H_{\mathrm{eff}})_{kr} = \sum_{\ell\in\mathcal E\cup\mathcal W_s}\big(a_k - a_{\mathrm{rep}(k)}\big)_\ell\,\big(a_r - a_{\mathrm{rep}(r)}\big)_\ell,
$$

where $\ell$ runs over the equality rows and the inequality rows with an active surplus.

**Remark 4.4 (relation to the full reduced Hessian).** Alternatively, one can factorize
$\bar H$ from (3.5) directly, with free surpluses as additional indices of the fixed-size
factorization. If these indices are ordered first, block elimination of the identity block
leaves exactly $H_{\mathrm{eff}}$ as the trailing block. A `rowadd` of surplus $i$ then performs
the downdate $-v_iv_i^\top$ on that block, and a `rowdel` performs the update. Both
realizations perform the same arithmetic. The formulation with $H_{\mathrm{eff}}$ is smaller
and keeps the factorization on the column indices. The formulation with $\bar H$ needs only
`rowadd`/`rowdel` and one fixed size $n_{\max} + m_I$.

### 4.4 Active-set method with efficient working-set updates

Algorithm 4.3 extends as follows. Only the marked lines differ from Chapter 4.

```text
Algorithm 4.4: Active-set method for the regularized master problem with inequality rows

Input: master problem data, working set W = W_x ∪ W_s, feasible iterate y = [x; s]
Initialize ordering [S, W_x, N_x | W_s, N_s], A_eff, and the LDLᵀ factor of H_eff
repeat
    compute π by (1.6)
  * solve (LDLᵀ) η_x = −δ Z_xᵀ (c − A_effᵀ π_eff);   p_x = Z_x η_x
  * (p_s)_i = (A_I p_x)_i − δ (π_I)_i  for i ∈ N_s,  0 for i ∈ W_s
    if ‖p‖ < ε then
        μ = g_S;  λ_x = reduced costs on W_x
      * λ_s = π_I on W_s
        λ_min = min(λ_x, λ_s);  if λ_min > −ε: optimal, stop
        if the minimum is attained in W_x: reduce the working set, case (iv)
      * else: reduce the working set, case (v):  H_eff ← H_eff − v_i v_iᵀ
    else
      * α = min{1, min over S ∪ N_x ∪ N_s with p_i < 0 of −y_i/p_i};   y ← y + αp
        if α < 1:
            if the blocking index is a column: expand, case (i) or (ii)
          * else: expand, case (iii):  H_eff ← H_eff + v_i v_iᵀ
until convergence
```

### 4.5 Singularity issues with the reduced Hessian

As in Section 4.4.5, expanding the working set cannot destroy positive definiteness. Cases (i)
and (ii) remove columns from $A_{\mathrm{eff}}Z_x$, and case (iii) adds a positive semidefinite
term. Singularity can only occur in the two reductions.

**Case (iv).** $H_{\mathrm{eff}}^{k+1}$ is singular if and only if $A_{\mathrm{eff}}z_r$ lies in the
span of the columns of $A_{\mathrm{eff}}Z_x$. This is the situation of Section 4.4.5, with $A$
replaced by $A_{\mathrm{eff}}$, and it is detected and resolved in the same way.

**Case (v).** By the matrix determinant lemma,

$$
\det\big(H_{\mathrm{eff}} - v_iv_i^\top\big) = \det(H_{\mathrm{eff}})\,\big(1 - v_i^\top H_{\mathrm{eff}}^{-1}v_i\big),
$$

so the downdated matrix is positive definite if and only if

$$
\omega_i := 1 - v_i^\top H_{\mathrm{eff}}^{-1}v_i > 0 .
$$

$\omega_i$ plays the role of the new diagonal pivot in case (iv). It is available after
the downdate as the change of the determinant, and is detected by inspecting $D$, as in
Section 4.4.5. If $\omega_i = 0$, let $t = H_{\mathrm{eff}}^{-1}v_i$. Then $v_i^\top t = 1$ and
$(H_{\mathrm{eff}} - v_iv_i^\top)\,t = 0$, so $t$ spans the kernel. The rule of Section 4.4.5
carries over: move $k = \arg\max_{j\in\mathcal N_x}|t_j|$ from $\mathcal N_x$ to
$\mathcal W_x$ (case (i)) and retry the reduction. Deleting row and column $k$ with $t_k\ne 0$
removes the kernel vector, so the retried downdate is positive definite.

**Remark 4.5.** As noted in the implementation notes, moving $k$ into the working set while
$x_k > 0$ violates $x_{\mathcal W} = 0$. The zero-curvature step used in the implementation
avoids this, and it extends to case (v). The direction $d$ with
$d_x = Z_xt$, $(d_s)_\ell = (A_Id_x)_\ell$ for $\ell\in\mathcal N_s\cup\{i\}$ and $d_s = 0$ otherwise
satisfies $D_{\mathcal W\setminus\{i\}}d = 0$, $\bar Ad = 0$ and $g^\top d = (\pi_I)_i < 0$. The
objective therefore decreases linearly along $d$ until a variable blocks. That variable enters
the working set, and the reduction is retried.

---

## 5. Practical Hot-Start Strategy for the Master Problem

### 5.1 Dimensions

The column weights use the fixed dimension $n_{\max}$ of Section 4.5.1. The surpluses add a
fixed block of dimension $m_I$, which is known from the start and never changes. The
factorization of $H_{\mathrm{eff}}$ keeps the dimension $n_{\max}$ (Section 4.3), so the
fixed-size $LDL^\top$ object of Chapter 4 can be reused without modification.

### 5.2 Initial iterate

The QPAS needs a feasible point of (2.1), i.e. $\bar Cy = \mathbf 1$ and $y\ge 0$. A natural
choice is the initial point of Chapter 4 for $x$, together with

$$
s = 0,\qquad \mathcal W_s = \{1,\dots,m_I\},\qquad \mathcal N_s = \emptyset .
$$

Then $A_{\mathrm{eff}} = A$ and $H_{\mathrm{eff}} = H$, so the initial factorization is the one of
Chapter 4. The first multiplier check releases the surpluses of all rows with $(\pi_I)_i < 0$
(case (v)). By (1.6), these are the rows that are covered by more than
$b_i + \delta(\hat\pi_I)_i$.

### 5.3 Hot-starting the regularized master problem

Between two column generation iterations, the changes are as in Section 4.5.2:

* **A new column** enters $\mathcal W_x$ with weight $0$. $Z_x$, $A_{\mathrm{eff}}$,
  $H_{\mathrm{eff}}$ and its factorization are unchanged. Its multiplier is the reduced cost found
  by the pricing problem (Corollary 3.7), so it is negative, and the column is typically
  released at the start of the next solve.
* **A new center** $\hat\pi$ changes only $\tilde c$, and hence $\pi$ by (1.6). It changes
  neither the ordering nor $H_{\mathrm{eff}}$, because $H_{\mathrm{eff}}$ depends on $A$ and the
  working set only.
* **The surpluses** are never added or removed. Their working-set status is carried over from
  one master problem to the next, exactly like that of the columns.

### 5.4 Column generation

Algorithm 4.1 is unchanged.

* **Pricing** uses $\pi = [\pi_E;\pi_I]$ from (1.6). The reduced cost of a new column is
  computed by the same formula as before, and the duals of the inequality rows enter the edge
  weights like any other dual.
* **Termination.** The run stops when the master problem is optimal, $\pi = \hat\pi$, and pricing
  finds no column with negative reduced cost. By Remark 3.8, $\pi$ is then an optimal
  dual solution of (1.2), including the sign condition (1.3).
* **Lagrangian bound.** The bound of Section 2.2 is valid only if $\pi_I\ge 0$, because a
  negative entry gives $\min_{v_i\ge b_i}(\pi_I)_iv_i = -\infty$. The QPAS guarantees
  $\pi_I\ge -\varepsilon$ only up to its tolerance. The bound should therefore be evaluated
  at $[\pi_E;\max(\pi_I, 0)]$, which is always a valid dual point.

---

## Summary of the extension

| | Chapter 4 (equality rows) | this chapter (with inequality rows) |
|---|---|---|
| variables | $x$ | $y = [x; s]$ |
| linking matrix | $A$ | $\bar A = \begin{bmatrix} A_E & 0\\ A_I & -I\end{bmatrix}$ |
| convexity matrix | $C$ (column generation properties) | $\bar C = [C\ \ 0]$ (properties lost) |
| ordering | $[\mathcal S,\mathcal W,\mathcal N]$ | $[\mathcal S,\mathcal W_x,\mathcal N_x\mid\mathcal W_s,\mathcal N_s]$ |
| nullspace basis | $Z$ | $\mathrm{diag}(Z_x, Z_s)$ |
| gradient | $c - A^\top\pi$ | $[\,c - A^\top\pi;\ \pi_I\,]$ |
| reduced Hessian to factorize | $H = (AZ)^\top AZ$ | $H_{\mathrm{eff}} = (A_{\mathrm{eff}}Z_x)^\top A_{\mathrm{eff}}Z_x$, $A_{\mathrm{eff}} = [A_E; W_sA_I]$ |
| surplus direction | – | $(p_s)_i = (A_Ip_x)_i - \delta(\pi_I)_i$ on $\mathcal N_s$ |
| multipliers | $\mu = h_{\mathcal S}$, $\lambda = h_{\mathcal W} - C_{\mathcal W}^\top\mu$ | in addition $\lambda_s = h_{\mathcal W_s}$ ($=\pi_I$ at $p=0$) |
| factorization updates | `rowadd`, `rowdel` | in addition rank-one `updown` with $v_i$ |
| hot start, pricing, termination | – | unchanged; bound evaluated at $\max(\pi_I,0)$ |

The identities of Lemma 2.1, Theorem 3.2, Corollaries 3.3 and 3.7, and Theorems 4.1/4.2 were
checked numerically on a random instance with 3 equality rows, 5 inequality rows and 10
columns in 2 blocks (3 free columns, 2 free surpluses). All agree to machine precision.
