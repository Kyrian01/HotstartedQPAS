# The Dense LDLᵀ Backend of the Hot-Started QPAS

*A companion to [`qpas_mathematical_notes.md`](qpas_mathematical_notes.md) §4 and
[`qpas_inequality_extension.md`](qpas_inequality_extension.md) §4.3*

The QPAS keeps an $LDL^\top$ factorization of the effective reduced Hessian
$H_{\mathrm{eff}} = (A_{\mathrm{eff}}Z)^\top(A_{\mathrm{eff}}Z)$ and updates it every time the
working set changes. Two interchangeable backends implement this factor:

| backend | code | what is stored |
|---|---|---|
| `:cholmod` (thesis method) | `src/backends/cholmod/` | sparse simplicial $LDL^\top$ of a fixed $n_{\max}\times n_{\max}$ matrix that is the identity outside $N$ |
| `:dense` (this note) | `src/backends/dense/DenseLDLt.jl` | dense $LDL^\top$ of the active $|N|\times|N|$ block only |

Select the backend per run:

```julia
QPASMaster(factorization = :dense)                     # Julia
```

```toml
[defaults.qpas]                                        # experiments.toml
factorization           = "dense"   # "cholmod" | "dense"
dense_refactor_interval = 0         # rebuild every k updates (0 = never)
```

This note explains why a dense factor fits the problem, states every update rule with its
derivation and cost, and reports the measurements. The notation follows
`qpas_mathematical_notes.md` §0: $u_j = A z_j = a_j - a_{\mathrm{rep}(j)}$ are the columns of
$AZ$, and $H_{kj} = u_k^\top u_j$ for $k, j \in N$.

---

## 1. Why a dense factor

### 1.1 The active block is at most $m \times m$

**Proposition 1.** While $H_{\mathrm{eff}}$ is positive definite, $|N| \le \operatorname{rank}(A_{\mathrm{eff}}) \le m$.

*Proof.* $H_{\mathrm{eff}} = U^\top U$ with $U = A_{\mathrm{eff}}Z_N \in \mathbb{R}^{m \times |N|}$.
It is positive definite iff $U$ has full column rank $|N|$, and a matrix with $m$ rows has
rank at most $m$. $\square$

The QPAS keeps every pivot above `singular_tolerance` (`qpas_mathematical_notes.md` §5), so
the proposition holds throughout. A dependent column is added and immediately undone, so at
most $m + 1$ indices are active at any moment. The factor size therefore depends on the
number of coupling rows $m$, never on the column capacity $n_{\max}$. On I3, $m = 356$ and
$|N|$ reaches 353 against $n_{\max} = 20\,000$.

### 1.2 The active block is dense

$H_{kj} = u_k^\top u_j \ne 0$ whenever $u_k$ and $u_j$ share a row. In the crew scheduling
instances, two kinds of overlap make almost every pair share one:

- **Within a block**, every $u_j = a_j - a_{\mathrm{rep}(b)}$ contains the representative's
  column $a_{\mathrm{rep}(b)}$. So $u_k^\top u_j \ne 0$ for all $k, j$ of the same pilot,
  unless the shared part cancels exactly.
- **Across blocks**, schedules of different pilots cover overlapping tasks, since all of
  them compete for the same coupling rows.

So $H_{\mathrm{eff}}$ is dense, and so is its factor; the numbers in §5.4 confirm it. A sparse
factor of a dense matrix stores the same numbers as a dense one, plus indices, column
pointers and an $n_{\max}$-sized embedding. Its operations walk those index structures
instead of streaming through contiguous memory.

### 1.3 Structural garbage

`cholmod_rowdel` sets a row and column of $L$ to zero numerically, but keeps the entries in
the sparsity pattern of $L$. `cholmod_updown` does the same with entries that cancel. Every
later solve and update pays for these stored zeros, so the CHOLMOD backend must be rebuilt
periodically (`refactor_interval = 150`, `qpas_mathematical_notes.md` §4.4). A dense factor
that *compacts* on deletion never holds a stored entry that is not part of the current
matrix, so it needs no periodic rebuild.

---

## 2. Data structure

With $k$ active indices, the dense factor stores

$$
M[\mathcal{A},\mathcal{A}] \;=\; P\,L\,D\,L^\top P^\top,
\qquad L\in\mathbb{R}^{k\times k}\ \text{unit lower triangular},\quad D=\operatorname{diag}(d_1,\dots,d_k),
$$

where $\mathcal{A}$ is the active index set ($= N$) and $P$ maps **slots** $1..k$ to indices:

| array | size | meaning |
|---|---|---|
| `L` | $\mathrm{cap}\times\mathrm{cap}$, column-major | strictly lower part of the leading $k\times k$ block is $L$ (diagonal and upper part are never read) |
| `d` | $\mathrm{cap}$ | $D$ |
| `index[s]` | $\mathrm{cap}$ | index held by slot $s$ |
| `slot[j]` | $n_{\max}$ | slot of index $j$, $0$ if $j$ is inactive (an identity row of $M$) |
| `rhs`, `sol` | $n_{\max}$ | right-hand side / solution in index space (the QPAS writes $-\delta Z^\top g$ into `rhs` by index) |
| `w`, `z` | $\mathrm{cap}$ | slot-space scratch vectors |

**Slots are kept in activation order.** A new index is always appended as slot $k+1$. When
slot $p$ is deleted, slots $p+1..k$ move down by one. The capacity starts at $m + 1$
(`make_factor`). It doubles if ever exceeded, which never happened in any run (§5.4).

Every inactive index is an identity row of $M$ and is not stored. `factor_matrices` (debugging
only) returns the factor in index space as $M = G D G^\top$ with
$G[\mathrm{index}[r], \mathrm{index}[c]] = L_{rc}$, a row and column permutation of a unit
lower-triangular matrix.

---

## 3. Update rules

Throughout, $k$ is the number of active slots before the operation.

### 3.1 Activate an index: append a row (`rowadd!`)

The QPAS activates index $j$ (a reduction, thesis §4.4.2) with the new column
$h = (H_{ij})_{i \in N}$ and the diagonal entry $h_{jj} = \|u_j\|^2$ (both over the rows of
$A_{\mathrm{eff}}$). Let $\tilde h = P^\top h$ be $h$ in slot order. We seek

$$
\begin{bmatrix} L & 0\\ \ell^\top & 1\end{bmatrix}
\begin{bmatrix} D & 0\\ 0 & d_{k+1}\end{bmatrix}
\begin{bmatrix} L^\top & \ell\\ 0 & 1\end{bmatrix}
=
\begin{bmatrix} LDL^\top & LD\ell\\ \ell^\top D L^\top & \ell^\top D\ell + d_{k+1}\end{bmatrix}
\stackrel{!}{=}
\begin{bmatrix} M_{\mathcal{A}\mathcal{A}} & \tilde h\\ \tilde h^\top & h_{jj}\end{bmatrix}.
$$

**Rule.**

$$
L\,w = \tilde h \quad(\text{forward substitution}),\qquad
\ell = D^{-1}w,\qquad
d_{k+1} = h_{jj} - w^\top D^{-1} w .
$$

The existing factor is untouched. **Cost:** one unit-lower triangular solve (BLAS `trsv`),
$O(k^2)$ flops.

**The new pivot is the singularity test.** $d_{k+1} = h_{jj} - \tilde h^\top M_{\mathcal{AA}}^{-1}\tilde h$
is the Schur complement of the new index, and with $H = U^\top U$ it equals

$$
d_{k+1} = \|u_j\|^2 - u_j^\top U (U^\top U)^{-1}U^\top u_j = \|(I - \Pi_U)\,u_j\|^2,
$$

the squared distance of $u_j$ from the span of the active columns. The scaled pivot
$d_{k+1}/\|u_j\|^2 = \sin^2\theta$ ($\theta$ = angle between $u_j$ and that span) is therefore
exactly the quantity that `reduce_working_set!` compares with `singular_tolerance`. CHOLMOD
inserts $j$ at its index position instead, and then also has to downdate the trailing
columns. Its smallest pivot detects the same singularity, but indirectly.

### 3.2 Deactivate an index: delete a slot (`rowdel!`)

An expansion (thesis §4.4.1) removes index $j$ from $N$. Let $p = \mathrm{slot}[j]$ and
partition the slots into $1..p-1 \mid p \mid p+1..k$:

$$
L = \begin{bmatrix} L_{11} & 0 & 0\\ \ell_{21}^\top & 1 & 0\\ L_{31} & \ell_{32} & L_{33}\end{bmatrix},
\qquad D = \operatorname{diag}(D_1, d_p, D_3).
$$

Multiplying out gives $M_{11} = L_{11}D_1L_{11}^\top$, $M_{31} = L_{31}D_1L_{11}^\top$ and

$$
M_{33} = L_{31}D_1L_{31}^\top + d_p\,\ell_{32}\ell_{32}^\top + L_{33}D_3L_{33}^\top .
$$

Deleting row and column $p$ leaves $\begin{bmatrix} M_{11} & M_{13}\\ M_{31} & M_{33}\end{bmatrix}$,
whose factor is

$$
\begin{bmatrix} L_{11} & 0\\ L_{31} & \tilde L_{33}\end{bmatrix},\quad
\operatorname{diag}(D_1, \tilde D_3),\qquad
\tilde L_{33}\tilde D_3\tilde L_{33}^\top = L_{33}D_3L_{33}^\top + d_p\,\ell_{32}\ell_{32}^\top .
$$

**Rule.** $L_{11}$ and $L_{31}$ are kept. The trailing block receives a **rank-one update** with
$\alpha = d_p > 0$ and $z = \ell_{32}$ (method C1, §3.3). Then every slot after $p$ moves one
position up and to the left.

The implementation fuses the two steps: C1 reads column $p+t$ of the old trailing block and
writes the updated column to position $p+t-1$. Column $p+t-1$ was fully read in the previous
step, so the update can run in place. The rows $p+1..k$ of $L_{31}$ are shifted up by one.

**Cost:** $O((k-p)^2)$ for the update plus $O(p\,(k-p))$ memory moves for $L_{31}$. Deleting
the newest slot costs nothing beyond bookkeeping. Deleting the oldest costs a full $O(k^2)$
update, which is the same as `cholmod_rowdel` deleting the first index. The benchmark
records the mean relative position $(k-p)/k$ (`mean_trailing_fraction`).

**Deletion never decreases a pivot.** With $\alpha > 0$, C1 produces
$\bar d_j = d_j + a\,p_j^2 \ge d_j$, and $a$ stays positive. An expansion therefore cannot
create a small pivot: the invariant "all pivots of $N$ above `singular_tolerance`" carries
over from the reduction that established it.

### 3.3 Rank-one update and downdate (`updown!`): method C1

An inequality slack entering or leaving the working set changes
$H_{\mathrm{eff}} \leftarrow H_{\mathrm{eff}} \pm vv^\top$, with $v$ the row of $AZ$ of that
inequality (`qpas_inequality_extension.md` Thm. 4.1/4.2). Method C1 of Gill, Golub, Murray and
Saunders (1974) computes $\bar L\bar D\bar L^\top = LDL^\top + \alpha zz^\top$ column by column:

$$
\begin{aligned}
&a \leftarrow \alpha,\quad w \leftarrow z\\
&\textbf{for } j = 1,\dots,k:\\
&\qquad p_j = w_j,\quad \bar d_j = d_j + a\,p_j^2,\quad \beta_j = p_j a/\bar d_j,\quad a \leftarrow d_j a/\bar d_j\\
&\qquad \textbf{for } r = j+1,\dots,k:\quad w_r \leftarrow w_r - p_j\,\ell_{rj},\qquad \bar\ell_{rj} = \ell_{rj} + \beta_j w_r .
\end{aligned}
$$

*Why it works (sketch).* Write
$LDL^\top + \alpha zz^\top = L\,(D + \alpha \tilde z\tilde z^\top)\,L^\top$ with $L\tilde z = z$.
Step $j$ splits off the first row and column of the bracket, which updates pivot $j$, and
leaves a rank-one modification of the trailing part with weight
$a_{\text{new}} = d_j a/\bar d_j$. The inner loop computes $\tilde z$ by forward substitution
and folds it into the new column at the same time.

**Rule for `updown!`.** $\alpha = +1$ (update) or $-1$ (downdate), $z = P^\top v$. The loop
starts at the first slot where $z \ne 0$: a column with $p_j = 0$ leaves $d_j$, $\ell_{\cdot j}$
and $a$ unchanged, so leading zeros of $v$ cost nothing. **Cost:** $O((k-j_0)^2)$, with $j_0$
that first slot.

**Numerics.**
- *Update* ($\alpha > 0$): $\bar d_j \ge d_j > 0$ and $a$ stays positive, so nothing cancels.
  This is the stable case, and it also covers every deletion (§3.2).
- *Downdate* ($\alpha < 0$): $\bar d_j = d_j - |a|\,p_j^2$ can cancel. If some
  $\bar d_j \le 0$, the downdated matrix is not positive definite: `updown!` returns `false`
  and the caller rebuilds the factor. This mirrors CHOLMOD's `CHOLMOD_NOT_POSDEF`.
  A singular but representable downdate shows up as a tiny pivot, which `reduce_surplus!`
  catches through `min_pivot_ratio` (`qpas_inequality_extension.md` §4.5).

### 3.4 Solve (`solve!`)

The QPAS solves $H_{\mathrm{eff}}\,y = -\delta Z^\top \tilde g$ with a right-hand side
supported on $N$. Gathering into slot order and scattering back:

$$
\tilde b = P^\top b,\qquad L z = \tilde b,\qquad z \leftarrow D^{-1}z,\qquad L^\top x = z,\qquad y = P x .
$$

These are two BLAS `trsv` calls on the leading $k\times k$ block plus a diagonal scaling, in
$O(k^2)$. The pattern variant `solve!(F, pattern, n)` reads `rhs` only on the pattern and
returns `sol` on $N$, as the CHOLMOD `Bset` solve does, but with no $n_{\max}$-dependent work.

### 3.5 Rebuild, pivots and reset

- `reset!` clears the `slot` map of the active indices, in $O(k)$. Stored entries do not need
  zeroing, because §3.1 writes every entry of a new row before it is read.
- `refactor!` (in `kkt_system.jl`, backend-independent) re-activates every index of $N$, i.e.
  $k$ appends, for $O(k^3)$ in total. For the dense backend it runs only on a decayed pivot,
  after a failed downdate, or every `dense_refactor_interval` updates (default 0 = never).
- `pivot(F, j)` is $d_{\mathrm{slot}[j]}$, or $1$ for an inactive index.
  `min_scaled_pivot(F, N, scale)` is $\min_{j\in N} d_{\mathrm{slot}[j]}/\|u_j\|^2$.

### 3.6 Summary

| operation | QPAS event | dense rule | cost | CHOLMOD |
|---|---|---|---|---|
| `rowadd!(j)` | reduction (index enters $N$), refactor | append slot $k+1$, §3.1 | $O(k^2)$ | insert at index $j$: solve and downdate the trailing columns |
| `rowdel!(j)` | expansion (index leaves $N$), representative swap | delete slot $p$ with a rank-one update and shift, §3.2 | $O((k-p)^2 + p(k-p))$ | `rowdel`: rank-one update, **entries stay in the pattern** |
| `updown!(±v)` | slack enters / leaves $W_s$ | C1 from the first nonzero, §3.3 | $O(k^2)$ | `updown`: pattern grows, entries stay |
| `solve!` | direction, singular step | 2 × `trsv`, §3.4 | $O(k^2)$ | `solve2` with `Bset`, plus an $n_{\max}$ workspace per call (§5.3) |
| periodic rebuild | – | not needed | – | every 150 updates |
| memory | – | $8\,\mathrm{cap}^2$ B $+\ O(n_{\max})$ index arrays | 1.4 MB on I3 ($m = 356$) | 11.3 MB on I3 |

---

## 4. Integration

- `src/backends/ldlt_interface.jl` declares the abstract type `UpdatableLDLt` and the generic
  operations. `Cholmod.LDLtFactor` and `DenseLDLt.DenseLDLtFactor` both implement them.
- `KKTSystem{F}` and `QPASState{F}` are parametric in the factor type, so the solver code is
  compiled separately for each backend, with no dynamic dispatch in the hot loop.
  `make_factor(:dense, n_max, m)` starts with $m+1$ slots.
- `refactor_interval(options)` returns `refactor_interval` for CHOLMOD and
  `dense_refactor_interval` for the dense backend.
- **Statistics.** `KKTSystem.timer` records time and call count of every factor operation
  (`assemble` = building a Hessian column or rank-one vector from the pool, `rowadd`, `rowdel`,
  `updown`, `solve`). `print_details` lists them as µs per call.
  `LDLt.backend_stats(F)` adds capacity, peak active size, reallocations, deletions and
  `mean_trailing_fraction`. The diagnostics traces (`CGOptions(diagnostics_every = k)`) record
  `factor_MB`, the stored and nonzero entries of $L$, and with `diagnostics_residual = true`
  the residual $\max|(LDL^\top)_{NN} - H_{\mathrm{eff}}|/\max(1,\max|H_{\mathrm{eff}}|)$, with
  $H_{\mathrm{eff}}$ rebuilt from the pool.
- **Tests.** `test/test_dense_ldlt.jl` covers each rule against dense reference matrices,
  deletion of the first, a middle and the last slot, capacity growth, and identical random
  operation sequences on both backends. Every test in `test/test_qpas.jl` runs with both
  backends, including debug mode, which verifies the factor after every QPAS iteration.
  `test/test_solvers.jl` checks end-to-end exactness against the full LP for both backends
  and the agreement of the two.

---

## 5. Measurements

All numbers are from `scripts/benchmark_factorization.jl` (AMD Zen 4, 16 threads, Julia 1.12.5,
one BLAS thread, CHOLMOD from SuiteSparse_jll, $n_{\max} = 20\,000$, $\delta = 1$). Timings are
the best of 3 repetitions (small instances) or 2 (large). Between repetitions, runs under a
few seconds varied by up to 20 %, longer runs by 1–5 %. The full report with every table
is `results/factorization/report.md` after running the script.

### 5.1 Exactness (small instances, against the full LP)

Both backends terminate `optimal` on every instance and pricing rule. Their objectives agree
with each other and with the exact LP to the accuracy the stabilized termination allows:

| instance | m | pricing | iterations CHOLMOD / dense | $c^\top\lambda$ − exact (CHOLMOD / dense) | master obj − exact | infeasibility |
|---|---|---|---|---|---|---|
| I1 | 38 | first improving | 3928 / 3927 | −2.8e−13 / −9.9e−13 | ≈ 1e−13 (both) | 3e−14 |
| I1 | 38 | all blocks | 4042 / 4043 | −6.8e−13 / −2.1e−12 | ≈ 1e−13 (both) | 4e−14 |
| I2 | 225 | first improving | 11974 / 11974 | −2.5e−2 / −2.6e−2 | −1.9e−4 (both) | 6e−5 |
| I2 | 225 | all blocks | 12123 / 12123 | −3.1e−2 / −2.9e−2 | −1.9e−4 (both) | 7e−5 |
| I3 | 356 | first improving | 8961 / 8879 | 1.6e−5 / −2.0e−5 | −1.0e−5 (both) | 1.4e−5 |
| I3 | 356 | all blocks | 5883 / 5885 | −1.2e−3 / −1.1e−3 | −2.5e−5 (both) | 1.4e−5 |
| I2_ineq | 225 (≥ rows) | first improving | 1070 / 1033 | 1.8e−11 / 1.5e−11 | ≈ 1e−11 (both) | 4e−13 |
| I2_ineq | 225 (≥ rows) | all blocks | 562 / 537 | 1.2e−11 / 1.3e−11 | ≈ 1e−11 (both) | 7e−13 |

The visible gaps on I2 and I3 are the same for both backends. They come from stopping at
`center_tolerance = 1e-6`: the final $\lambda$ satisfies $A\lambda = b$ only to about 1e−5,
so $c^\top\lambda$ can lie slightly below the optimum (README, "Results"). The master
objective is within $2\cdot10^{-7}$ relative of the optimum. Where the iteration counts
differ, they differ by under 5 %: round-off differences break ties in the ratio test
differently, and the run continues from there on a slightly different path.

### 5.2 Speed

| instance | pricing | total CHOLMOD [s] | total dense [s] | master CHOLMOD [s] | master dense [s] | master speedup |
|---|---|---|---|---|---|---|
| I1 | first improving | 0.042 | 0.020 | 0.033 | 0.011 | 2.9× |
| I1 | all blocks | 0.054 | 0.030 | 0.041 | 0.017 | 2.4× |
| I2 | first improving | 2.58 | 1.23 | 2.35 | 1.00 | 2.3× |
| I2 | all blocks | 2.20 | 1.14 | 1.94 | 0.90 | 2.2× |
| I3 | first improving | 7.49 | 2.85 | 7.18 | 2.58 | 2.8× |
| I3 | all blocks | 3.31 | 1.38 | 3.02 | 1.15 | 2.6× |
| I2_ineq | first improving | 0.62 | 0.28 | 0.60 | 0.26 | 2.3× |
| I2_ineq | all blocks | 0.150 | 0.074 | 0.125 | 0.056 | 2.2× |
| L1 (m = 480) | all blocks | 3.84 | 1.22 | 3.58 | 1.05 | 3.4× |
| L2 (m = 762) | all blocks | 77.0 | 22.6 | 72.1 | 18.0 | 4.0× |
| L3 (m = 1137) | all blocks | 119.6 | 29.8 | 115.0 | 25.2 | 4.6× |
| L2_ineq (m = 762, 229 ≥ rows) | all blocks | 18.8 | 7.5 | 17.1 | 5.9 | 2.9× |

**The speedup grows with $m$**, from about 2.5× at $m \approx 200$–$350$ to 4.6× at
$m = 1137$, because every operation scales as $|N|^2$ and CHOLMOD's constant per stored entry
is several times the dense one (§5.5). CHOLMOD's periodic rebuilds also become a large share
of its time: 27 s of 72 s on L2 (587 rebuilds) and 51 s of 115 s on L3 (631 rebuilds). The
dense backend rebuilt the factor 1–3 times per run, taking 0.05 s in total.

On the large instances, which have no exact reference, the two backends agree: identical status
(`optimal`), iteration counts within 10 of each other, objectives equal to $3\cdot10^{-10}$
(L2) to $6\cdot10^{-8}$ (L3) relative, and the same coupling infeasibility (1e−5 to 6e−5,
set by `center_tolerance`).

Where the time goes on I3 (first improving), in seconds:

| section | CHOLMOD | dense |
|---|---|---|
| direction (solve) | 3.14 | 0.90 |
| multipliers ($g_j$ on $W$) | 0.88 | 0.87 |
| primal step | 0.35 | 0.35 |
| expand (`rowdel`) | 0.97 | 0.08 |
| reduce (assemble + `rowadd`) | 1.73 | 0.29 |
| of expand + reduce: refactorizations | 1.23 (509 rebuilds) | 0.007 (5 rebuilds) |

Per operation on I3 (first improving): a solve takes 35 µs with CHOLMOD and 7.0 µs dense,
`rowadd` 14 µs and 2.8 µs, `rowdel` 8.1 µs and 1.3 µs. On L3: solve 358 µs and 64 µs,
`rowadd` 120 µs and 28 µs, `rowdel` 55 µs and 11 µs.

**The next bottleneck is no longer the factor.** With the dense backend, the multiplier check
(computing $g_j = c_j - a_j^\top\pi$ on all of $W$), which does not touch the factor, costs as
much as the direction on I3 and more than it on L2 and L3 (L3: multipliers 10.5 s, direction
8.4 s, expand plus reduce 3.3 s). Assembling a Hessian column from the pool now costs about
as much as adding it to the factor.

### 5.3 Memory and allocations

| | CHOLMOD | dense |
|---|---|---|
| factor memory, peak (I1 / I2 / I3) | 5.6 / 7.2 / 11.3 MB | 0.47 / 0.85 / 1.44 MB |
| factor memory, peak (L1 / L2 / L3) | 17.1 / 37.5 / 88.2 MB | 2.2 / 4.9 / 10.4 MB |
| Julia-tracked allocations over a run (I1 / I2 / I3, first improving) | 0.75 / 7.3 / 14.3 GB | 4 / 10 / 13 MB |
| Julia-tracked allocations (L2 / L3) | 28.8 / 37.3 GB | 38 / 45 MB |
| GC time (I3 / L3) | 0.25 / 1.32 s | 0 / 0 s |

The allocations come from CHOLMOD itself, whose `malloc` Julia routes through its own
allocator: `cholmod_l_solve2` with a `Bset` allocates an $n_{\max}$-length workspace on every
call. Measured directly with 30 active rows:

| $n_{\max}$ | allocated per pattern solve | time per pattern solve |
|---|---|---|
| 2 000 | 16 kB | 1.6 µs |
| 20 000 | 160 kB | 4.2 µs |
| 200 000 | 1.6 MB | 14.4 µs |

So the CHOLMOD solve does depend on $n_{\max}$, even though its arithmetic is restricted to
the reach of $N$. The dense backend allocates nothing per operation. Its $O(n_{\max})$ arrays
(`slot`, `rhs`, `sol`) are allocated once.

### 5.4 Factor contents and accuracy over a run

Each diagnostics run samples the factor about 60 times along the column generation run:

| instance (first improving) | max $\lvert N\rvert$ | entries stored, CHOLMOD (max) | dense triangle $\lvert N\rvert(\lvert N\rvert-1)/2$ | true fill of $L$ (min over samples) | max residual CHOLMOD | max residual dense |
|---|---|---|---|---|---|---|
| I1 | 35 | 1 435 | 595 | 0.98 | 5.7e−16 | 5.3e−16 |
| I2 | 219 | 33 872 | 23 871 | 0.998 | 2.2e−15 | 1.3e−15 |
| I3 | 355 | 72 309 | 62 835 | 0.993 | 2.3e−15 | 1.8e−15 |
| I2_ineq | 157 | 13 684 | 12 246 | 0.92 | 1.1e−15 | 1.4e−15 |
| L1 (all blocks) | 478 | 129 108 | 114 003 | 0.996 | 2.5e−15 | 2.1e−15 |
| L2 (all blocks) | 757 | 321 795 | 286 146 | 0.998 | 3.4e−15 | 2.5e−15 |
| L3 (all blocks) | 1130 | 675 196 | 637 885 | 0.999 | 4.1e−15 | 3.6e−15 |
| L2_ineq (all blocks) | 671 | 247 246 | 224 785 | 0.999 | 3.2e−15 | 2.8e−15 |

- **Density.** The true factor of $H_{\mathrm{eff}}$ is 92–100 % dense in every sample, which
  confirms §1.2. With inequality rows it is slightly sparser, because masked rows remove
  overlap.
- **Garbage.** CHOLMOD stores up to 3.2× the entries of the dense triangle between rebuilds
  (up to 90 % of the stored entries are zeros on I2_ineq). The dense factor stores exactly the
  triangle at all times.
- **Accuracy.** The residual $\|LDL^\top - H_{\mathrm{eff}}\|_{\max}/\max(1,\|H_{\mathrm{eff}}\|_{\max})$
  stays at round-off level for the whole run, for both backends. The dense factor is
  rebuilt only when a singular-Hessian event forces it (1–6 times per run) and does not
  drift in between, even over 120 000 active-set iterations.
- **Capacity.** The $m+1$ initial slots were never exceeded (peak active size ≤ $m+1$, no
  reallocation in any run). The
  mean deletion position $(k-p)/k$ was 0.21–0.35: most deletions remove recently activated
  indices, so the average deletion touches only about a quarter of the factor.

### 5.5 Microbenchmark: cost per operation versus active size

$H$ is the Gram matrix of $k+1$ random sparse ±1 columns (dense in practice). The table shows
the median µs per operation; memory is in MB.

| operation | backend | k = 100 | k = 350 | k = 1000 | k = 2000 |
|---|---|---|---|---|---|
| solve | CHOLMOD | 7.9 | 68 | 511 | 3 196 |
| solve | dense | 2.6 | 15 | 95 | 370 |
| append a row | CHOLMOD | 4.2 | 41 | 307 | 1 530 |
| append a row | dense | 1.2 | 7.1 | 48 | 188 |
| delete the first row | CHOLMOD | 3.1 | 26 | 204 | 903 |
| delete the first row | dense | 1.3 | 11.5 | 103 | 413 |
| delete a middle row | CHOLMOD | 1.6 | 9.9 | 67 | 303 |
| delete a middle row | dense | 0.8 | 6.1 | 53 | 198 |
| rank-one update | CHOLMOD | 3.2 | 27 | 204 | 968 |
| rank-one update | dense | 1.2 | 10.4 | 88 | 340 |
| rebuild (k appends) | CHOLMOD | 757 | 10 812 | 137 887 | 1 094 454 |
| rebuild (k appends) | dense | 54 | 1 157 | 20 120 | 133 322 |
| memory [MB] | CHOLMOD | 5.6 | 10.0 | 62 | 257 |
| memory [MB] | dense | 0.56 | 1.5 | 8.3 | 31 |

The dense backend is faster for every operation at every size: by 4–9× for solves and appends,
2–3× for deletions and rank-one updates, and about 8× for rebuilds. The gap grows with $k$,
because CHOLMOD's per-entry index handling costs more than the dense kernels' streaming access.

---

## 6. When CHOLMOD would still be the better choice

The dense backend wins because $H_{\mathrm{eff}}$ is dense and $|N| \le m$ is moderate. A dense
factor costs $8m^2$ bytes and $O(m^2)$ per operation regardless of sparsity. If an application
had many thousands of coupling rows *and* a genuinely sparse $H_{\mathrm{eff}}$ (columns that
rarely share rows, e.g. many independent blocks with disjoint task sets), CHOLMOD's sparse
factor could be smaller and faster. Measure that case before deciding; it is one option
(`factorization = "cholmod"`) away.
