# Adaptive Stabilization for the Hot-Started QPAS

*A companion to the thesis, §3.5 (An Adaptive Stabilization Parameter) and §9.9 (its
measurements); see also §6.6 (hot start, center-only iterations) and §9.4 (the two phases of a
run). The thesis tables are produced by section F of `thesis/scripts/generate_data.jl`; this
note has the full variant sweep.*

With a fixed stabilization parameter $\delta$, regularized column generation spends most of its
iterations in a **proximal phase**: pricing finds no column, the center moves to the current
dual, and the master is solved again (thesis §3.4, Tab. 9.4). On I2 (equality variant) these
center-only iterations are 10\,717 of 11\,974 iterations. This note gives the theory for
changing $\delta$ during a run, derives a geometric update rule, and measures it. The rule is
implemented for the QPAS master:

```julia
StabilizationOptions(delta = 1.0,                  # base value δ₀ (the largest δ used)
                     delta_update = :geometric,    # :fixed (thesis, the default) | :geometric
                     delta_decrease = 0.5,         # δ ← max(δ_min, 0.5·δ) after a center-only iteration
                     delta_increase = 2.0,         # δ ← min(δ₀, 2·δ) after a column; Inf = reset to δ₀
                     delta_min = 1e-6)
```

```toml
[defaults.stabilization]                           # experiments.toml
delta_update = "geometric"                         # "fixed" | "geometric"
delta        = 1.0                                 # δ₀: 1 with "all_blocks" pricing, 3 with "first_improving"
```

The rule needs the QPAS master. With `delta_update = "geometric"`, the experiment runner skips
the HiGHS QP masters with a message, and `solve_column_generation` rejects them.

The benchmark is `scripts/benchmark_adaptive_stabilization.jl`, the tests are
`test/test_adaptive.jl`. Notation follows the thesis: $y = [\lambda; s]$, $\bar A = [A\ K]$,
$r = b - \bar A y$ the residual, $\pi(y) = \hat\pi + r/\delta$ the dual estimate
(Def. 3.3.2), $Z$ the nullspace basis.

---

## 1. Why δ is free for this method

**Proposition 1.** Changing $\delta$ to $\delta'$ between two master solves leaves the
iterate feasible and leaves the ordering, the nullspace basis, $H_{\mathrm{eff}}$ and its
factorization unchanged. Only the dual estimate changes, to $\pi' = \hat\pi + r/\delta'$.

*Proof.* The constraints $\bar C y = \mathbf 1$, $y \ge 0$ do not contain $\delta$, and
$H_{\mathrm{eff}} = (A_{\mathrm{eff}}Z_x)^\top(A_{\mathrm{eff}}Z_x)$ does not either (thesis
Rem. 5.2.8). $\delta$ enters only the right-hand side $-\delta Z^\top g$ of the reduced system
and the dual estimate, which the solver recomputes at the start of every solve. $\square$

The Hessian of the eliminated master is $\frac1\delta\bar A^\top\bar A$, so the factor of the
QPAS is the factor of $\delta$ times the Hessian. A general QP solver (HiGHS) sees a new Hessian
after every change of $\delta$ and has to refactorize. For the QPAS, a new $\delta$ costs as
much as a new center: nothing (code: `set_delta!`, `src/qpas/hotstart_state.jl`).

## 2. What δ does: a step size of the proximal point method

Let $D(\pi)$ be the restricted dual function with $\mu$ maximized out,
$D(\pi) = b^\top\pi + \sum_k \min_{j \in \mathcal J'_k}(c_j - a_j^\top\pi)$ on
$\{\pi_I \ge 0\}$. It is concave and polyhedral. A master solve computes the proximal point

$$
\pi^+ = \arg\max_\pi\; D(\pi) - \tfrac{\delta}{2}\lVert\pi - \hat\pi\rVert^2 ,
$$

whose optimality condition is $\delta(\pi^+ - \hat\pi) \in \partial D(\pi^+)$. By
Prop. 3.3.5 the supergradient is the residual: $\pi^+ = \hat\pi + \frac1\delta r$ with
$r = b - \bar A y \in \partial D(\pi^+)$. **A master solve is an implicit supergradient step
of length $1/\delta$.** The center-only iterations of the proximal phase are the proximal
point method on a fixed restricted model, with step parameter $c_k = 1/\delta_k$.

The classical theory of the proximal point method with variable $c_k$:

- **Convergence** for any sequence with $\sum_k c_k = \infty$, in particular for any
  $\delta_k \le \delta_0$ (Rockafellar 1976; Güler 1991).
- **Rate.** Güler (1991) gives, for the maximization of a concave function,
  $$D^\star - D(\pi_k) \le \frac{\operatorname{dist}(\pi_0, \Pi^\star)^2}{2\sum_{i<k} c_i}.$$
  With fixed $\delta$ this is $O(\delta/k)$. With $\delta_k = \delta_0\gamma^k$, $\gamma < 1$, the
  sum grows geometrically and the bound decreases linearly. Rockafellar (1976) shows
  superlinear convergence when $c_k \to \infty$ and $\partial D^{-1}$ is Lipschitz at $0$.
- **Finite termination for polyhedral $D$** (Rockafellar 1976; Ferris 1991): the method
  reaches $\Pi^\star$ in finitely many steps for any $c_k$ bounded away from 0. The number of
  steps depends on $c_k$.
- **One step is enough for small δ.** For a linear program with an optimum and fixed
  $\hat\pi$, there is a $\bar\delta(\hat\pi) > 0$ such that for all $\delta \le \bar\delta$ the
  proximal point is the projection of $\hat\pi$ onto the optimal set $\Pi^\star$ (exact
  regularization: Mangasarian–Meyer 1979; Friedlander–Tseng 2007).

So on a fixed restricted model, a small enough $\delta$ ends the proximal phase in **one**
center-only iteration (if pricing then finds nothing, the next solve has $\pi = \hat\pi$ and
the run stops by Thm. 3.4.1).

The thesis shows why a fixed δ produces long tails instead. By Prop. 6.6.2, once the working
set has settled, every center-only iteration moves the dual by the same vector
$w = r/\delta$ while $y$ stays put. The restricted dual then increases by
$r^\top w = \lVert r\rVert^2/\delta$ per iteration. Climbing a fixed amount therefore takes a
number of iterations proportional to $\delta$.

The price of a small $\delta$ is the reason stabilization exists: with new columns, small
$\delta$ duals behave like LP duals and oscillate (thesis §9.4, Fig. 9.1). That suggests
**small δ while the model is fixed, large δ when it changes**.

## 3. The geometric rule

$$
\delta \leftarrow \begin{cases}
\max(\delta_{\min},\ \gamma\,\delta) & \text{after a center-only iteration,}\\
\min(\delta_0,\ \beta\,\delta) & \text{after an iteration that adds a column}
\end{cases}
\qquad (\gamma = \texttt{delta\_decrease},\ \beta = \texttt{delta\_increase}).
$$

The new δ is applied together with the usual center update $\hat\pi \leftarrow \pi$ and is
used by the next master solve (code: `_next_delta`, `src/cg/column_generation.jl`). The
defaults are $\gamma = 1/2$ and $\beta = 2$; $\beta = \infty$ resets δ to $\delta_0$ on every
column, which the measurements show to be slower and less accurate (§4.3, §4.5).

- **While the model is fixed**, consecutive center-only iterations have
  $c_k = 2^k/\delta_0$. By §2 the proximal phase on that model converges linearly instead of
  as $O(1/k)$, and after $O(\log_2(\delta_0/\bar\delta))$ halvings one step lands on the
  restricted optimal set. When $y$ stays put, the dual step doubles at every iteration.
- **When a column arrives**, δ grows back towards $\delta_0$, so the new column is released
  under (nearly) full stabilization, as in the thesis method. In the column phase, where
  almost every iteration adds a column, δ stays at $\delta_0$.
- **Termination stays exact.** The stopping test is unchanged (Thm. 3.4.1). With a smaller
  δ at the end, the final residual bound $\lVert b - \bar A y\rVert_\infty \le
  \delta\,\varepsilon_c\max(1, \lVert\pi\rVert_\infty)$ (Prop. 3.4.2) is tighter.
- **Cost:** none in the linear algebra (Proposition 1). `delta_min` only guards against
  amplifying the round-off in $r$ by $1/\delta$.

The rule is a simple form of the proximity control of bundle methods (Kiwiel 1990), which
adapts $t = 1/\delta$ to the observed progress. A center-only iteration is a "null step" of the
model that brings no new information, so a larger step is safe. A new column changes the
model, so the step shrinks again. Stabilized column generation has used similar schedules for
penalty and box parameters (du Merle et al. 1999; Ben Amor, Desrosiers, Frangioni 2009;
Pessoa et al. 2018 automate smoothing parameters). What is particular here is that
Proposition 1 makes every change of δ free.

## 4. Measurements

**Setup.** Instances I1–I4 of the thesis (Tab. 9.1), equality variant and 30 % inequality
rows, QPAS with the dense factor, one BLAS thread, best of 3 runs. The exact LP (HiGHS) is
the reference $z^\star$. Two pricing rules:

- **first-improving** (`CGOptions` default, thesis Alg. 3.1 and Ch. 9): pilots in random order,
  stop at the first column with negative reduced cost;
- **all-blocks** (default of `experiments.toml`): price every pilot and add one column per
  pilot with negative reduced cost.

Raw data: `results/adaptive_stabilization/<pricing>/summary.csv` and one trace per run (with δ
per iteration). Reproduce with

```sh
julia --project scripts/benchmark_adaptive_stabilization.jl                        # first-improving
HQ_PRICING=all_blocks julia --project scripts/benchmark_adaptive_stabilization.jl  # all-blocks
```

The variants:

| name | rule |
|---|---|
| `fixed` | thesis method, δ = 1 (first-improving: reproduces Tab. 9.3, I2 1.2 s, I3 2.9 s, I4 6.9 s) |
| `fixed_d0.1`, … `fixed_d10` | controls: a different but constant δ |
| `geometric_x2_d0.3`, `geometric_x2`, `_x2_d3`, `_x2_d10` | geometric rule (doubling), δ₀ = 0.3, 1, 3, 10 |
| `geometric`, `_d3`, `_d10` | geometric rule with reset (`delta_increase = Inf`), δ₀ = 1, 3, 10 |

All 192 runs (2 pricing rules × 8 instances × 12 variants) ended `optimal` and reached
$z^\star$; the remaining error is discussed in §4.5.

### 4.1 Iterations

CG iterations (in parentheses: iterations that add a column), **first-improving** pricing:

| instance | `fixed` | `geometric_x2_d0.3` | `geometric_x2` | `geometric_x2_d3` | `geometric_x2_d10` |
|---|---:|---:|---:|---:|---:|
| I1 | 3\,927 (129) | 146 (126) | 163 (134) | 193 (136) | 250 (155) |
| I1 (30 % ineq.) | 370 (87) | 97 (91) | 105 (91) | 124 (91) | 178 (116) |
| I2 | 11\,974 (1\,257) | 1\,233 (1\,179) | 1\,490 (1\,257) | 1\,543 (1\,201) | 1\,933 (1\,357) |
| I2 (30 % ineq.) | 1\,669 (711) | 907 (886) | 725 (710) | 752 (674) | 941 (736) |
| I3 | 8\,879 (1\,670) | 2\,052 (2\,034) | 1\,726 (1\,676) | 1\,475 (1\,402) | 1\,869 (1\,577) |
| I3 (30 % ineq.) | 5\,193 (1\,382) | 1\,773 (1\,760) | 1\,416 (1\,385) | 1\,163 (1\,119) | 1\,312 (1\,155) |
| I4 | 3\,252 (2\,417) | 2\,980 (2\,966) | 2\,436 (2\,416) | 1\,948 (1\,907) | 1\,818 (1\,695) |
| I4 (30 % ineq.) | 2\,960 (2\,089) | 2\,515 (2\,505) | 2\,104 (2\,088) | 1\,672 (1\,643) | 1\,385 (1\,331) |

**all-blocks** pricing:

| instance | `fixed` | `geometric_x2_d0.3` | `geometric_x2` | `geometric_x2_d3` | `geometric_x2_d10` |
|---|---:|---:|---:|---:|---:|
| I1 | 4\,043 (81) | 76 (55) | 108 (71) | 142 (87) | 206 (116) |
| I1 (30 % ineq.) | 388 (48) | 46 (38) | 62 (42) | 118 (76) | 150 (83) |
| I2 | 12\,123 (656) | 605 (433) | 812 (510) | 993 (608) | 1\,213 (729) |
| I2 (30 % ineq.) | 1\,976 (239) | 200 (176) | 272 (198) | 401 (265) | 623 (401) |
| I3 | 5\,885 (399) | 602 (512) | 448 (343) | 833 (600) | 1\,249 (829) |
| I3 (30 % ineq.) | 2\,090 (201) | 401 (348) | 230 (191) | 511 (378) | 838 (572) |
| I4 | 1\,640 (304) | 437 (398) | 337 (267) | 603 (458) | 1\,090 (760) |
| I4 (30 % ineq.) | 1\,721 (246) | 306 (286) | 279 (226) | 306 (227) | 609 (436) |

The geometric rule removes the proximal tail almost completely under both pricing rules.
Center-only iterations drop from 10\,717 to 233 on I2 and from 7\,209 to 50 on I3
(first-improving, `geometric_x2`), and from 11\,467 to 302 and from 5\,486 to 105 (all-blocks).
What remains is the column phase, whose length is set by δ₀.

### 4.2 Time

Total time of the column generation run in seconds (in parentheses: speedup over `fixed` with
the same pricing rule). **first-improving** pricing:

| instance | `fixed` | `geometric_x2_d0.3` | `geometric_x2` | `geometric_x2_d3` | `geometric_x2_d10` |
|---|---:|---:|---:|---:|---:|
| I1 | 0.020 | 0.002 (8.9×) | 0.003 (7.6×) | 0.003 (6.6×) | 0.004 (5.2×) |
| I1 (30 % ineq.) | 0.003 | 0.002 (1.8×) | 0.002 (1.6×) | 0.002 (1.9×) | 0.002 (1.4×) |
| I2 | 1.199 | **0.669 (1.8×)** | 0.718 (1.7×) | 0.694 (1.7×) | 0.815 (1.5×) |
| I2 (30 % ineq.) | 0.360 | 0.462 (0.8×) | 0.318 (1.1×) | **0.280 (1.3×)** | 0.332 (1.1×) |
| I3 | 2.917 | 3.246 (0.9×) | 2.194 (1.3×) | **1.578 (1.8×)** | 1.999 (1.5×) |
| I3 (30 % ineq.) | 2.123 | 2.762 (0.8×) | 1.754 (1.2×) | **1.170 (1.8×)** | 1.278 (1.7×) |
| I4 | 6.885 | 10.135 (0.7×) | 6.852 (1.0×) | 4.125 (1.7×) | **3.374 (2.0×)** |
| I4 (30 % ineq.) | 6.265 | 7.933 (0.8×) | 6.084 (1.0×) | 3.675 (1.7×) | **2.505 (2.5×)** |
| sum | 19.77 | 25.21 | 17.93 | 11.53 | **10.31** |

**all-blocks** pricing:

| instance | `fixed` | `geometric_x2_d0.3` | `geometric_x2` | `geometric_x2_d3` | `geometric_x2_d10` |
|---|---:|---:|---:|---:|---:|
| I1 | 0.022 | 0.002 (11.4×) | 0.002 (9.4×) | 0.003 (8.0×) | 0.003 (7.4×) |
| I1 (30 % ineq.) | 0.003 | 0.001 (2.2×) | 0.001 (2.2×) | 0.002 (1.4×) | 0.002 (1.4×) |
| I2 | 1.111 | **0.421 (2.6×)** | 0.439 (2.5×) | 0.478 (2.3×) | 0.560 (2.0×) |
| I2 (30 % ineq.) | 0.219 | 0.168 (1.3×) | **0.138 (1.6×)** | 0.154 (1.4×) | 0.222 (1.0×) |
| I3 | 1.361 | 1.639 (0.8×) | **0.720 (1.9×)** | 1.295 (1.1×) | 1.880 (0.7×) |
| I3 (30 % ineq.) | 0.537 | 1.012 (0.5×) | **0.365 (1.5×)** | 0.668 (0.8×) | 1.062 (0.5×) |
| I4 | 1.156 | 2.262 (0.5×) | **0.994 (1.2×)** | 1.395 (0.8×) | 2.464 (0.5×) |
| I4 (30 % ineq.) | 1.154 | 1.809 (0.6×) | 0.926 (1.2×) | **0.776 (1.5×)** | 1.237 (0.9×) |
| sum | 5.56 | 7.32 | **3.59** | 4.77 | 7.43 |

**The best δ₀ depends on the pricing rule:** 3–10 with first-improving pricing, 1 with
all-blocks pricing. A plausible explanation: with first-improving pricing one column enters
per iteration, and a strong stabilization pays off by making each of those columns count;
with all-blocks pricing up to one column per pilot enters, the model improves much faster per
iteration, and less stabilization is needed. In both cases δ₀ that is
too small (0.3) produces LP-like duals and many useless columns on the larger instances.

**All-blocks pricing with `geometric_x2` is the fastest combination overall**: 3.59 s for all
eight instances, against 19.77 s for the thesis setting (first-improving, fixed δ = 1) and
5.56 s for all-blocks with fixed δ. On I4 it takes 0.99 s, against 6.89 s for the thesis
setting and 23.1 s for the exact LP.

With all-blocks pricing the speedup on I4 is small (1.2×), because the tail was already short:
most of the time is in the column phase, which the rule does not change.

### 4.3 Reset or double

The same rule with reset (`delta_increase = Inf`) instead of doubling, time in seconds:

| instance | first-imp. `geometric` | first-imp. `geometric_x2` | all-blocks `geometric` | all-blocks `geometric_x2` |
|---|---:|---:|---:|---:|
| I2 | 0.783 | 0.718 | 0.594 | 0.439 |
| I3 | 2.262 | 2.194 | 0.877 | 0.720 |
| I4 | 6.634 | 6.852 | 1.055 | 0.994 |
| sum over all 8 | 17.76 | 17.93 | 4.07 | 3.59 |

With δ₀ = 1, a reset and doubling are about equal under first-improving pricing, and doubling
is up to 1.4× faster under all-blocks pricing. With a larger δ₀ the difference grows: with
δ₀ = 10 and first-improving pricing, a reset is 1.7× slower on I2 (1.42 s vs 0.82 s). In the
proximal phase, columns arrive one at a time between center-only iterations, and a reset
throws away the progress of the halvings before it. Doubling is therefore the default.

### 4.4 Why the geometric rule beats every constant δ

Controls with a constant δ (time in s / CG iterations), first-improving pricing:

| instance | `fixed` (δ = 1) | δ = 0.1 | δ = 0.01 | δ = 3 | δ = 10 |
|---|---:|---:|---:|---:|---:|
| I1 | 0.020 / 3\,927 | 0.003 / 200 | 0.004 / 157 | 0.061 / 12\,114 | 0.209 / 40\,479 |
| I1 (30 % ineq.) | 0.003 / 370 | 0.002 / 103 | 0.003 / 116 | 0.006 / 1\,154 | 0.020 / 3\,864 |
| I2 | 1.199 / 11\,974 | 0.868 / 2\,132 | 1.553 / 1\,845 | 1.966 / 26\,789 | 4.128 / 62\,978 |
| I2 (30 % ineq.) | 0.360 / 1\,669 | 0.658 / 1\,193 | 1.164 / 1\,442 | 0.450 / 4\,434 | 0.925 / 15\,356 |
| I3 | 2.917 / 8\,879 | 4.521 / 2\,914 | 8.236 / 3\,189 | 2.425 / 10\,810 | 3.268 / 18\,821 |
| I3 (30 % ineq.) | 2.123 / 5\,193 | 3.728 / 2\,633 | 6.226 / 2\,717 | 1.772 / 7\,909 | 1.830 / 9\,358 |
| I4 | 6.885 / 3\,252 | 14.416 / 3\,811 | 28.285 / 4\,948 | 4.463 / 4\,005 | 4.257 / 10\,156 |
| I4 (30 % ineq.) | 6.265 / 2\,960 | 11.334 / 3\,110 | 20.094 / 4\,083 | 4.157 / 5\,636 | 3.881 / 11\,667 |

all-blocks pricing:

| instance | `fixed` (δ = 1) | δ = 0.1 | δ = 0.01 | δ = 3 | δ = 10 |
|---|---:|---:|---:|---:|---:|
| I1 | 0.022 / 4\,043 | 0.002 / 143 | 0.004 / 80 | 0.061 / 12\,143 | 0.210 / 40\,478 |
| I1 (30 % ineq.) | 0.003 / 388 | 0.002 / 44 | 0.004 / 59 | 0.007 / 1\,159 | 0.019 / 3\,866 |
| I2 | 1.111 / 12\,123 | 0.667 / 2\,788 | 1.449 / 564 | 2.105 / 26\,844 | 4.592 / 63\,000 |
| I2 (30 % ineq.) | 0.219 / 1\,976 | 0.386 / 597 | 0.815 / 357 | 0.361 / 4\,621 | 0.995 / 15\,444 |
| I3 | 1.361 / 5\,885 | 3.834 / 1\,775 | 10.267 / 870 | 2.567 / 11\,494 | 4.087 / 19\,103 |
| I3 (30 % ineq.) | 0.537 / 2\,090 | 1.826 / 938 | 5.541 / 668 | 0.959 / 3\,349 | 1.963 / 9\,684 |
| I4 | 1.156 / 1\,640 | 5.380 / 790 | 21.236 / 897 | 1.885 / 3\,683 | 4.013 / 10\,701 |
| I4 (30 % ineq.) | 1.154 / 1\,721 | 3.528 / 489 | 12.736 / 669 | 1.345 / 3\,955 | 3.031 / 11\,442 |

δ acts in opposite directions in the two phases:

- **Column phase: a large enough δ is needed.** Small δ gives LP-like duals that generate many
  useless columns: on I4 (first-improving), δ = 0.01 needs 4\,928 column iterations instead
  of 2\,417 and is 4.1× slower. With first-improving pricing, even more stabilization helps:
  2\,417 (δ = 1) → 1\,903 (δ = 3) → 1\,656 (δ = 10) column iterations on I4.
- **Proximal phase: smaller δ is better** (§2). The tail grows with δ: about 63\,000 iterations
  on I2 with δ = 10, under both pricing rules.

A constant δ has to compromise between the two, and the best constant differs per instance
(0.1 to 10). The geometric rule removes the compromise: δ₀ is chosen for the column phase,
and the halvings remove the tail that it would otherwise cause. With the right δ₀ for the
pricing rule (`geometric_x2_d3` for first-improving, `geometric_x2` for all-blocks), the
geometric rule is at least as fast as the best constant δ on every instance (equal only on
I1, within the timer resolution).

### 4.5 Accuracy

Relative error of the operational cost $|c^\top\lambda - z^\star|/|z^\star|$ / final
infeasibility:

| instance | first-imp. `fixed` | first-imp. `geometric_x2_d3` | all-blocks `fixed` | all-blocks `geometric_x2` |
|---|---:|---:|---:|---:|
| I1 | 4e-15 / 3e-14 | 9e-16 / 3e-16 | 9e-15 / 4e-14 | 1e-16 / 2e-15 |
| I1 (30 % ineq.) | 5e-16 / 1e-14 | 1e-16 / 8e-16 | 1e-15 / 2e-14 | 4e-16 / 4e-16 |
| I2 | 2e-05 / 6e-05 | 5e-13 / 7e-16 | 3e-05 / 7e-05 | 5e-13 / 7e-16 |
| I2 (30 % ineq.) | 4e-07 / 7e-06 | 2e-15 / 1e-15 | 5e-07 / 8e-06 | 5e-15 / 2e-15 |
| I3 | 1e-08 / 1e-05 | 4e-15 / 9e-16 | 6e-07 / 1e-05 | 5e-15 / 7e-16 |
| I3 (30 % ineq.) | 2e-08 / 6e-06 | 8e-15 / 7e-16 | 7e-09 / 6e-06 | 7e-15 / 7e-16 |
| I4 | 7e-08 / 7e-06 | 1e-14 / 1e-15 | 4e-08 / 6e-06 | 1e-14 / 7e-16 |
| I4 (30 % ineq.) | 6e-10 / 4e-06 | 9e-15 / 7e-16 | 8e-15 / 3e-13 | 8e-15 / 1e-15 |

The geometric rule also makes the final solution **exact to machine precision**: the cost
error drops from up to $3\cdot10^{-5}$ (I2) to $10^{-13}$–$10^{-15}$, and the infeasibility
from $10^{-5}$ to $10^{-15}$. This is Prop. 3.4.2: the residual at termination is bounded by
$\delta\,\varepsilon_c\max(1,\lVert\pi\rVert_\infty)$ with the final δ, which is far smaller
than δ₀ when the run ends in the proximal phase. The Lagrangian bound gap of the doubling
variants is at most $5\cdot10^{-13}$ on all instances, against up to $2\cdot10^{-7}$ for
`fixed`.

With a reset, the run can end shortly after a column has brought δ back to δ₀. It then stops
at that larger δ, still within the tolerance of the stopping test, but less accurate:
`geometric_d3` on I2 has a cost error of $10^{-5}$ (first-improving), and `geometric` on I2 one
of $6\cdot10^{-6}$ (all-blocks). Doubling avoids this, since δ only grows step by step.

## 5. Recommendation

- **Toggle:** `delta_update = "geometric"` in `[defaults.stabilization]` or per experiment
  (`"fixed"` is the thesis method and stays the default, so all thesis numbers remain
  reproducible). Doubling (`delta_increase = 2`) is the default of the geometric rule.
- **Base value δ₀ = `delta`:** 1 with all-blocks pricing, 3 with first-improving pricing (10 on
  the largest instance I4). δ₀ now only tunes the column phase.
- **Fastest combination:** all-blocks pricing with the geometric rule and δ₀ = 1, 1.2–2.5×
  faster than all-blocks with fixed δ on I2–I4, and 5.5× faster in total than the thesis
  setting on I1–I4. The final solution is exact to machine precision.

**Open directions.** The remaining time is the column phase. Two directions follow from §2:
(i) proximity control based on the Lagrangian bound (Kiwiel 1990), which adapts δ inside
the column phase but needs the bound, i.e. all pilots priced; (ii) choosing δ₀ from the
instance, e.g. from the size of the duals after the first iterations.

## References

- R. T. Rockafellar, *Monotone operators and the proximal point algorithm*, SIAM J. Control
  Optim. 14 (1976) 877–898.
- O. Güler, *On the convergence of the proximal point algorithm for convex minimization*,
  SIAM J. Control Optim. 29 (1991) 403–419.
- M. C. Ferris, *Finite termination of the proximal point algorithm*, Math. Programming 50
  (1991) 359–366.
- O. L. Mangasarian, R. R. Meyer, *Nonlinear perturbation of linear programs*, SIAM J.
  Control Optim. 17 (1979) 745–752.
- M. P. Friedlander, P. Tseng, *Exact regularization of convex programs*, SIAM J. Optim. 18
  (2007) 1326–1350.
- K. C. Kiwiel, *Proximity control in bundle methods for convex nondifferentiable
  minimization*, Math. Programming 46 (1990) 105–122.
- O. du Merle, D. Villeneuve, J. Desrosiers, P. Hansen, *Stabilized column generation*,
  Discrete Math. 194 (1999) 229–237.
- H. Ben Amor, J. Desrosiers, A. Frangioni, *On the choice of explicit stabilizing terms in
  column generation*, Discrete Appl. Math. 157 (2009) 1167–1184.
- A. Pessoa, R. Sadykov, E. Uchoa, F. Vanderbeck, *Automation and combination of
  linear-programming based stabilization techniques in column generation*, INFORMS J.
  Comput. 30 (2018) 339–360.
