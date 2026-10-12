# HotstartedQPAS

Column generation for the multi-pilot crew scheduling LP, with quadratic proximal
stabilization and a **hot-started nullspace active-set QP solver (QPAS)** whose reduced-Hessian
LDLᵀ factorization is updated in place, either with CHOLMOD (the thesis method) or with a dense
factor of the active block (`factorization = "dense"`, ~2.5x faster master solves, see
[`docs/dense_ldlt_backend.md`](docs/dense_ldlt_backend.md)). This is the code for the master's
thesis *"Hot-Started Active-Set Method for Regularized Column Generation"* (K. Bäumer,
[`docs/Masterthesis_Kyrian_Baumer.pdf`](docs/Masterthesis_Kyrian_Baumer.pdf)).

## Quick start

```bash
# Julia ≥ 1.10 (developed with 1.13 via juliaup: https://julialang.org/install)
julia --project -e 'using Pkg; Pkg.instantiate()'     # once
julia --project run.jl                                 # run every experiment in experiments.toml
julia --project run.jl --only I1,I2                    # run a selection
julia --project run.jl --list                          # show experiments and their solvers
julia --project run.jl other.toml                      # use another config file
julia --project -e 'using Pkg; Pkg.test()'             # tests
```

### Experiments are defined in `experiments.toml`

All parameters are listed and commented once under `[defaults.*]`. Each `[[experiment]]` only
lists what differs:

```toml
[[experiment]]
name = "I3"
[experiment.instance]
n_nodes = 175
edge_probability = 0.091
n_tasks = 390
n_pilots = 10
[experiment.solvers]        # turn solvers off per experiment
explicit_qp   = false
eliminated_qp = false
```

| section | contents |
|---|---|
| `[solvers]` | `exact`, `lp`, `explicit_qp`, `eliminated_qp`, `qpas` (true/false) |
| `[instance]` | instance generator: nodes, edge probability, tasks, pilots, inequality fraction, seed, … |
| `[column_generation]` | iteration/column/time limits, pricing rule, tolerances, verbosity |
| `[stabilization]` | `delta`, `center_update` |
| `[qpas]` | QPAS tolerances, factorization backend (`cholmod` / `dense`), refactorization interval, sparse solves, debug |
| `[lp]` | `big_m` |
| `[output]` | output directory, per-iteration traces, saving the instance |

Unknown keys are rejected with the list of allowed names, so a typo cannot silently fall back
to a default. Results are written to `results/<experiment>/summary.txt` (comparison table and
details), `results/<experiment>/<solver>.csv` (per-iteration traces) and `results/summary.csv`
(one row per experiment and solver, rewritten after each experiment).

From Julia (e.g. the VS Code REPL) the same API is available directly:

```julia
using HotstartedQPAS
inst  = generate_instance(InstanceConfig(n_nodes = 150, edge_probability = 0.09, n_tasks = 230, n_pilots = 7))
exact = solve_exact(inst)
qpas  = solve_column_generation(inst, QPASMaster();
                                options = CGOptions(stabilization = StabilizationOptions(delta = 1.0)))
compare_results(exact, qpas; reference = exact)
print_details(qpas)
```

`scripts/quickstart.jl` is a longer tour of this API.

## The solvers

| # | solver | restricted master problem | thesis |
|---|--------|---------------------------|--------|
| 1 | `solve_exact` | — : the edge-based LP with all coupling and flow rows stacked in one matrix, HiGHS | eq. 5.2 |
| 2 | `HighsLPMaster` | LP master with Big-M slacks, persistent HiGHS instance (`addCols`, warm-started simplex) | eq. 1.23, Alg. 5.1 |
| 3a | `HighsExplicitQPMaster` | stabilized QP that keeps `w`: Hessian `δI` on `w` only, HiGHS QP | eq. 4.3 |
| 3b | `HighsEliminatedQPMaster` | stabilized QP without `w`: Hessian `(AX)ᵀ(AX)/δ`, HiGHS QP | eq. 4.5/4.6 |
| 4 | `QPASMaster` | the same QP as 3b, solved by the hot-started QPAS of Chapter 4 | Alg. 4.3, §4.5 |

All column generation variants share one driver (`solve_column_generation`), one pricing
oracle (DAG shortest path), one column pool, and one statistics handler (`CGStats`).
Each run returns a `CGResult` with the final weights, duals, Lagrangian bound, per-pilot
edge flows (`edge_flows`), active schedules (`active_columns`) and full per-iteration traces.

**Termination is exact.** CG stops when a complete pricing round over all pilots finds no
column with reduced cost below `-reduced_cost_tolerance`. Stabilized runs also require the
stability center to have converged (`π = π̂`, equivalently `AXλ = b`). There is no
"k iterations without improvement" heuristic.

## Results

Instance I3 (sizes as in thesis Table 5.1), δ = 1, Apple M-series, Julia 1.13, single thread,
600 s time limit per method. "gap" is `objective − exact optimum`.

```
I3: |V|=175, |E|=1546, pilots=10, tasks=360
method                       status        iter   time [s]  master [ms]  inner it.       objective        gap   infeas
Exact LP (HiGHS simplex)     optimal      21096      3.821          NaN      21096     1860.368950   0.00e+00  0.0e+00
LP CG (HiGHS)                optimal       4771    504.198      105.616    1109096     1860.368950   1.35e-10  1.2e-12
Explicit QP CG (HiGHS)       optimal       3047    522.231      171.309    5089921     1860.368537  -4.13e-04  5.7e-06
Eliminated QP CG (HiGHS)     time_limit    2569    600.565      233.517    3688735     1860.367287  -1.66e-03  3.7e-05
Hot-start QPAS CG            optimal       3020      5.439        1.762     119327     1860.368427  -5.23e-04  1.1e-05
```

The QPAS is ~93x faster than LP column generation in total time (~60x per master solve). The
stabilized gaps (relative ~3e-7) come from stopping at `center_tolerance`: the final iterate
satisfies `AXλ = b` only to ~1e-5, and its master objective equals the exact optimum to all
printed digits. Lower `center_tolerance` for tighter gaps. On I2 the QPAS takes ~1.0 s
versus ~46 s for LP CG.

## Project layout

```
src/
├── HotstartedQPAS.jl            module definition, includes, exports
├── backends/                    the only code that calls into C libraries
│   ├── ldlt_interface.jl        UpdatableLDLt: the factor interface both backends implement
│   ├── dense/DenseLDLt.jl       dense LDLᵀ of the active block (append / compacting delete / C1)
│   ├── cholmod/                 CHOLMOD (SuiteSparse) layer
│   │   ├── Cholmod.jl           submodule
│   │   ├── types.jl             Julia mirrors of cholmod_dense / _sparse / _factor (+ layout check)
│   │   ├── libcholmod.jl        raw ccall bindings — the ONLY ccalls into CHOLMOD
│   │   ├── common.jl            owned cholmod_common workspace, error type
│   │   └── ldlt_factor.jl       LDLtFactor: identity-initialised LDLᵀ with rowadd!/rowdel!/solve!
│   └── highs/HighsBackend.jl    HighsModel: persistent HiGHS instance (LP/QP, addCols, duals)
├── utils/                       IndexSet (O(1) sets for W and N), SectionTimer
├── problem/                     FlowNetwork, CrewSchedulingInstance, generator, direct formulation, I/O
├── columns/column_pool.jl       append-only CSC column store + sparse kernels
├── pricing/shortest_path.jl     DAG shortest-path pricing oracle
├── exact/exact_lp.jl            solver 1
├── master/                      AbstractMasterSolver interface + the four master solvers
├── qpas/                        the hot-started active-set method
│   ├── options.jl               QPASOptions, QPASCounters
│   ├── ordering.jl              ColumnOrdering: the partition [S, W, N]
│   ├── nullspace.jl             implicit nullspace basis Z (explicit forms for debugging)
│   ├── kkt_system.jl            KKTSystem: ordering + LDLᵀ factor of the reduced Hessian
│   ├── hotstart_state.jl        QPASState: everything carried between CG iterations
│   └── solver.jl                Alg. 4.3: direction, ratio test, expand / reduce, singular steps
├── cg/                          CGOptions, CGResult, the column generation driver
├── experiments/                 experiment config files and runner
└── stats/                       CGStats / ExactStats, comparison tables, CSV traces
test/                            unit + integration tests (CHOLMOD, dense factor, problem, QPAS on both backends, end-to-end)
src/experiments/                 TOML config loading (config.jl) and the experiment runner (runner.jl)
scripts/quickstart.jl            API tour
scripts/benchmark_factorization.jl  CHOLMOD vs dense factor: CG runs, diagnostics, micro benchmark
experiments.toml                 the experiment definitions (all parameters documented)
run.jl                           entry point: julia --project run.jl [config] [--only …]
claude_context/                  context for Claude: original sketches and the earlier thesis (not used by the code)
docs/                            the thesis and the notes on the QPAS, inequality rows and the dense factor
```

## Architecture

```
CrewSchedulingInstance ──► solve_exact ──► ExactResult
          │
          ▼
solve_column_generation(instance, master; options)
   ├── ColumnPool            columns aⱼ = A_task xʲ, costs, blocks, paths   (capacity n_max)
   ├── ShortestPathPricer    min (cₖ − A_taskᵀπ)ᵀx over source–sink paths
   ├── AbstractMasterSolver  initialize! / add_columns! / set_center! / solve! → MasterSolution
   │     ├── HighsLPMaster, HighsExplicitQPMaster, HighsEliminatedQPMaster  (HighsModel)
   │     └── QPASMaster ──► QPASState
   │                          ├── x, π, μ, cost scale, counters, timers
   │                          └── KKTSystem
   │                                ├── ColumnOrdering [S, W, N]  (⇒ Z and AZ implicitly)
   │                                ├── UpdatableLDLt: Cholmod.LDLtFactor (n_max × n_max, sparse)
   │                                │   or DenseLDLt.DenseLDLtFactor (|N| × |N|, dense)
   │                                └── scratch buffers, optional debug copy of H
   └── CGStats               per-iteration traces + solver details ──► CGResult
```

The master interface (`src/master/interface.jl`) is the only contract between the driver and
a master solver, so a new master (another QP solver, …) plugs
in without touching the driver.

### Thesis ↔ code

| thesis | code |
|--------|------|
| multi-block DW master, convexity matrix `C` (§1.4) | `ColumnPool.block`, convexity rows in the HiGHS masters |
| regularized master, explicit / eliminated (eq. 4.3 / 4.5–4.6) | `HighsExplicitQPMaster` / `HighsEliminatedQPMaster`, `QPASMaster` |
| `π = π̂ + (b − AXλ)/δ` (eq. 4.4) | `QPASState.π`, `refresh_duals!`, `take_step!` |
| reordering theorem `[S, W, N]` (Thm 4.2.3) | `ColumnOrdering` (`rep`, `W`, `N`, `set_of`) |
| nullspace basis `zⱼ = eⱼ − e_rep(j)` (Cor. 4.3.4) | implicit; `nullspace_basis` builds it for debugging |
| reduced system `Hy = −δZᵀg` (eq. 4.32) | `compute_direction!` |
| multipliers `μ = h_S`, `λ = h_W − C_Wᵀμ` (Thm 4.3.5) | `update_convexity_duals!`, `most_negative_multiplier` |
| expanding the working set (§4.4.1, Thm 4.4.2/4.4.4) | `expand_working_set!`, `_swap_representative!` |
| reducing the working set (§4.4.2, Thm 4.4.5) | `reduce_working_set!`, `hessian_column!`, `insert_column!` |
| CHOLMOD `rowadd` / `rowdel` (§4.4.3) | `Cholmod.rowadd!` / `Cholmod.rowdel!` |
| singular reduced Hessian (§4.4.5) | `reduce_working_set!` + `_singular_step!` |
| fixed size `n_max` (§4.5.1) | `CGOptions.max_columns` → pool / factor capacity |
| hot start (§4.5.2) | `register_columns!` (new column ⇒ W, factor unchanged) |

## Implementation notes for the QPAS

These go beyond the thesis text. Each was measured or tested. Derivations and proofs are in
[`docs/qpas_mathematical_notes.md`](docs/qpas_mathematical_notes.md).

1. **The gradient is never stored.** With `π = π̂ + (b − Ax)/δ` the gradient of the eliminated QP is
   `g = c − Aᵀπ`. So `gⱼ` is the reduced cost of column `j` without the convexity dual.
   The solver maintains the m-vector `π` (updated by `π −= (α/δ)·Ap` after each step) and
   evaluates `gⱼ` only where needed: on `N ∪ S` for the right-hand side, and on `W` only
   at a subspace minimizer. A primal step costs `O((|N| + p)·nnz)`, independent of the
   number of columns.
2. **Z and AZ are implicit.** Column `j` of the reduced Hessian is assembled on demand,
   `H[k,j] = (a_k − a_rep(k))ᵀ(a_j − a_rep(j))`, using a dense scatter of `u = a_j − a_rep(j)`.
   No transposed structures have to be kept in sync.
3. **After a full step the next direction is known to be zero**, so the multipliers are
   checked directly without a solve.
4. **Singular reduced Hessian.** A dependent new column is detected from the LDLᵀ pivots
   right after the `rowadd`, which saves one solve per reduction compared with checking
   the Schur complement first. The rare dependent case is undone. Then, instead of moving
   `argmax|tₖ|` into W (thesis §4.4.5, which can leave a positive variable in W), the solver
   takes the zero-curvature descent step `d = z_r − Σ tₖzₖ` (`Ad = 0`, `gᵀd = λ_r < 0`) up to
   the first blocking variable. That variable joins W, the dependency disappears, and
   `x_W = 0` holds throughout. Tested by `test/test_qpas.jl`, including the case where the
   representative blocks.
5. **Periodic refactorization.** `cholmod_rowdel` zeroes a row numerically but keeps its
   entries in the pattern of `L`. After ~28k updates on I2, `L` stored 438k entries instead
   of 50k, and every solve paid for them. Rebuilding the factor every 150 updates (in
   increasing index order, so every `rowadd` appends a last row) made the QPAS ~8x faster.
   `QPASOptions(refactor_interval = …)` controls this.
6. **Sparse right-hand-side solves.** `cholmod_solve2` with a `Bset` pattern (= N) solves only
   on the reach of N. The arithmetic does not depend on `n_max`, but CHOLMOD allocates an
   `n_max`-length workspace on every such call (160 kB at `n_max` = 20 000; 14 GB allocated
   over an I3 run).
7. **Replacement representative**: when a representative blocks, the column of its block in
   N with the largest weight replaces it.
8. **Debug mode** (`QPASOptions(debug = true)`): keeps an explicit copy of H and checks
   after every iteration that it equals `(AZ)ᵀ(AZ)` rebuilt from scratch and that `LDLᵀ`
   reproduces it.
9. **Dense factor backend** (`QPASOptions(factorization = :dense)`). While `H_eff` is positive
   definite, `|N| ≤ m`, and on these instances `H_eff` is ~100 % dense. The dense backend
   stores only the active `|N| × |N|` block in activation order: a new index is appended (one
   triangular solve), a deleted slot is removed by a rank-one update of the trailing block
   fused with compaction, and slack changes use method C1. It never holds structural zeros, so
   it needs no periodic refactorization. On I1–I3 the master solve is 2.2–2.9x faster with
   identical iteration counts and accuracy. Derivations and measurements:
   [`docs/dense_ldlt_backend.md`](docs/dense_ldlt_backend.md).

### On iteration counts of the stabilized methods

With exact termination, the stabilized runs add far fewer *columns* than LP CG (I2: ~920 vs
~2400). However, they continue with "center-only" iterations after pricing stops finding
columns, until `π = π̂`, which is a proximal-point tail on the restricted model. The thesis
runs stopped after five non-improving iterations, which cuts this tail at the price of a
larger gap (thesis Table 5.5). For the QPAS these iterations are cheap (hot start, no new
column). `CGOptions.center_tolerance` controls how far the tail is followed.

## Statistics

`CGStats` records per iteration: master time, pricing time, master objective, operational cost
`cᵀλ`, coupling infeasibility, Lagrangian bound (when all pilots were priced), dual change
`‖πₖ − πₖ₋₁‖∞`, number of columns, inner iterations (simplex / QP / active-set), columns
added, and center moves. Solver-specific totals are in `stats.details`: for the QPAS these
are iteration, step, expansion, reduction, singular-step, factor-update and refactorization
counts, a time breakdown, the time and call count of every factor operation (assemble, rowadd,
rowdel, updown, solve) and the factor memory. Use `compare_results`, `print_details`,
`write_trace_csv`. The diagnostics traces (`diagnostics_every`) sample memory, factor contents
and, with `diagnostics_residual = true`, the factor residual `‖LDLᵀ − H_eff‖`.

## Inequality coupling rows

Every coupling row has bounds `task_lower ≤ Σₖ A_task xₖ ≤ task_upper`. Equalities have equal
bounds; `≥ 1` rows are made with `generate_instance(...; inequality_fraction = 0.3)`.

| component | inequality rows |
|-----------|-----------------|
| exact LP, LP CG, explicit QP CG | supported (row bounds; the duals get their sign automatically) |
| QPAS | `=`, `≥` and `≤` rows (one finite bound); range rows raise `ArgumentError` |
| pricing, Lagrangian bound, infeasibility measure | supported (duals are projected onto their sign conditions before pricing) |
| eliminated QP (HiGHS) | raises `ArgumentError` (`supports_inequalities(master) == false`) |

The QPAS gives every inequality row a slack, `aᵢᵀλ − sᵢ = bᵢ` for `≥` rows and
`aᵢᵀλ + sᵢ = bᵢ` for `≤` rows (a `≤` row is a `≥` row with both sides negated; flipping the
slack sign instead keeps the columns unchanged). The slacks belong to no convexity block and
are split into a working set `W_s` (`sᵢ = 0`, the row acts as an equality) and a free set
`N_s`. The factorized matrix is the **effective reduced Hessian**
`H_eff = (A_eff Z)ᵀ(A_eff Z)`, where `A_eff` keeps the equality rows and the inequality rows
whose slack is in `W_s`, so the factor stays `n_max × n_max`. A slack entering `W_s` is a
rank-one update `H_eff + vvᵀ`, a slack leaving it a downdate `H_eff − vvᵀ`
(`cholmod_l_updown`), with `v` the row of `AZ` of that inequality. The multiplier of a slack in
`W_s` is the dual of its row (signed), so the optimality test also checks the dual sign
conditions. Theory: [`docs/qpas_inequality_extension.md`](docs/qpas_inequality_extension.md).

## Reproducing the benchmarks

`julia --project run.jl --only I3`. The shipped config disables the two HiGHS QP masters for
I3; set them to `true` in `experiments.toml` to include them (~10 min each).

CHOLMOD vs dense factor: `julia --project scripts/benchmark_factorization.jl` (`--suite
small|large|micro|report`, `--reps`, `--only`). Writes `results/factorization/report.md`.
