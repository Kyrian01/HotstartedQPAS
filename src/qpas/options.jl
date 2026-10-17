"""
    QPASOptions(; kwargs...)

Settings of the hot-started active-set solver.

| field                 | default  | meaning                                                                     |
|:----------------------|:---------|:----------------------------------------------------------------------------|
| `max_iterations`      | 100_000  | active-set iterations per master solve                                      |
| `step_tolerance`      | 1e-10    | `‖p‖∞` below which the search direction is treated as zero                  |
| `dual_tolerance`      | 1e-10    | multipliers `λⱼ ≥ -dual_tolerance ⋅ max(1, maxⱼ|cⱼ|)` are accepted          |
| `ratio_tolerance`     | 1e-14    | components `pⱼ ≥ -ratio_tolerance` are ignored in the ratio test            |
| `singular_tolerance`  | 1e-10    | scaled pivot `D[k]/‖a_k - a_rep(k)‖²` (unmasked norm) below which the reduced Hessian is singular |
| `check_pivots`        | true     | also inspect the pivots after expansions (rebuild the factor if one decays) |
| `factorization`       | :cholmod | LDLᵀ backend of the reduced Hessian: `:cholmod` or `:dense`, see below      |
| `refactor_interval`   | 150      | CHOLMOD: rebuild the factor after this many updates (0 = never), see below  |
| `dense_refactor_interval` | 0    | dense: rebuild after this many updates (0 = never; only numerical refresh)  |
| `sparse_solves`       | true     | solve only on the pattern `N` instead of all `n_max` rows                   |
| `debug`               | false    | keep the explicit reduced Hessian and verify every update (slow, tests only)|

Backends (`src/backends/ldlt_interface.jl`, `docs/dense_ldlt_backend.md`):

- `:cholmod`: sparse simplicial LDLᵀ of fixed size `n_max × n_max` (identity outside `N`),
  updated with `cholmod_l_rowadd` / `rowdel` / `updown` (thesis Ch. 6);
- `:dense`: dense LDLᵀ of the active block only (`|N| × |N|`, `|N| ≤ m+1`), new indices are
  appended, deletions compact the factor (thesis Ch. 8).

Why periodic refactorization (CHOLMOD): `cholmod_rowdel` zeroes a row/column numerically but
leaves its entries in the sparsity pattern of `L`. Over thousands of working-set updates these
structural zeros accumulate (e.g. 438k stored entries instead of 50k on instance I2), and
every solve and row update pays for them. Rebuilding the factor from the current set `N`
(`refactor!`) removes them; with a 150-update interval this is ~8x faster on I2. The dense
factor has no such structural garbage, so its interval is separate and off by default; pivots
that decay below `singular_tolerance` trigger a rebuild for both backends (`check_pivots`).
"""
Base.@kwdef struct QPASOptions
    max_iterations::Int = 100_000
    step_tolerance::Float64 = 1e-10
    dual_tolerance::Float64 = 1e-10
    ratio_tolerance::Float64 = 1e-14
    singular_tolerance::Float64 = 1e-10
    check_pivots::Bool = true
    factorization::Symbol = :cholmod
    refactor_interval::Int = 150
    dense_refactor_interval::Int = 0
    sparse_solves::Bool = true
    debug::Bool = false
end

"LDLᵀ backends accepted by `QPASOptions(factorization = …)`."
const FACTORIZATIONS = (:cholmod, :dense)

function validate(o::QPASOptions)
    o.factorization in FACTORIZATIONS ||
        throw(ArgumentError("qpas factorization must be one of $(join(FACTORIZATIONS, ", ")), " *
                            "got :$(o.factorization)"))
    o.refactor_interval >= 0 || throw(ArgumentError("refactor_interval must be ≥ 0"))
    o.dense_refactor_interval >= 0 || throw(ArgumentError("dense_refactor_interval must be ≥ 0"))
    return o
end

"Refactorization interval that applies to the configured backend (0 = never)."
refactor_interval(o::QPASOptions) =
    o.factorization === :dense ? o.dense_refactor_interval : o.refactor_interval

"""
    QPASCounters

Event counters of the active-set solver (per solve and cumulative).

- `iterations`: active-set iterations (direction computations + multiplier checks)
- `steps`, `full_steps`: primal steps taken, and those with `α = 1`
- `expansions`: blocking constraints added to W (thesis §6.2, Expanding the working set)
- `representative_swaps`: expansions where the blocking index was a representative
- `reductions`: constraints released from W (thesis §6.3, Reducing the working set)
- `slack_expansions`, `slack_reductions`: of the expansions/reductions, those of an
  inequality-row slack (rank-one update/downdate of `H_eff`)
- `singular_steps`: zero-curvature steps taken because the reduced Hessian would become
  singular (thesis Ch. 7: Prop. 7.1.1, Singular reductions; Thm. 7.3.1 / 7.3.2,
  zero-curvature directions for a released column / surplus)
- `factor_updates`: factor row additions + deletions + rank-one updates
- `refactorizations`: rebuilds of the factor from scratch
"""
Base.@kwdef mutable struct QPASCounters
    solves::Int = 0
    iterations::Int = 0
    steps::Int = 0
    full_steps::Int = 0
    expansions::Int = 0
    representative_swaps::Int = 0
    reductions::Int = 0
    slack_expansions::Int = 0
    slack_reductions::Int = 0
    singular_steps::Int = 0
    factor_updates::Int = 0
    refactorizations::Int = 0
end

function Base.empty!(c::QPASCounters)
    for f in fieldnames(QPASCounters)
        setfield!(c, f, 0)
    end
    return c
end

function accumulate!(dst::QPASCounters, src::QPASCounters)
    for f in fieldnames(QPASCounters)
        setfield!(dst, f, getfield(dst, f) + getfield(src, f))
    end
    return dst
end

"Names of the sections timed inside the QPAS."
const QPAS_TIMER_SECTIONS = (:direction, :multipliers, :step, :expand, :reduce, :refactor)

"""
Names of the sections timed around the individual factor operations (`KKTSystem.timer`):
assembling a Hessian column or rank-one vector from the pool, and the backend calls
`rowadd!`, `rowdel!`, `updown!`, `solve!`. They overlap the QPAS sections (e.g. a `rowadd`
during a reduction counts in both `:reduce` and `:rowadd`; rows re-added by a refactorization
count in `:rowadd` too).
"""
const FACTOR_TIMER_SECTIONS = (:assemble, :rowadd, :rowdel, :updown, :solve)
