"""
    QPASOptions(; kwargs...)

Settings of the hot-started active-set solver.

| field                 | default  | meaning                                                                     |
|:----------------------|:---------|:----------------------------------------------------------------------------|
| `max_iterations`      | 100_000  | active-set iterations per master solve                                      |
| `step_tolerance`      | 1e-10    | `‖p‖∞` below which the search direction is treated as zero                  |
| `dual_tolerance`      | 1e-10    | multipliers `λⱼ ≥ -dual_tolerance ⋅ max(1, maxⱼ|cⱼ|)` are accepted          |
| `ratio_tolerance`     | 1e-14    | components `pⱼ ≥ -ratio_tolerance` are ignored in the ratio test            |
| `singular_tolerance`  | 1e-10    | relative pivot `D[k]/H[k,k]` below which the reduced Hessian is singular    |
| `check_pivots`        | true     | also inspect the pivots after expansions (rebuild the factor if one decays) |
| `refactor_interval`   | 150      | rebuild the factor after this many row updates (0 = never), see below       |
| `sparse_solves`       | true     | solve only on the pattern `N` (CHOLMOD `Bset`) instead of all `n_max` rows  |
| `debug`               | false    | keep the explicit reduced Hessian and verify every update (slow, tests only)|

Why periodic refactorization: `cholmod_rowdel` zeroes a row/column numerically but leaves
its entries in the sparsity pattern of `L`. Over thousands of working-set updates these
structural zeros accumulate (e.g. 438k stored entries instead of 50k on instance I2), and
every solve and row update pays for them. Rebuilding the factor from the current set `N`
(`refactor!`) removes them; with a 150-update interval this is ~8x faster on I2.
"""
Base.@kwdef struct QPASOptions
    max_iterations::Int = 100_000
    step_tolerance::Float64 = 1e-10
    dual_tolerance::Float64 = 1e-10
    ratio_tolerance::Float64 = 1e-14
    singular_tolerance::Float64 = 1e-10
    check_pivots::Bool = true
    refactor_interval::Int = 150
    sparse_solves::Bool = true
    debug::Bool = false
end

"""
    QPASCounters

Event counters of the active-set solver (per solve and cumulative).

- `iterations`: active-set iterations (direction computations + multiplier checks)
- `steps`, `full_steps`: primal steps taken, and those with `α = 1`
- `expansions`: blocking constraints added to W (thesis §4.4.1)
- `representative_swaps`: expansions where the blocking index was a representative
- `reductions`: constraints released from W (thesis §4.4.2)
- `singular_steps`: zero-curvature steps taken because the reduced Hessian would become
  singular (thesis §4.4.5)
- `factor_updates`: CHOLMOD row additions + deletions
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
