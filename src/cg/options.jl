"""
    StabilizationOptions(; delta = 1.0, center_update = :every_iteration)

Quadratic proximal stabilization (thesis Chapter 3: §3.1 Proximal Stabilization of the Dual,
§3.4.1 Center updates).

- `delta`: proximal parameter `δ > 0` (larger = stronger stabilization)
- `center_update`:
  - `:every_iteration`: `π̂ ← π` after every master solve (proximal point rule)
  - `:best_bound`: in iterations that add columns, `π̂ ← π` only when the Lagrangian bound
    `θ(π⁺)` improves, i.e. the stability center is the best dual point found so far (bundle
    rule, thesis §3.4.1 Center updates). This needs the bound at every iteration, so all
    pilots are priced every iteration. In iterations without a new column the center always
    moves (center-only iteration, as in thesis Alg. 3.1), unless the run has converged.
- `initial_center`: starting `π̂` (`nothing` = zero vector)

Adaptive stabilization (QPAS master only; thesis §3.5, Def. 3.5.3, Geometric stabilization
parameter; `docs/adaptive_stabilization.md`). `delta` is then the base value `δ₀` and the
largest δ used:
- `delta_update`:
  - `:fixed`: δ stays `delta` (thesis behaviour)
  - `:geometric`: `δ ← max(delta_min, delta_decrease·δ)` after every center-only iteration,
    `δ ← min(δ₀, delta_increase·δ)` after every iteration that adds a column
    (default: double; `delta_increase = Inf`: reset to `δ₀`)
"""
Base.@kwdef struct StabilizationOptions
    delta::Float64 = 1.0
    center_update::Symbol = :every_iteration
    initial_center::Union{Nothing,Vector{Float64}} = nothing
    delta_update::Symbol = :fixed
    delta_decrease::Float64 = 0.5
    delta_increase::Float64 = 2.0
    delta_min::Float64 = 1e-6
end

"True if δ follows an adaptive rule (needs `supports_delta_update`)."
is_adaptive(s::StabilizationOptions) = s.delta_update !== :fixed

"""
    CGOptions(; kwargs...)

Settings of [`solve_column_generation`](@ref).

| field                 | default            | meaning                                                         |
|:----------------------|:-------------------|:----------------------------------------------------------------|
| `max_iterations`      | 100_000            | maximum number of CG iterations                                 |
| `max_columns`         | 20_000             | column capacity `n_max` (fixed-size structures, thesis §6.1)    |
| `time_limit`          | `Inf`              | wall-clock limit in seconds                                     |
| `pricing`             | `:first_improving` | `:first_improving`: pilots in random order, stop at the first   |
|                       |                    | negative reduced cost (thesis Alg. 3.1); `:all_blocks`: one     |
|                       |                    | column per pilot with negative reduced cost (thesis Alg. 2.1)   |
| `reduced_cost_tolerance` | 1e-6            | a column is improving if its reduced cost is `< -tol`           |
| `center_tolerance`    | 1e-6               | stabilized runs stop only once `‖π - π̂‖∞ ≤ tol·max(1, ‖π‖∞)`   |
| `compute_bound`       | false              | price all pilots every iteration to record the Lagrangian bound |
| `stabilization`       | `StabilizationOptions()` | used by stabilized masters only                           |
| `seed`                | 1234               | seed of the pricing order RNG                                   |
| `verbose`             | false              | print a progress line every `log_every` iterations              |
| `log_every`           | 100                |                                                                 |
| `diagnostics_every`   | 0                  | record memory use and solver internals (factor nnz, set sizes)  |
|                       |                    | every k iterations and at the end; 0 = off                      |
| `diagnostics_residual`| false              | the diagnostics also rebuild `H_eff` and record the factor      |
|                       |                    | residual `‖LDLᵀ - H_eff‖` (QPAS; `O(m|N|² + |N|³)` per sample)  |

Termination is exact: column generation stops when a complete pricing round over all pilots
finds no column with negative reduced cost (and, for stabilized masters, the stability
center has converged). There is no heuristic "k iterations without improvement" rule.
"""
Base.@kwdef struct CGOptions
    max_iterations::Int = 100_000
    max_columns::Int = 20_000
    time_limit::Float64 = Inf
    pricing::Symbol = :first_improving
    reduced_cost_tolerance::Float64 = 1e-6
    center_tolerance::Float64 = 1e-6
    compute_bound::Bool = false
    stabilization::StabilizationOptions = StabilizationOptions()
    seed::Int = 1234
    verbose::Bool = false
    log_every::Int = 100
    diagnostics_every::Int = 0
    diagnostics_residual::Bool = false
end

function validate(opts::CGOptions)
    opts.pricing in (:first_improving, :all_blocks) ||
        throw(ArgumentError("pricing must be :first_improving or :all_blocks"))
    opts.stabilization.center_update in (:every_iteration, :best_bound) ||
        throw(ArgumentError("center_update must be :every_iteration or :best_bound"))
    s = opts.stabilization
    s.delta > 0 || throw(ArgumentError("δ must be positive"))
    s.delta_update in (:fixed, :geometric) ||
        throw(ArgumentError("delta_update must be :fixed or :geometric"))
    0 < s.delta_decrease <= 1 || throw(ArgumentError("delta_decrease must lie in (0, 1]"))
    s.delta_increase >= 1 || throw(ArgumentError("delta_increase must be ≥ 1"))
    0 < s.delta_min <= s.delta || throw(ArgumentError("delta_min must lie in (0, delta]"))
    opts.diagnostics_every >= 0 || throw(ArgumentError("diagnostics_every must be ≥ 0"))
    opts.max_columns >= 1 || throw(ArgumentError("max_columns must be positive"))
    return opts
end
