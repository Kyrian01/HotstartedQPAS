"""
    StabilizationOptions(; delta = 1.0, center_update = :every_iteration)

Quadratic proximal stabilization (thesis §2.3, Chapter 4).

- `delta`: proximal parameter `δ > 0` (larger = stronger stabilization)
- `center_update`:
  - `:every_iteration`: `π̂ ← π` after every master solve (proximal point method)
  - `:best_bound`: `π̂ ← π` only when the Lagrangian bound improves, i.e. the stability
    center is the best dual point found so far (bundle method, thesis §2.3.1). This needs the
    bound at every iteration, so all pilots are priced every iteration.
- `initial_center`: starting `π̂` (`nothing` = zero vector)
"""
Base.@kwdef struct StabilizationOptions
    delta::Float64 = 1.0
    center_update::Symbol = :every_iteration
    initial_center::Union{Nothing,Vector{Float64}} = nothing
end

"""
    CGOptions(; kwargs...)

Settings of [`solve_column_generation`](@ref).

| field                 | default            | meaning                                                         |
|:----------------------|:-------------------|:----------------------------------------------------------------|
| `max_iterations`      | 100_000            | maximum number of CG iterations                                 |
| `max_columns`         | 20_000             | column capacity `n_max` (fixed-size structures, thesis §4.5.1)  |
| `time_limit`          | `Inf`              | wall-clock limit in seconds                                     |
| `pricing`             | `:first_improving` | `:first_improving`: pilots in random order, stop at the first   |
|                       |                    | negative reduced cost (thesis Alg. 1.3); `:all_blocks`: one     |
|                       |                    | column per pilot with negative reduced cost                     |
| `reduced_cost_tolerance` | 1e-6            | a column is improving if its reduced cost is `< -tol`           |
| `center_tolerance`    | 1e-6               | stabilized runs stop only once `‖π - π̂‖∞ ≤ tol·max(1, ‖π‖∞)`   |
| `compute_bound`       | false              | price all pilots every iteration to record the Lagrangian bound |
| `stabilization`       | `StabilizationOptions()` | used by stabilized masters only                           |
| `seed`                | 1234               | seed of the pricing order RNG                                   |
| `verbose`             | false              | print a progress line every `log_every` iterations              |
| `log_every`           | 100                |                                                                 |
| `diagnostics_every`   | 0                  | record memory use and solver internals (factor nnz, set sizes)  |
|                       |                    | every k iterations and at the end; 0 = off                      |

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
end

function validate(opts::CGOptions)
    opts.pricing in (:first_improving, :all_blocks) ||
        throw(ArgumentError("pricing must be :first_improving or :all_blocks"))
    opts.stabilization.center_update in (:every_iteration, :best_bound) ||
        throw(ArgumentError("center_update must be :every_iteration or :best_bound"))
    opts.stabilization.delta > 0 || throw(ArgumentError("δ must be positive"))
    opts.diagnostics_every >= 0 || throw(ArgumentError("diagnostics_every must be ≥ 0"))
    opts.max_columns >= 1 || throw(ArgumentError("max_columns must be positive"))
    return opts
end
