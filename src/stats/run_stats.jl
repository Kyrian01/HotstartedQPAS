"""
    CGStats(method)

Statistics of one column generation run: totals plus one trace entry per CG iteration.

Per-iteration traces (all of equal length `iterations`):

| field              | meaning                                                                 |
|:-------------------|:------------------------------------------------------------------------|
| `master_time`      | wall-clock seconds of the restricted master solve                       |
| `pricing_time`     | wall-clock seconds of the pricing round                                 |
| `master_objective` | objective of the restricted master (incl. Big-M / proximal terms)       |
| `operational_cost` | `cᵀλ` of the master solution (the "Obj" column of the thesis tables)    |
| `infeasibility`    | `‖violation of row_lower ≤ Aλ ≤ row_upper‖∞`                            |
| `lower_bound`      | Lagrangian bound `θ(π)` (NaN if not all pilots were priced)             |
| `dual_change`      | `‖πₖ - πₖ₋₁‖∞` (dual oscillation, thesis Fig. 5.5)                      |
| `n_columns`        | columns in the master after the iteration                               |
| `inner_iterations` | simplex / QP / active-set iterations of the master solve                |
| `columns_added`    | columns added by the pricing round                                      |
| `center_updated`   | stabilised runs: whether the stability center moved                     |

`diagnostics` holds sampled snapshots (one `NamedTuple` per sample, see
`CGOptions(diagnostics_every = k)`): memory use and solver internals such as the factor
size. `details` holds solver-specific totals (e.g. QPAS working-set updates and time breakdown)
as an ordered list of `name => value` pairs.
"""
mutable struct CGStats
    method::String
    status::Symbol
    iterations::Int
    total_time::Float64
    setup_time::Float64
    duplicate_columns::Int
    master_time::Vector{Float64}
    pricing_time::Vector{Float64}
    master_objective::Vector{Float64}
    operational_cost::Vector{Float64}
    infeasibility::Vector{Float64}
    lower_bound::Vector{Float64}
    dual_change::Vector{Float64}
    n_columns::Vector{Int}
    inner_iterations::Vector{Int}
    columns_added::Vector{Int}
    center_updated::Vector{Bool}
    diagnostics::Vector{NamedTuple}
    details::Vector{Pair{String,Any}}
end

CGStats(method::AbstractString) =
    CGStats(String(method), :not_started, 0, 0.0, 0.0, 0,
            Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
            Int[], Int[], Int[], Bool[], NamedTuple[], Pair{String,Any}[])

"Append one CG iteration to the traces."
function record_iteration!(s::CGStats; master_time, pricing_time, master_objective,
                           operational_cost, infeasibility, lower_bound, dual_change,
                           n_columns, inner_iterations, columns_added, center_updated)
    s.iterations += 1
    push!(s.master_time, master_time)
    push!(s.pricing_time, pricing_time)
    push!(s.master_objective, master_objective)
    push!(s.operational_cost, operational_cost)
    push!(s.infeasibility, infeasibility)
    push!(s.lower_bound, lower_bound)
    push!(s.dual_change, dual_change)
    push!(s.n_columns, n_columns)
    push!(s.inner_iterations, inner_iterations)
    push!(s.columns_added, columns_added)
    push!(s.center_updated, center_updated)
    return s
end

total_master_time(s::CGStats) = sum(s.master_time; init = 0.0)
total_pricing_time(s::CGStats) = sum(s.pricing_time; init = 0.0)
mean_master_time(s::CGStats) = s.iterations == 0 ? NaN : total_master_time(s) / s.iterations
mean_pricing_time(s::CGStats) = s.iterations == 0 ? NaN : total_pricing_time(s) / s.iterations
total_inner_iterations(s::CGStats) = sum(s.inner_iterations; init = 0)

"First iteration whose master solution satisfies the coupling rows within `tol` (or `nothing`)."
first_feasible_iteration(s::CGStats; tol::Real = 1e-6) = findfirst(<=(tol), s.infeasibility)

"Best (largest) Lagrangian bound encountered, `-Inf` if none was computed."
best_lower_bound(s::CGStats) = maximum(filter(!isnan, s.lower_bound); init = -Inf)

"Number of iterations in which the stability center moved."
center_updates(s::CGStats) = count(s.center_updated)

"""
    ExactStats

Statistics of a direct (monolithic) LP solve.
"""
struct ExactStats
    method::String
    status::Symbol
    setup_time::Float64
    solve_time::Float64
    simplex_iterations::Int
    ipm_iterations::Int
    n_variables::Int
    n_constraints::Int
    n_nonzeros::Int
end

"""
    write_diagnostics_csv(path, stats::CGStats)

Write the sampled diagnostics of a CG run (`CGOptions(diagnostics_every = k)`) as CSV, one
row per sample. Columns are the fields of the first sample; missing fields are left empty.
"""
function write_diagnostics_csv(path::AbstractString, s::CGStats)
    mkpath(dirname(abspath(path)))
    isempty(s.diagnostics) && return path
    keys_ = collect(keys(s.diagnostics[1]))
    open(path, "w") do io
        println(io, join(keys_, ','))
        for d in s.diagnostics
            println(io, join((haskey(d, k) ? d[k] : "" for k in keys_), ','))
        end
    end
    return path
end

"""
    write_trace_csv(path, stats::CGStats)

Write the per-iteration traces of a CG run as CSV (one row per iteration).
"""
function write_trace_csv(path::AbstractString, s::CGStats)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "iteration,master_time,pricing_time,master_objective,operational_cost,",
                "infeasibility,lower_bound,dual_change,n_columns,inner_iterations,",
                "columns_added,center_updated")
        for k in 1:s.iterations
            println(io, k, ',', s.master_time[k], ',', s.pricing_time[k], ',',
                    s.master_objective[k], ',', s.operational_cost[k], ',',
                    s.infeasibility[k], ',', s.lower_bound[k], ',', s.dual_change[k], ',',
                    s.n_columns[k], ',', s.inner_iterations[k], ',', s.columns_added[k], ',',
                    Int(s.center_updated[k]))
        end
    end
    return path
end
