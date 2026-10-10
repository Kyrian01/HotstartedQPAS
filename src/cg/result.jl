"""
    CGResult

Outcome of [`solve_column_generation`](@ref).

- `method`, `status` (`:optimal`, `:iteration_limit`, `:column_limit`, `:time_limit`,
  `:master_failure`)
- `objective`: operational cost `cᵀλ` of the final master solution
- `master_objective`: final restricted master objective (incl. penalty / proximal terms)
- `lower_bound`: best Lagrangian bound found (`-Inf` if none was computed)
- `infeasibility`: coupling-row violation of the final solution (∞-norm)
- `weights`: final column weights `λ`; `duals`: coupling duals `π`; `convexity_duals`: `μ`
- `pool`: all generated columns (paths, costs, coefficients)
- `stats::CGStats`
"""
struct CGResult
    method::String
    status::Symbol
    objective::Float64
    master_objective::Float64
    lower_bound::Float64
    infeasibility::Float64
    weights::Vector{Float64}
    duals::Vector{Float64}
    convexity_duals::Vector{Float64}
    pool::ColumnPool
    stats::CGStats
end

objective_value(r::CGResult) = r.objective
method_name(r::CGResult) = r.method
total_time(r::CGResult) = r.stats.total_time
n_columns(r::CGResult) = n_columns(r.pool)

"""
    edge_flows(result, instance) -> Matrix{Float64}

Recover the edge-based solution `xₖ = Σⱼ λⱼ xʲ` (an `n_edges × n_pilots` matrix) from the
column weights, comparable to [`ExactResult`](@ref)`.flows`.
"""
function edge_flows(r::CGResult, inst::CrewSchedulingInstance)
    flows = zeros(n_edges(inst), n_pilots(inst))
    pool = r.pool
    for j in 1:pool.n
        λ = r.weights[j]
        λ == 0.0 && continue
        k = pool.block[j]
        for e in pool.paths[j]
            flows[e, k] += λ
        end
    end
    return flows
end

"""
    active_columns(result; tol = 1e-9) -> Vector{NamedTuple}

Columns with weight above `tol`: `(index, pilot, weight, cost, path)`, sorted by pilot.
"""
function active_columns(r::CGResult; tol::Real = 1e-9)
    pool = r.pool
    cols = [(index = j, pilot = pool.block[j], weight = r.weights[j], cost = pool.cost[j],
             path = pool.paths[j]) for j in 1:pool.n if r.weights[j] > tol]
    sort!(cols; by = c -> (c.pilot, -c.weight))
    return cols
end

function Base.show(io::IO, ::MIME"text/plain", r::CGResult)
    s = r.stats
    println(io, "CGResult — ", r.method)
    @printf(io, "  status              : %s\n", r.status)
    @printf(io, "  objective (cᵀλ)     : %.6f\n", r.objective)
    @printf(io, "  master objective    : %.6f\n", r.master_objective)
    @printf(io, "  Lagrangian bound    : %.6f\n", r.lower_bound)
    @printf(io, "  infeasibility       : %.2e\n", r.infeasibility)
    @printf(io, "  iterations          : %d (%d columns)\n", s.iterations, n_columns(r.pool))
    @printf(io, "  total time          : %.3f s (master %.3f s, pricing %.3f s)\n",
            s.total_time, total_master_time(s), total_pricing_time(s))
    @printf(io, "  avg master time     : %.3f ms", 1e3 * mean_master_time(s))
end
