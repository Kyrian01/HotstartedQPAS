"""
    ShortestPathPricer(instance)

Pricing oracle of the crew scheduling master problem (thesis eq. 2.5, the pricing problem, as
a shortest path problem in §9.1.1). For pilot `k` and coupling duals `π`, it finds the
source–sink path `x` minimising the reduced edge costs

    (cₖ - A_taskᵀ π)ᵀ x

by dynamic programming over the topologically numbered DAG (linear in `|V| + |E|`). The
column's reduced cost is this value minus the convexity dual `μₖ`.

Usage per pricing round: `set_duals!(pricer, π)` once, then `price_block!(pricer, k)` for any
number of pilots. The most recent path stays in `pricer.path` until the next call.
"""
mutable struct ShortestPathPricer
    network::FlowNetwork
    edge_costs::Matrix{Float64}
    task_matrix::SparseMatrixCSC{Float64,Int}
    task_dual::Vector{Float64}     # (A_taskᵀ π)[e]
    dist::Vector{Float64}
    pred::Vector{Int}
    path::Vector{Int}
    coverage::Vector{Float64}      # dense scatter buffer, all zeros between calls
    cover_rows::Vector{Int}
    cover_vals::Vector{Float64}
end

function ShortestPathPricer(inst::CrewSchedulingInstance)
    E = n_edges(inst)
    V = n_nodes(inst)
    T = n_tasks(inst)
    return ShortestPathPricer(inst.network, inst.edge_costs, inst.task_matrix,
                              zeros(E), fill(Inf, V), zeros(Int, V), Int[],
                              zeros(T), Int[], Float64[])
end

"Store the coupling duals `π` for the following pricing calls."
function set_duals!(pricer::ShortestPathPricer, π::AbstractVector{Float64})
    # per-edge dual prize (A_taskᵀπ)_e, shared by all pilots in this pricing round
    mul!(pricer.task_dual, transpose(pricer.task_matrix), π)
    return pricer
end

"""
    price_block!(pricer, k) -> (path_value, path_cost)

Shortest path for pilot `k` under the current duals. `path_value = (cₖ - A_taskᵀπ)ᵀ x` (the
reduced cost before subtracting `μₖ`), `path_cost = cₖᵀ x`. The path is left in `pricer.path`.
"""
function price_block!(pricer::ShortestPathPricer, k::Integer)
    net = pricer.network
    dist = pricer.dist
    pred = pricer.pred
    costs = pricer.edge_costs
    tdual = pricer.task_dual
    fill!(dist, Inf)
    dist[net.source] = 0.0
    # DAG relaxation in topological (index) order: each node is final when it is scanned
    @inbounds for v in net.source:net.n_nodes
        dv = dist[v]
        isfinite(dv) || continue
        for e in out_edges(net, v)
            h = net.head[e]
            # reduced cost of edge e for pilot k: c_ek − (A_taskᵀπ)_e
            d = dv + (costs[e, k] - tdual[e])
            if d < dist[h]
                dist[h] = d
                pred[h] = e
            end
        end
    end
    isfinite(dist[net.sink]) || error("pricing: sink unreachable for pilot $k")

    # backtrack predecessor edges sink → source, accumulating the true cost cₖᵀx
    path = pricer.path
    empty!(path)
    v = net.sink
    cost = 0.0
    @inbounds while v != net.source
        e = pred[v]
        push!(path, e)
        cost += costs[e, k]
        v = net.tail[e]
    end
    reverse!(path)
    return dist[net.sink], cost
end

"""
    path_coverage!(pricer) -> (rows, vals)

Coupling coefficients `A_task x` of the last priced path, as sorted sparse indices/values
(views into internal buffers, valid until the next call).
"""
function path_coverage!(pricer::ShortestPathPricer)
    A = pricer.task_matrix
    cov = pricer.coverage
    rows = pricer.cover_rows
    vals = pricer.cover_vals
    empty!(rows)
    # scatter-add the task columns of all path edges: cov = A_task x (records touched rows)
    @inbounds for e in pricer.path
        for t in A.colptr[e]:(A.colptr[e + 1] - 1)
            i = A.rowval[t]
            cov[i] == 0.0 && push!(rows, i)
            cov[i] += A.nzval[t]
        end
    end
    # gather into sorted sparse form and zero the scatter buffer for the next call
    sort!(rows)
    resize!(vals, length(rows))
    @inbounds for (t, i) in enumerate(rows)
        vals[t] = cov[i]
        cov[i] = 0.0
    end
    return rows, vals
end
