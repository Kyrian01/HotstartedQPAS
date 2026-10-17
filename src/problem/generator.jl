# Random instance generator for the crew scheduling LP (thesis §9.1.2, Instance generator).
#
# Pipeline:
#   1. `generate_network`       : time-ordered DAG = backbone chain i → i+1 plus random skip edges
#   2. `generate_task_matrix`   : skip edges carry tasks; edges sharing a task overlap in time,
#                                 so a single path covers each task at most once
#   3. `coverable_tasks`        : max-coverage LP with all pilots; keep tasks covered exactly once
#   4. `generate_instance`      : subsample tasks, choose row senses, perturb costs per pilot

"""
    InstanceConfig(; kwargs...)

Parameters of [`generate_instance`](@ref).

| field                 | meaning                                                              |
|:----------------------|:---------------------------------------------------------------------|
| `n_nodes`             | number of network nodes `|V|`                                        |
| `edge_probability`    | probability of a skip edge `i → j` (`j ≥ i+2`)                       |
| `n_tasks`             | number of candidate tasks before selection                           |
| `max_tasks_per_edge`  | maximum number of tasks attached to one skip edge                    |
| `task_window`         | half-width of the task window around an edge (0 → `n_tasks ÷ 10`)   |
| `n_pilots`            | number of pilots (blocks) `p`                                        |
| `task_fraction`       | fraction of the coverable tasks that become coupling rows            |
| `inequality_fraction` | fraction of the coupling rows turned into `≥ 1` rows                 |
| `pilot_cost_spread`   | pilot `k` pays `c_e ⋅ (1 + spread ⋅ U(0,1))` on edge `e`             |
| `seed`                | RNG seed                                                             |
| `name`                | instance name                                                        |
"""
Base.@kwdef struct InstanceConfig
    n_nodes::Int = 50
    edge_probability::Float64 = 0.1
    n_tasks::Int = 40
    max_tasks_per_edge::Int = 10
    task_window::Int = 0
    n_pilots::Int = 4
    task_fraction::Float64 = 1.0
    inequality_fraction::Float64 = 0.0
    pilot_cost_spread::Float64 = 0.1
    seed::Int = 1
    name::String = "generated"
end

"""
    generate_instance(config::InstanceConfig) -> CrewSchedulingInstance

Generate a random, feasible crew scheduling instance. Experiments normally get their
`InstanceConfig` from the `[instance]` section of a config file (see `experiments.toml`).
"""
function generate_instance(cfg::InstanceConfig)
    _validate(cfg)
    rng = Xoshiro(cfg.seed)

    network, base_cost, task_edges = generate_network(rng, cfg.n_nodes, cfg.edge_probability)
    task_matrix = generate_task_matrix(rng, network, task_edges, cfg)

    # keep only tasks that some feasible schedule covers, then subsample task_fraction of them
    covered = coverable_tasks(network, task_matrix, cfg.n_pilots)
    keep = [t for t in covered if rand(rng) < cfg.task_fraction]
    isempty(keep) && error("generated instance has no coverable tasks; adjust the configuration")
    A = task_matrix[keep, :]
    T = length(keep)

    # all rows start as equalities Σₖ A_task xₖ = 1; a random subset is relaxed to ≥ 1
    lower = ones(T)
    upper = ones(T)
    n_ineq = round(Int, cfg.inequality_fraction * T)
    for i in randperm(rng, T)[1:n_ineq]
        upper[i] = Inf                       # cover at least once
    end

    # pilot-specific costs: multiplicative perturbation of the shared base edge cost
    costs = Matrix{Float64}(undef, length(base_cost), cfg.n_pilots)
    for k in 1:cfg.n_pilots, e in eachindex(base_cost)
        costs[e, k] = base_cost[e] * (1 + cfg.pilot_cost_spread * rand(rng))
    end

    return CrewSchedulingInstance(network, costs, dropzeros(A), lower, upper, cfg.name)
end

function _validate(cfg::InstanceConfig)
    cfg.n_nodes >= 3 || throw(ArgumentError("n_nodes must be ≥ 3"))
    0 <= cfg.edge_probability <= 1 || throw(ArgumentError("edge_probability must be in [0, 1]"))
    cfg.n_tasks >= 1 || throw(ArgumentError("n_tasks must be ≥ 1"))
    cfg.n_pilots >= 1 || throw(ArgumentError("n_pilots must be ≥ 1"))
    0 < cfg.task_fraction <= 1 || throw(ArgumentError("task_fraction must be in (0, 1]"))
    0 <= cfg.inequality_fraction <= 1 || throw(ArgumentError("inequality_fraction must be in [0, 1]"))
    cfg.pilot_cost_spread >= 0 || throw(ArgumentError("pilot_cost_spread must be ≥ 0"))
    return nothing
end

"""
    generate_network(rng, n_nodes, edge_probability) -> (network, cost, skip_edges)

Backbone edges `i → i+1` guarantee a source–sink path; each skip edge `i → j` (`j ≥ i+2`)
exists with probability `edge_probability`. Edge cost is `(j - i) + U(0,1)`. Returns the
indices of the skip edges, which are the ones that can carry tasks.
"""
function generate_network(rng::AbstractRNG, n_nodes::Integer, edge_probability::Real)
    tail = Int[]
    head = Int[]
    skip_edges = Int[]
    for i in 1:n_nodes
        # backbone edge i → i+1 (keeps the sink reachable)
        if i < n_nodes
            push!(tail, i)
            push!(head, i + 1)
        end
        # random skip edges i → j, j ≥ i+2 (the only edges that may carry tasks)
        for j in (i + 2):n_nodes
            if rand(rng) < edge_probability
                push!(tail, i)
                push!(head, j)
                push!(skip_edges, length(tail))
            end
        end
    end
    # cost grows with the time span j − i of the edge, plus U(0,1) noise
    cost = [(head[e] - tail[e]) + rand(rng) for e in eachindex(tail)]
    return FlowNetwork(n_nodes, tail, head), cost, skip_edges
end

# Two edges overlap in time if their node intervals intersect; a path never uses two
# overlapping edges, so tasks whose edges pairwise overlap are covered at most once per path.
@inline _overlap(net::FlowNetwork, e1::Int, e2::Int) =
    net.tail[e1] < net.head[e2] && net.tail[e2] < net.head[e1]

"""
    generate_task_matrix(rng, network, skip_edges, cfg) -> SparseMatrixCSC

Attach each skip edge to a random subset of tasks in a window around its position in time.
An edge may join a task only if it overlaps every edge already attached to that task.
"""
function generate_task_matrix(rng::AbstractRNG, net::FlowNetwork, skip_edges::Vector{Int},
                              cfg::InstanceConfig)
    T = cfg.n_tasks
    M = net.n_nodes
    window = cfg.task_window > 0 ? cfg.task_window : max(1, T ÷ 10)
    task_edges = [Int[] for _ in 1:T]
    rows = Int[]
    cols = Int[]
    valid = Int[]
    for e in skip_edges
        # map the edge's midpoint in time to a task index; tasks are searched in ±window of it
        progress = ((net.tail[e] - 1) + (net.head[e] - 1)) / (2M)
        center = min(floor(Int, progress * T), T - 1) + 1
        # candidate tasks: e must overlap all edges already on t (≤ 1 coverage per path)
        empty!(valid)
        for t in max(1, center - window):min(T, center + window)
            if all(e2 -> _overlap(net, e, e2), task_edges[t])
                push!(valid, t)
            end
        end
        isempty(valid) && continue
        # attach e to a random subset of 1..max_tasks_per_edge valid tasks
        n_pick = rand(rng, 1:min(length(valid), cfg.max_tasks_per_edge))
        for t in valid[randperm(rng, length(valid))[1:n_pick]]
            push!(rows, t)
            push!(cols, e)
            push!(task_edges[t], e)
        end
    end
    return sparse(rows, cols, ones(length(rows)), T, n_edges(net))
end

"""
    coverable_tasks(network, task_matrix, n_pilots; tol = 1e-7) -> Vector{Int}

Solve the max-coverage LP `max Σᵢ coverageᵢ s.t. coverage ≤ 1, F xₖ = b, 0 ≤ x ≤ 1` and
return the tasks covered exactly once. The equality-constrained instance restricted to these
tasks is feasible by construction (the LP solution itself is feasible).
"""
function coverable_tasks(net::FlowNetwork, task_matrix::SparseMatrixCSC, n_pilots::Integer;
                         tol::Real = 1e-7)
    T = size(task_matrix, 1)
    A = stacked_constraint_matrix(net, task_matrix, n_pilots)
    nvar = size(A, 2)
    # objective weight of edge e = number of tasks it covers, i.e. 1ᵀA_task
    cover_weight = vec(sum(task_matrix; dims = 1))
    cost = -repeat(cover_weight, n_pilots)            # maximise coverage
    flow_rhs = stacked_flow_rhs(net, n_pilots)
    # coverage rows Σₖ A_task xₖ ≤ 1; flow rows F xₖ = b as equalities
    row_lower = vcat(fill(-Inf, T), flow_rhs)
    row_upper = vcat(ones(T), flow_rhs)

    model = HighsModel()
    # This LP is highly degenerate: dual simplex needs minutes from |V| ≈ 200 on, interior
    # point (with crossover) a second or two. The selected tasks are the same on I1–I3.
    set_option!(model, "solver", "ipm")
    pass_lp!(model, A, cost, zeros(nvar), ones(nvar), row_lower, row_upper)
    status = run!(model)
    status === :optimal || error("coverable_tasks: max-coverage LP returned $status")
    x = zeros(nvar)
    get_solution!(model, x, zeros(nvar), zeros(size(A, 1)), zeros(size(A, 1)))
    close(model)

    # aggregate edge flows over pilots (x is pilot-major) and compute A_task Σₖ xₖ
    E = n_edges(net)
    total_flow = zeros(E)
    for k in 1:n_pilots
        total_flow .+= view(x, ((k - 1) * E + 1):(k * E))
    end
    coverage = task_matrix * total_flow
    return [i for i in 1:T if coverage[i] >= 1 - tol]
end
