"""
    CrewSchedulingInstance

LP relaxation of the multi-pilot crew scheduling problem (thesis eq. 9.1, §9.1.1 The model):

```
min  Σₖ cₖᵀ xₖ
s.t. task_lower ≤ Σₖ A_task xₖ ≤ task_upper      (coupling / flight coverage rows)
     F xₖ = b                     for every pilot k   (flow conservation)
     0 ≤ xₖ ≤ 1
```

Every pilot uses the same network. Coupling rows are equalities when
`task_lower[i] == task_upper[i]` and inequalities otherwise, so mixed equality/inequality
coverage is part of the data model from the start.

Fields:
- `network::FlowNetwork`: shared DAG
- `edge_costs::Matrix{Float64}`: `n_edges × n_pilots`, column `k` is `cₖ`
- `task_matrix::SparseMatrixCSC{Float64,Int}`: `n_tasks × n_edges` task–edge incidence
- `task_lower`, `task_upper`: coupling row bounds (`±Inf` allowed)
- `name::String`
"""
struct CrewSchedulingInstance
    network::FlowNetwork
    edge_costs::Matrix{Float64}
    task_matrix::SparseMatrixCSC{Float64,Int}
    task_lower::Vector{Float64}
    task_upper::Vector{Float64}
    name::String

    function CrewSchedulingInstance(network::FlowNetwork, edge_costs::Matrix{Float64},
                                    task_matrix::SparseMatrixCSC{Float64,Int},
                                    task_lower::Vector{Float64}, task_upper::Vector{Float64},
                                    name::AbstractString = "instance")
        E = n_edges(network)
        size(edge_costs, 1) == E ||
            throw(DimensionMismatch("edge_costs must have $E rows (one per edge)"))
        size(edge_costs, 2) >= 1 || throw(ArgumentError("need at least one pilot"))
        size(task_matrix, 2) == E ||
            throw(DimensionMismatch("task_matrix must have $E columns (one per edge)"))
        T = size(task_matrix, 1)
        length(task_lower) == length(task_upper) == T ||
            throw(DimensionMismatch("task bounds must have length $T"))
        all(task_lower .<= task_upper) || throw(ArgumentError("task_lower must be ≤ task_upper"))
        return new(network, edge_costs, task_matrix, task_lower, task_upper, String(name))
    end
end

n_pilots(inst::CrewSchedulingInstance) = size(inst.edge_costs, 2)
n_tasks(inst::CrewSchedulingInstance) = size(inst.task_matrix, 1)
n_edges(inst::CrewSchedulingInstance) = n_edges(inst.network)
n_nodes(inst::CrewSchedulingInstance) = n_nodes(inst.network)

"Number of variables of the edge-based (direct) LP formulation."
n_flow_variables(inst::CrewSchedulingInstance) = n_edges(inst) * n_pilots(inst)

"Logical mask of coupling rows that are equalities."
equality_rows(inst::CrewSchedulingInstance) = inst.task_lower .== inst.task_upper

"`true` if at least one coupling row is an inequality."
has_inequalities(inst::CrewSchedulingInstance) = !all(equality_rows(inst))

"Edge costs of pilot `k` (a view)."
pilot_costs(inst::CrewSchedulingInstance, k::Integer) = view(inst.edge_costs, :, k)

function Base.show(io::IO, inst::CrewSchedulingInstance)
    neq = count(equality_rows(inst))
    print(io, "CrewSchedulingInstance(\"", inst.name, "\": |V|=", n_nodes(inst),
          ", |E|=", n_edges(inst), ", pilots=", n_pilots(inst), ", tasks=", n_tasks(inst),
          " (", neq, " eq, ", n_tasks(inst) - neq, " ineq), N=", n_flow_variables(inst), ")")
end
