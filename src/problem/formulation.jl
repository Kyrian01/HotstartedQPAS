# Direct (edge-based) LP formulation of the crew scheduling problem, shared by the exact
# solver and the instance generator.
#
# Variable order is pilot-major: x = [x₁; x₂; …; x_P] with xₖ ∈ R^{|E|}.
# Row order: [coupling rows (n_tasks); flow rows of pilot 1 (|V|); …; flow rows of pilot P].

"""
    stacked_constraint_matrix(network, task_matrix, n_pilots) -> SparseMatrixCSC

Constraint matrix of the direct formulation (thesis eq. 9.1, §9.1.1 The model):

```
[ A_task  A_task  ⋯  A_task ]
[   F                       ]
[          F                ]
[                ⋱          ]
[                         F ]
```
"""
function stacked_constraint_matrix(network::FlowNetwork, task_matrix::SparseMatrixCSC,
                                   n_pilots::Integer)
    F = incidence_matrix(network)
    # every pilot's edges cover tasks the same way: coupling block [A_task ⋯ A_task]
    coupling = repeat(task_matrix, 1, n_pilots)
    # flow conservation is separate per pilot: block-diagonal diag(F, …, F)
    flows = blockdiag(ntuple(_ -> F, n_pilots)...)
    return vcat(coupling, flows)
end

"Row bounds of the flow conservation rows for all pilots (equalities `F xₖ = b`)."
function stacked_flow_rhs(network::FlowNetwork, n_pilots::Integer)
    return repeat(supply_vector(network), n_pilots)
end
