"""
    ExactResult

Solution of the direct edge-based LP relaxation (thesis eq. 5.2 with `0 ≤ x ≤ 1`).

- `objective`: optimal value
- `flows`: `n_edges × n_pilots` edge flows (column `k` is pilot `k`)
- `duals`: coupling-row duals `π` (HiGHS sign convention, comparable to the CG duals)
- `stats::ExactStats`
"""
struct ExactResult
    status::Symbol
    objective::Float64
    flows::Matrix{Float64}
    duals::Vector{Float64}
    stats::ExactStats
end

"""
    solve_exact(instance; output = false, solver = "simplex") -> ExactResult

Solve the full LP relaxation in one shot with HiGHS: all coupling rows and all pilots' flow
conservation rows are stacked into a single constraint matrix
(see [`stacked_constraint_matrix`](@ref)). `solver` is passed to HiGHS' `solver` option
(`"simplex"`, `"ipm"` or `"choose"`).
"""
function solve_exact(inst::CrewSchedulingInstance; output::Bool = false,
                     solver::AbstractString = "simplex")
    t_setup = time_ns()
    E, P, T = n_edges(inst), n_pilots(inst), n_tasks(inst)
    A = stacked_constraint_matrix(inst.network, inst.task_matrix, P)
    nvar, ncon = size(A, 2), size(A, 1)
    cost = vec(inst.edge_costs)                       # pilot-major, matches the column order
    # coupling rows keep their task bounds; flow rows are equalities F xₖ = b
    flow_rhs = stacked_flow_rhs(inst.network, P)
    row_lower = vcat(inst.task_lower, flow_rhs)
    row_upper = vcat(inst.task_upper, flow_rhs)

    model = HighsModel(; output)
    set_option!(model, "solver", solver)
    # LP relaxation: box 0 ≤ x ≤ 1 on every pilot's edge flows
    pass_lp!(model, A, cost, zeros(nvar), ones(nvar), row_lower, row_upper)
    setup_time = (time_ns() - t_setup) * 1e-9

    t_solve = time_ns()
    status = run!(model)
    solve_time = (time_ns() - t_solve) * 1e-9

    x = zeros(nvar)
    row_dual = zeros(ncon)
    objective = NaN
    if status === :optimal
        get_solution!(model, x, zeros(nvar), zeros(ncon), row_dual)
        objective = model_objective(model)
    end
    its = iteration_counts(model)
    close(model)

    stats = ExactStats("Exact LP (HiGHS $solver)", status, setup_time, solve_time,
                       its.simplex, its.ipm, nvar, ncon, nnz(A))
    # pilot-major x → E × P flow matrix; the first T row duals are the coupling duals π
    return ExactResult(status, objective, reshape(x, E, P), row_dual[1:T], stats)
end

objective_value(r::ExactResult) = r.objective
method_name(r::ExactResult) = r.stats.method
total_time(r::ExactResult) = r.stats.setup_time + r.stats.solve_time

function Base.show(io::IO, ::MIME"text/plain", r::ExactResult)
    s = r.stats
    println(io, "ExactResult — ", s.method)
    @printf(io, "  status            : %s\n", r.status)
    @printf(io, "  objective         : %.6f\n", r.objective)
    @printf(io, "  size              : %d variables, %d constraints, %d nonzeros\n",
            s.n_variables, s.n_constraints, s.n_nonzeros)
    @printf(io, "  setup / solve time: %.3f s / %.3f s\n", s.setup_time, s.solve_time)
    @printf(io, "  iterations        : %d simplex, %d ipm", s.simplex_iterations, s.ipm_iterations)
end
