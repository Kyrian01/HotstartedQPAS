"""
    HighsLPMaster(; big_m = 1e4, output = false)

Classical (unstabilized) LP restricted master problem solved by a persistent HiGHS instance
(thesis Alg. 2.1, Classical Column Generation; setup in §9.2). New columns are appended with `Highs_addCols`, so every solve
warm-starts from the previous simplex basis.

```
min  M·1ᵀ(s⁺ + s⁻) + cᵀλ
s.t. row_lower ≤ AXλ + s⁺ - s⁻ ≤ row_upper
     Cλ = 1,   λ, s⁺, s⁻ ≥ 0
```

The Big-M slacks keep the master feasible before enough columns exist. Equality and
inequality coupling rows are both supported.
"""
mutable struct HighsLPMaster <: AbstractMasterSolver
    big_m::Float64
    output::Bool
    model::Union{Nothing,HighsModel}
    n_rows::Int
    n_blocks::Int
    n_columns::Int
    solution::Union{Nothing,MasterSolution}
    artificial_activity::Float64
    # buffers
    col_value::Vector{Float64}
    col_dual::Vector{Float64}
    row_value::Vector{Float64}
    row_dual::Vector{Float64}
    add_colptr::Vector{Int}
    add_rowval::Vector{Int}
    add_nzval::Vector{Float64}
end

HighsLPMaster(; big_m::Real = 1e4, output::Bool = false) =
    HighsLPMaster(Float64(big_m), output, nothing, 0, 0, 0, nothing, 0.0,
                  Float64[], Float64[], Float64[], Float64[], Int[], Int[], Float64[])

method_name(::HighsLPMaster) = "LP CG (HiGHS)"
is_stabilized(::HighsLPMaster) = false
supports_inequalities(::HighsLPMaster) = true

function initialize!(m::HighsLPMaster, spec::MasterSpec, pool::ColumnPool)
    release!(m)
    T, P = n_rows(spec), spec.n_blocks
    m.n_rows, m.n_blocks, m.n_columns = T, P, 0
    m.solution = MasterSolution(spec)
    model = HighsModel(; output = m.output)
    m.model = model

    # rows 1..T: coupling rows AXλ ∈ [row_lower, row_upper]; rows T+1..T+P: convexity Cλ = 1
    append_rows!(model, spec.row_lower, spec.row_upper)
    append_rows!(model, ones(P), ones(P))

    # slacks s⁺ (+1) and s⁻ (−1), one per coupling row (single nonzero each), cost M
    for sign in (1.0, -1.0)
        append_columns!(model, fill(m.big_m, T), zeros(T), fill(Inf, T),
                        collect(1:(T + 1)), collect(1:T), fill(sign, T))
    end

    # solution buffers sized for [s⁺; s⁻; λ] columns and [coupling; convexity] rows
    nslack = 2T
    resize!(m.col_value, nslack + spec.capacity)
    resize!(m.col_dual, nslack + spec.capacity)
    resize!(m.row_value, T + P)
    resize!(m.row_dual, T + P)

    add_columns!(m, pool, 1:n_columns(pool))
    return m
end

function add_columns!(m::HighsLPMaster, pool::ColumnPool, cols::UnitRange{Int})
    isempty(cols) && return m
    first(cols) == m.n_columns + 1 || throw(ArgumentError("columns must be added in order"))
    colptr, rowval, nzval = m.add_colptr, m.add_rowval, m.add_nzval
    empty!(colptr)
    empty!(rowval)
    empty!(nzval)
    # assemble the new columns in CSC form: coupling entries (AX)_j plus a 1 in its convexity row
    push!(colptr, 1)
    for j in cols
        append!(rowval, column_rows(pool, j))
        append!(nzval, column_vals(pool, j))
        push!(rowval, m.n_rows + pool.block[j])        # convexity row of the block
        push!(nzval, 1.0)
        push!(colptr, length(rowval) + 1)
    end
    n = length(cols)
    # λ_j ∈ [0, ∞) with cost c_j; Highs_addCols keeps the current basis for the warm start
    append_columns!(m.model, view(pool.cost, cols), zeros(n), fill(Inf, n), colptr, rowval, nzval)
    m.n_columns += n
    return m
end

function solve!(m::HighsLPMaster)
    sol = m.solution
    model = m.model
    sol.status = run!(model)
    sol.inner_iterations = iteration_counts(model).simplex
    sol.status === :optimal || return sol
    get_solution!(model, m.col_value, m.col_dual, m.row_value, m.row_dual)
    T, P, n = m.n_rows, m.n_blocks, m.n_columns
    # λ follows the 2T slack columns; π / μ are the duals of coupling / convexity rows
    copyto!(sol.weights, 1, m.col_value, 2T + 1, n)
    copyto!(sol.π, 1, m.row_dual, 1, T)
    copyto!(sol.μ, 1, m.row_dual, T + 1, P)
    sol.objective = model_objective(model)
    # 1ᵀ(s⁺ + s⁻): nonzero means the restricted master still relies on Big-M slacks
    m.artificial_activity = sum(view(m.col_value, 1:2T))
    return sol
end

solver_details(m::HighsLPMaster) =
    Pair{String,Any}["Big-M" => m.big_m, "final artificial activity" => m.artificial_activity]

function release!(m::HighsLPMaster)
    m.model === nothing || close(m.model)
    m.model = nothing
    return nothing
end
