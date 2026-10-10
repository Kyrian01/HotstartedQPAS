"""
    HighsEliminatedQPMaster(; output = false)

Stabilized restricted master problem in the *eliminated* form (thesis eq. 4.5/4.6), where
`w = (b - AXλ)/δ` has been substituted, solved by HiGHS' QP solver on a persistent instance:

```
min  1/(2δ) λᵀ(AX)ᵀ(AX)λ + c̃ᵀλ,   c̃ = c - (AX)ᵀ(π̂ + b/δ)
s.t. Cλ = 1,   λ ≥ 0
```

This is the same QP the hot-started QPAS solves. The Gram matrix `G = (AX)ᵀ(AX)` is kept
incrementally (one new row/column per added column, O(nnz(AX)) work); because HiGHS has
no Hessian update API, the full lower triangle of `G/δ` is re-passed whenever columns were
added, and all costs change when the center moves.

Only equality coupling rows are supported (eliminating `w` relies on `AXλ + δw = b`).
"""
mutable struct HighsEliminatedQPMaster <: AbstractMasterSolver
    output::Bool
    model::Union{Nothing,HighsModel}
    delta::Float64
    rhs::Vector{Float64}
    n_rows::Int
    n_blocks::Int
    n_columns::Int
    pool::Union{Nothing,ColumnPool}
    center::Vector{Float64}
    solution::Union{Nothing,MasterSolution}
    hessian_stale::Bool
    # lower triangle of the Gram matrix, column-wise: rows ≥ column index, increasing
    gram_rows::Vector{Vector{Int}}
    gram_vals::Vector{Vector{Float64}}
    # buffers
    scatter::Vector{Float64}
    shifted_dual::Vector{Float64}
    cost::Vector{Float64}
    Ax::Vector{Float64}
    col_value::Vector{Float64}
    col_dual::Vector{Float64}
    row_value::Vector{Float64}
    row_dual::Vector{Float64}
    hess_colptr::Vector{Int}
    hess_rowval::Vector{Int}
    hess_nzval::Vector{Float64}
end

HighsEliminatedQPMaster(; output::Bool = false) =
    HighsEliminatedQPMaster(output, nothing, NaN, Float64[], 0, 0, 0, nothing, Float64[], nothing,
                            true, Vector{Int}[], Vector{Float64}[], Float64[], Float64[],
                            Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
                            Int[], Int[], Float64[])

method_name(::HighsEliminatedQPMaster) = "Eliminated QP CG (HiGHS)"
is_stabilized(::HighsEliminatedQPMaster) = true
supports_inequalities(::HighsEliminatedQPMaster) = false

function initialize!(m::HighsEliminatedQPMaster, spec::MasterSpec, pool::ColumnPool)
    equality_only(spec) ||
        throw(ArgumentError("$(method_name(m)) supports equality coupling rows only"))
    isfinite(spec.delta) && spec.delta > 0 ||
        throw(ArgumentError("stabilization parameter δ must be positive"))
    release!(m)
    T, P = n_rows(spec), spec.n_blocks
    m.delta = spec.delta
    m.rhs = copy(spec.row_lower)
    m.n_rows, m.n_blocks, m.n_columns = T, P, 0
    m.pool = pool
    m.center = zeros(T)
    m.solution = MasterSolution(spec)
    m.model = HighsModel(; output = m.output)
    append_rows!(m.model, ones(P), ones(P))

    m.gram_rows = Vector{Int}[]
    m.gram_vals = Vector{Float64}[]
    m.scatter = zeros(T)
    m.shifted_dual = zeros(T)
    m.Ax = zeros(T)
    m.cost = zeros(spec.capacity)
    m.col_value = zeros(spec.capacity)
    m.col_dual = zeros(spec.capacity)
    m.row_value = zeros(P)
    m.row_dual = zeros(P)

    add_columns!(m, pool, 1:n_columns(pool))
    return m
end

function add_columns!(m::HighsEliminatedQPMaster, pool::ColumnPool, cols::UnitRange{Int})
    isempty(cols) && return m
    first(cols) == m.n_columns + 1 || throw(ArgumentError("columns must be added in order"))
    for j in cols
        _extend_gram!(m, pool, j)
        rows = [pool.block[j]]
        append_columns!(m.model, [pool.cost[j]], [0.0], [Inf], [1, 2], rows, [1.0])
        m.n_columns += 1
    end
    m.hessian_stale = true
    return m
end

# New Gram entries G[j, k] = a_jᵀ a_k for k ≤ j (row j is the largest so far, so pushing keeps
# every column's row list sorted).
function _extend_gram!(m::HighsEliminatedQPMaster, pool::ColumnPool, j::Int)
    buf = m.scatter
    column_axpy!(buf, 1.0, pool, j)
    for k in 1:(j - 1)
        v = column_dot(pool, k, buf)
        if v != 0.0
            push!(m.gram_rows[k], j)
            push!(m.gram_vals[k], v)
        end
    end
    push!(m.gram_rows, [j])
    push!(m.gram_vals, [column_dot(pool, j, buf)])
    @inbounds for i in column_rows(pool, j)
        buf[i] = 0.0
    end
    return nothing
end

function _pass_hessian!(m::HighsEliminatedQPMaster)
    n = m.n_columns
    colptr, rowval, nzval = m.hess_colptr, m.hess_rowval, m.hess_nzval
    empty!(rowval)
    empty!(nzval)
    resize!(colptr, n + 1)
    colptr[1] = 1
    inv_delta = 1.0 / m.delta
    for k in 1:n
        append!(rowval, m.gram_rows[k])
        for v in m.gram_vals[k]
            push!(nzval, v * inv_delta)
        end
        colptr[k + 1] = length(rowval) + 1
    end
    pass_hessian!(m.model, n, colptr, rowval, nzval)
    m.hessian_stale = false
    return m
end

function set_center!(m::HighsEliminatedQPMaster, center::AbstractVector{Float64})
    copyto!(m.center, center)
    return m
end

function solve!(m::HighsEliminatedQPMaster)
    sol = m.solution
    pool = m.pool
    n, T, δ = m.n_columns, m.n_rows, m.delta
    m.hessian_stale && _pass_hessian!(m)

    # c̃ = c - (AX)ᵀ(π̂ + b/δ)
    @. m.shifted_dual = m.center + m.rhs / δ
    @inbounds for j in 1:n
        m.cost[j] = pool.cost[j] - column_dot(pool, j, m.shifted_dual)
    end
    change_costs!(m.model, 1, view(m.cost, 1:n))

    sol.status = run!(m.model)
    its = iteration_counts(m.model)
    sol.inner_iterations = its.qp + its.simplex
    sol.status === :optimal || return sol
    get_solution!(m.model, m.col_value, m.col_dual, m.row_value, m.row_dual)

    λ = view(m.col_value, 1:n)
    copyto!(sol.weights, 1, m.col_value, 1, n)
    coupling_product!(m.Ax, pool, m.col_value)
    @. sol.π = m.center + (m.rhs - m.Ax) / δ              # thesis eq. 4.4
    copyto!(sol.μ, 1, m.row_dual, 1, m.n_blocks)
    sol.objective = stabilized_objective(dot(view(pool.cost, 1:n), λ), sol.π, m.center, δ)
    return sol
end

solver_details(m::HighsEliminatedQPMaster) =
    Pair{String,Any}["Hessian nonzeros (lower)" => sum(length, m.gram_rows; init = 0)]

function release!(m::HighsEliminatedQPMaster)
    m.model === nothing || close(m.model)
    m.model = nothing
    return nothing
end
