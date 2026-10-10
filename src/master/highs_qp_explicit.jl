"""
    HighsExplicitQPMaster(; output = false)

Stabilized restricted master problem in the *explicit* form that keeps the auxiliary
variable `w` (thesis eq. 4.3, "explicit QP formulation"), solved by HiGHS' QP solver on a
persistent instance:

```
min  cᵀλ + δ/2‖w + π̂‖² - δ/2‖π̂‖²
s.t. row_lower ≤ AXλ + δw ≤ row_upper
     Cλ = 1,   λ ≥ 0,   w free
```

Columns are ordered `[w₁ … w_m, λ₁ … λ_n]`; the Hessian is `δI` on the `w` block only, so
it stays diagonal and tiny no matter how many columns are added. A center update only
changes the linear cost `δπ̂` of `w`. Inequality coupling rows are supported: the row dual
`π = π̂ + w` then carries the sign restriction automatically.
"""
mutable struct HighsExplicitQPMaster <: AbstractMasterSolver
    output::Bool
    model::Union{Nothing,HighsModel}
    delta::Float64
    n_rows::Int
    n_blocks::Int
    n_columns::Int
    center::Vector{Float64}
    solution::Union{Nothing,MasterSolution}
    # buffers
    col_value::Vector{Float64}
    col_dual::Vector{Float64}
    row_value::Vector{Float64}
    row_dual::Vector{Float64}
    w_cost::Vector{Float64}
    add_colptr::Vector{Int}
    add_rowval::Vector{Int}
    add_nzval::Vector{Float64}
    hess_colptr::Vector{Int}
    hess_rowval::Vector{Int}
    hess_nzval::Vector{Float64}
end

HighsExplicitQPMaster(; output::Bool = false) =
    HighsExplicitQPMaster(output, nothing, NaN, 0, 0, 0, Float64[], nothing,
                          Float64[], Float64[], Float64[], Float64[], Float64[],
                          Int[], Int[], Float64[], Int[], Int[], Float64[])

method_name(::HighsExplicitQPMaster) = "Explicit QP CG (HiGHS)"
is_stabilized(::HighsExplicitQPMaster) = true
supports_inequalities(::HighsExplicitQPMaster) = true

function initialize!(m::HighsExplicitQPMaster, spec::MasterSpec, pool::ColumnPool)
    isfinite(spec.delta) && spec.delta > 0 ||
        throw(ArgumentError("stabilization parameter δ must be positive"))
    release!(m)
    T, P = n_rows(spec), spec.n_blocks
    m.delta = spec.delta
    m.n_rows, m.n_blocks, m.n_columns = T, P, 0
    m.center = zeros(T)
    m.solution = MasterSolution(spec)
    model = HighsModel(; output = m.output)
    m.model = model

    append_rows!(model, spec.row_lower, spec.row_upper)
    append_rows!(model, ones(P), ones(P))

    # w columns: free, coefficient δ in their own coupling row, cost δπ̂ (π̂ = 0 initially)
    m.w_cost = zeros(T)
    append_columns!(model, m.w_cost, fill(-Inf, T), fill(Inf, T),
                    collect(1:(T + 1)), collect(1:T), fill(m.delta, T))

    resize!(m.col_value, T + spec.capacity)
    resize!(m.col_dual, T + spec.capacity)
    resize!(m.row_value, T + P)
    resize!(m.row_dual, T + P)

    # Hessian buffers: δ on the w diagonal, empty λ columns (extended in `_pass_hessian!`)
    m.hess_rowval = collect(1:T)
    m.hess_nzval = fill(m.delta, T)
    m.hess_colptr = collect(1:(T + 1))

    add_columns!(m, pool, 1:n_columns(pool))
    return m
end

function add_columns!(m::HighsExplicitQPMaster, pool::ColumnPool, cols::UnitRange{Int})
    isempty(cols) && return m
    first(cols) == m.n_columns + 1 || throw(ArgumentError("columns must be added in order"))
    colptr, rowval, nzval = m.add_colptr, m.add_rowval, m.add_nzval
    empty!(colptr)
    empty!(rowval)
    empty!(nzval)
    push!(colptr, 1)
    for j in cols
        append!(rowval, column_rows(pool, j))
        append!(nzval, column_vals(pool, j))
        push!(rowval, m.n_rows + pool.block[j])
        push!(nzval, 1.0)
        push!(colptr, length(rowval) + 1)
    end
    n = length(cols)
    append_columns!(m.model, view(pool.cost, cols), zeros(n), fill(Inf, n), colptr, rowval, nzval)
    m.n_columns += n
    _pass_hessian!(m)
    return m
end

# HiGHS requires the Hessian dimension to equal the number of columns, so it is re-passed
# after columns are added. Only the w-diagonal has entries: O(m) work.
function _pass_hessian!(m::HighsExplicitQPMaster)
    dim = m.n_rows + m.n_columns
    T = m.n_rows
    resize!(m.hess_colptr, dim + 1)
    @inbounds for k in (T + 2):(dim + 1)
        m.hess_colptr[k] = T + 1
    end
    pass_hessian!(m.model, dim, m.hess_colptr, m.hess_rowval, m.hess_nzval)
    return m
end

function set_center!(m::HighsExplicitQPMaster, center::AbstractVector{Float64})
    copyto!(m.center, center)
    m.w_cost .= m.delta .* m.center
    change_costs!(m.model, 1, m.w_cost)
    return m
end

function solve!(m::HighsExplicitQPMaster)
    sol = m.solution
    model = m.model
    sol.status = run!(model)
    its = iteration_counts(model)
    sol.inner_iterations = its.qp + its.simplex
    sol.status === :optimal || return sol
    get_solution!(model, m.col_value, m.col_dual, m.row_value, m.row_dual)
    T, P, n = m.n_rows, m.n_blocks, m.n_columns
    copyto!(sol.weights, 1, m.col_value, T + 1, n)
    @inbounds for i in 1:T
        sol.π[i] = m.center[i] + m.col_value[i]           # π = π̂ + w   (thesis eq. 4.4)
    end
    copyto!(sol.μ, 1, m.row_dual, T + 1, P)
    sol.objective = model_objective(model)                # = cᵀλ + δ/2‖w+π̂‖² - δ/2‖π̂‖²
    return sol
end

function release!(m::HighsExplicitQPMaster)
    m.model === nothing || close(m.model)
    m.model = nothing
    return nothing
end
