"""
    HighsEliminatedQPMaster(; output = false)

Stabilized restricted master problem in the *eliminated* form (thesis Thm. 3.3.1, Eliminated
regularized master problem, eq. 3.5 with c̃ from eq. 3.6), where
`w = (b - Āy)/δ` has been substituted, solved by HiGHS' QP solver on a persistent instance.
Every inequality coupling row gets a slack as in the QPAS (thesis §2.4, Rem. 2.4.4 and 2.4.5),
`aᵢᵀλ + κᵢ sᵢ = bᵢ` with `0 ≤ sᵢ ≤ Uᵢ` (`slack_structure`), so with `y = [s; λ]` and
`Ā = [K  AX]` (`K` places `κᵢ` in the row of slack `i`):

```
min  1/(2δ) yᵀĀᵀĀy + c̃ᵀy,   c̃ = [0; c] - Āᵀ(π̂ + b/δ)
s.t. Cλ = 1,   λ ≥ 0,   0 ≤ s ≤ U
```

This is the same QP the hot-started QPAS solves. HiGHS handles the slack bounds directly, so a
range row needs no bound flips here (`Uᵢ = uᵢ - ℓᵢ`; `Uᵢ = ∞` for one-sided rows).

The Gram matrix `G = ĀᵀĀ` is kept incrementally: the slack block is `KᵀK = I` (one slack per
row), a new column `j` adds `a_kᵀa_j` for the earlier columns and `κᵢ (a_j)ᵢ` for the slacks
(O(nnz(AX)) work). Because HiGHS has no Hessian update API, the full lower triangle of `G/δ`
is re-passed whenever columns were added, and all costs change when the center moves.
"""
mutable struct HighsEliminatedQPMaster <: AbstractMasterSolver
    output::Bool
    model::Union{Nothing,HighsModel}
    delta::Float64
    rhs::Vector{Float64}
    slack_rows::Vector{Int}       # coupling row of slack σ (HiGHS variable σ)
    slack_signs::Vector{Float64}  # κσ
    n_rows::Int
    n_blocks::Int
    n_columns::Int
    pool::Union{Nothing,ColumnPool}
    center::Vector{Float64}
    solution::Union{Nothing,MasterSolution}
    hessian_stale::Bool
    # lower triangle of the Gram matrix ĀᵀĀ over the variables y = [s; λ] (slack σ is
    # variable σ, column j is variable m_I + j), column-wise: rows ≥ column index, increasing
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
    HighsEliminatedQPMaster(output, nothing, NaN, Float64[], Int[], Float64[], 0, 0, 0, nothing,
                            Float64[], nothing,
                            true, Vector{Int}[], Vector{Float64}[], Float64[], Float64[],
                            Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
                            Int[], Int[], Float64[])

method_name(::HighsEliminatedQPMaster) = "Eliminated QP CG (HiGHS)"
is_stabilized(::HighsEliminatedQPMaster) = true
supports_inequalities(::HighsEliminatedQPMaster) = true

"Number of slack variables (they come first among the HiGHS variables)."
n_slacks(m::HighsEliminatedQPMaster) = length(m.slack_rows)

function initialize!(m::HighsEliminatedQPMaster, spec::MasterSpec, pool::ColumnPool)
    isfinite(spec.delta) && spec.delta > 0 ||
        throw(ArgumentError("stabilization parameter δ must be positive"))
    release!(m)
    T, P = n_rows(spec), spec.n_blocks
    m.delta = spec.delta
    # coupling rows as equalities aᵢᵀλ + κᵢsᵢ = bᵢ, 0 ≤ sᵢ ≤ Uᵢ (the same as in the QPAS)
    m.rhs, m.slack_rows, m.slack_signs, upper = slack_structure(spec.row_lower, spec.row_upper)
    mI = length(m.slack_rows)
    m.n_rows, m.n_blocks, m.n_columns = T, P, 0
    m.pool = pool
    m.center = zeros(T)
    m.solution = MasterSolution(spec)
    m.model = HighsModel(; output = m.output)
    # w is eliminated, so the only rows are the P convexity rows Cλ = 1
    append_rows!(m.model, ones(P), ones(P))
    # slack variables 0 ≤ sσ ≤ Uσ: in no constraint row, cost set in `solve!`
    append_columns!(m.model, zeros(mI), zeros(mI), upper, ones(Int, mI + 1), Int[], Float64[])

    # slack block of the Gram matrix: KᵀK = I (one slack per row, κσ² = 1)
    m.gram_rows = [[σ] for σ in 1:mI]
    m.gram_vals = [[1.0] for _ in 1:mI]
    m.scatter = zeros(T)
    m.shifted_dual = zeros(T)
    m.Ax = zeros(T)
    m.cost = zeros(mI + spec.capacity)
    m.col_value = zeros(mI + spec.capacity)
    m.col_dual = zeros(mI + spec.capacity)
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
        # λ_j only appears in its convexity row; cost c_j is overwritten by c̃_j in `solve!`
        rows = [pool.block[j]]
        append_columns!(m.model, [pool.cost[j]], [0.0], [Inf], [1, 2], rows, [1.0])
        m.n_columns += 1
    end
    m.hessian_stale = true
    return m
end

# New Gram entries for column j (variable v = m_I + j, the largest so far, so pushing keeps
# every column's row list sorted): G[v, σ] = κσ (a_j)_row(σ) for the slacks and
# G[v, m_I + k] = a_jᵀ a_k for the earlier columns k.
function _extend_gram!(m::HighsEliminatedQPMaster, pool::ColumnPool, j::Int)
    buf = m.scatter
    mI = n_slacks(m)
    v_j = mI + j
    # scatter a_j into a dense buffer so each a_kᵀa_j is a sparse–dense dot product
    column_axpy!(buf, 1.0, pool, j)
    # slack–column entries (Ā = [K  AX]): stored in the slack's column; exact zeros skipped
    @inbounds for σ in 1:mI
        a = buf[m.slack_rows[σ]]
        if a != 0.0
            push!(m.gram_rows[σ], v_j)
            push!(m.gram_vals[σ], m.slack_signs[σ] * a)
        end
    end
    # off-diagonal entries G[j, k], stored in column k (lower triangle); exact zeros skipped
    for k in 1:(j - 1)
        v = column_dot(pool, k, buf)
        if v != 0.0
            push!(m.gram_rows[mI + k], v_j)
            push!(m.gram_vals[mI + k], v)
        end
    end
    # new column j starts with its diagonal entry G[j, j] = ‖a_j‖²
    push!(m.gram_rows, [v_j])
    push!(m.gram_vals, [column_dot(pool, j, buf)])
    # reset only the touched entries so the scatter buffer is all zeros again
    @inbounds for i in column_rows(pool, j)
        buf[i] = 0.0
    end
    return nothing
end

function _pass_hessian!(m::HighsEliminatedQPMaster)
    n = n_slacks(m) + m.n_columns               # all variables y = [s; λ]
    colptr, rowval, nzval = m.hess_colptr, m.hess_rowval, m.hess_nzval
    empty!(rowval)
    empty!(nzval)
    resize!(colptr, n + 1)
    colptr[1] = 1
    inv_delta = 1.0 / m.delta
    # flatten the per-column Gram lists into CSC: Hessian Q = G/δ, lower triangle only
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
    # π̂ only enters through c̃, which is recomputed at the start of every `solve!`
    copyto!(m.center, center)
    return m
end

function solve!(m::HighsEliminatedQPMaster)
    sol = m.solution
    pool = m.pool
    n, T, δ = m.n_columns, m.n_rows, m.delta
    m.hessian_stale && _pass_hessian!(m)

    mI = n_slacks(m)
    # c̃ = [0; c] - Āᵀ(π̂ + b/δ): -κσ times the shifted dual of its row for a slack, one sparse
    # dot product with the shifted dual per column
    @. m.shifted_dual = m.center + m.rhs / δ
    @inbounds for σ in 1:mI
        m.cost[σ] = -m.slack_signs[σ] * m.shifted_dual[m.slack_rows[σ]]
    end
    @inbounds for j in 1:n
        m.cost[mI + j] = pool.cost[j] - column_dot(pool, j, m.shifted_dual)
    end
    change_costs!(m.model, 1, view(m.cost, 1:(mI + n)))

    sol.status = run!(m.model)
    its = iteration_counts(m.model)
    sol.inner_iterations = its.qp + its.simplex
    sol.status === :optimal || return sol
    get_solution!(m.model, m.col_value, m.col_dual, m.row_value, m.row_dual)

    λ = view(m.col_value, (mI + 1):(mI + n))
    copyto!(sol.weights, 1, m.col_value, mI + 1, n)
    # recover π = π̂ + w with w = (b - Āy)/δ = (b - AXλ - Ks)/δ from the primal y
    coupling_product!(m.Ax, pool, λ)
    @inbounds for σ in 1:mI
        m.Ax[m.slack_rows[σ]] += m.slack_signs[σ] * m.col_value[σ]
    end
    @. sol.π = m.center + (m.rhs - m.Ax) / δ              # thesis eq. 3.9 (dual estimate)
    copyto!(sol.μ, 1, m.row_dual, 1, m.n_blocks)
    # HiGHS' objective drops the constants of c̃, so the stabilized objective is rebuilt
    sol.objective = stabilized_objective(dot(view(pool.cost, 1:n), λ), sol.π, m.center, δ)
    return sol
end

solver_details(m::HighsEliminatedQPMaster) =
    Pair{String,Any}["Hessian nonzeros (lower)" => sum(length, m.gram_rows; init = 0),
                     "slack variables" => n_slacks(m)]

function release!(m::HighsEliminatedQPMaster)
    m.model === nothing || close(m.model)
    m.model = nothing
    return nothing
end
