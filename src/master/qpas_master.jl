"""
    QPASMaster(; kwargs...)

Stabilized restricted master problem in the eliminated form (thesis eq. 3.5, written as eq. 4.1),
solved by the hot-started nullspace active-set method of thesis Chapters 4–8 (Alg. 6.1,
Hot-Started Nullspace Active-Set Method). The [`QPASState`](@ref) (iterate,
working set, ordering and LDLᵀ factor of the reduced Hessian) is carried over between
column generation iterations; a new column only joins the working set. The factor backend is
`QPASOptions.factorization` (`:cholmod` or `:dense`).

Coupling rows may be equalities, `≥` rows or `≤` rows (one finite bound). Every inequality row
gets a slack variable, `aᵢᵀλ - sᵢ = ℓᵢ` or `aᵢᵀλ + sᵢ = uᵢ`, which makes the eliminated form
valid again (thesis §2.4, Surplus Variables and the Unified Formulation; Rem. 2.4.4 for `≤`
rows). Range rows with two different finite bounds
are not supported.

Keyword arguments are forwarded to [`QPASOptions`](@ref).
"""
mutable struct QPASMaster <: AbstractMasterSolver
    options::QPASOptions
    state::Union{Nothing,QPASState}
    solution::Union{Nothing,MasterSolution}
end

QPASMaster(; kwargs...) = QPASMaster(QPASOptions(; kwargs...))
QPASMaster(options::QPASOptions) = QPASMaster(validate(options), nothing, nothing)

method_name(m::QPASMaster) =
    m.options.factorization === :cholmod ? "Hot-start QPAS CG" : "Hot-start QPAS CG (dense)"
is_stabilized(::QPASMaster) = true
supports_inequalities(::QPASMaster) = true

"""
    slack_structure(row_lower, row_upper) -> (b, slack_rows, slack_signs)

Write every coupling row `ℓᵢ ≤ aᵢᵀλ ≤ uᵢ` as an equality `aᵢᵀλ + κ sᵢ = bᵢ`:

| row                  | `bᵢ` | slack                      |
|:---------------------|:-----|:---------------------------|
| `ℓᵢ = uᵢ`            | `ℓᵢ` | none                       |
| `uᵢ = +∞` (`≥` row)  | `ℓᵢ` | surplus, `κ = -1`          |
| `ℓᵢ = -∞` (`≤` row)  | `uᵢ` | slack, `κ = +1`            |

A `≤` row is the `≥` row `-aᵢᵀλ ≥ -uᵢ`; flipping the sign of its slack instead of the row
keeps the columns unchanged. Rows with two different finite bounds, or none, are rejected.
"""
function slack_structure(lo::AbstractVector{Float64}, hi::AbstractVector{Float64})
    m = length(lo)
    b = zeros(m)
    rows = Int[]
    signs = Float64[]
    for i in 1:m
        if lo[i] == hi[i]
            b[i] = lo[i]                               # equality row
        elseif isfinite(lo[i]) && hi[i] == Inf
            b[i] = lo[i]                               # aᵢᵀλ - sᵢ = ℓᵢ
            push!(rows, i)
            push!(signs, -1.0)
        elseif lo[i] == -Inf && isfinite(hi[i])
            b[i] = hi[i]                               # aᵢᵀλ + sᵢ = uᵢ
            push!(rows, i)
            push!(signs, 1.0)
        else
            throw(ArgumentError("QPAS master: coupling row $i has bounds [$(lo[i]), $(hi[i])]; " *
                                "only equalities and one-sided inequalities are supported"))
        end
    end
    return b, rows, signs
end

function initialize!(m::QPASMaster, spec::MasterSpec, pool::ColumnPool)
    isfinite(spec.delta) && spec.delta > 0 ||
        throw(ArgumentError("stabilization parameter δ must be positive"))
    release!(m)
    b, rows, signs = slack_structure(spec.row_lower, spec.row_upper)
    m.state = QPASState(pool, b, spec.delta; options = m.options, slack_rows = rows,
                        slack_signs = signs)
    m.solution = MasterSolution(spec)
    return m
end

function add_columns!(m::QPASMaster, ::ColumnPool, ::UnitRange{Int})
    # hot start: new columns join the working set at weight 0, the factor is untouched
    register_columns!(m.state)
    return m
end

set_center!(m::QPASMaster, center::AbstractVector{Float64}) = (set_center!(m.state, center); m)

function solve!(m::QPASMaster)
    st = m.state
    sol = m.solution
    sol.status = qpas_solve!(st)
    sol.inner_iterations = st.last.iterations
    n = st.pool.n
    # the slacks are internal to the QPAS; the driver only sees λ, π and μ
    copyto!(sol.weights, 1, st.x, 1, n)
    copyto!(sol.π, st.π)
    copyto!(sol.μ, st.μ)
    sol.objective = master_objective(st)
    return sol
end

diagnostics(m::QPASMaster; residual::Bool = false) =
    m.state === nothing ? nothing : qpas_diagnostics(m.state; residual)

function solver_details(m::QPASMaster)
    st = m.state
    st === nothing && return Pair{String,Any}[]
    c = st.total
    t = st.timer
    details = Pair{String,Any}[
        "active-set iterations" => c.iterations,
        "primal steps (full)" => "$(c.steps) ($(c.full_steps))",
        "working-set expansions" => c.expansions,
        "  of which representative swaps" => c.representative_swaps,
        "working-set reductions" => c.reductions,
        "singular-Hessian steps" => c.singular_steps,
        "factorization backend" => LDLt.backend_name(st.kkt.factor),
        "factor updates (row add/del, rank-one)" => c.factor_updates,
        "refactorizations" => c.refactorizations,
        "final |N_x| (dim. of H_eff)" => nullspace_dimension(st.kkt.ordering),
    ]
    # contents of the final factor: stored pattern vs numerically nonzero entries
    dg = qpas_diagnostics(st)
    push!(details, "factor nnz stored / effective (off-diag.)" =>
                   "$(dg.factor_stored) / $(dg.factor_effective)")
    push!(details, "  dense factor of H_eff would have" => dg.factor_dense)
    push!(details, "factor memory now / peak [MB]" =>
                   "$(round(dg.factor_MB; digits = 2)) / $(round(dg.factor_peak_MB; digits = 2))")
    for (k, v) in pairs(LDLt.backend_stats(st.kkt.factor))
        push!(details, "  $(k)" => v isa AbstractFloat ? round(v; sigdigits = 4) : v)
    end
    sp = st.kkt.surplus
    if n_surplus(sp) > 0
        # inequality rows: slack activity and the size of A_eff
        push!(details, "inequality rows (slacks)" => n_surplus(sp))
        push!(details, "  slack expansions / reductions" =>
                       "$(c.slack_expansions) / $(c.slack_reductions)")
        push!(details, "  final free slacks |N_s|" => length(sp.free))
    end
    for (k, name) in enumerate(t.names)
        label = name === :refactor ? "time refactor [s] (within expand/reduce)" : "time $(name) [s]"
        push!(details, label => round(t.seconds[k]; sigdigits = 4))
    end
    # individual factor operations (overlapping the sections above)
    ft = st.kkt.timer
    for (k, name) in enumerate(ft.names)
        ft.calls[k] == 0 && continue
        push!(details, "factor op $(name) [s] (calls, µs/call)" =>
                       "$(round(ft.seconds[k]; sigdigits = 4)) ($(ft.calls[k]), " *
                       "$(round(1e6 * ft.seconds[k] / ft.calls[k]; sigdigits = 3)))")
    end
    return details
end

function release!(m::QPASMaster)
    if m.state !== nothing
        close(m.state.kkt.factor)
        m.state = nothing
    end
    return nothing
end
