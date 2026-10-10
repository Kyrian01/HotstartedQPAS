"""
    QPASMaster(; kwargs...)

Stabilized restricted master problem in the eliminated form (thesis eq. 4.6), solved by the
hot-started nullspace active-set method of thesis Chapter 4. The [`QPASState`](@ref) (iterate,
working set, ordering and CHOLMOD LDLᵀ factor of the reduced Hessian) is carried over between
column generation iterations; a new column only joins the working set.

Keyword arguments are forwarded to [`QPASOptions`](@ref).

Only equality coupling rows are supported for now; see the README for the planned extension
to inequality rows.
"""
mutable struct QPASMaster <: AbstractMasterSolver
    options::QPASOptions
    state::Union{Nothing,QPASState}
    solution::Union{Nothing,MasterSolution}
end

QPASMaster(; kwargs...) = QPASMaster(QPASOptions(; kwargs...))
QPASMaster(options::QPASOptions) = QPASMaster(options, nothing, nothing)

method_name(::QPASMaster) = "Hot-start QPAS CG"
is_stabilized(::QPASMaster) = true
supports_inequalities(::QPASMaster) = false

function initialize!(m::QPASMaster, spec::MasterSpec, pool::ColumnPool)
    equality_only(spec) ||
        throw(ArgumentError("$(method_name(m)) supports equality coupling rows only (for now)"))
    isfinite(spec.delta) && spec.delta > 0 ||
        throw(ArgumentError("stabilization parameter δ must be positive"))
    release!(m)
    m.state = QPASState(pool, spec.row_lower, spec.delta; options = m.options)
    m.solution = MasterSolution(spec)
    return m
end

function add_columns!(m::QPASMaster, ::ColumnPool, ::UnitRange{Int})
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
    copyto!(sol.weights, 1, st.x, 1, n)
    copyto!(sol.π, st.π)
    copyto!(sol.μ, st.μ)
    sol.objective = master_objective(st)
    return sol
end

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
        "CHOLMOD row updates" => c.factor_updates,
        "refactorizations" => c.refactorizations,
        "final |N| (nullspace dim.)" => nullspace_dimension(st.kkt.ordering),
    ]
    for (k, name) in enumerate(t.names)
        label = name === :refactor ? "time refactor [s] (within expand/reduce)" : "time $(name) [s]"
        push!(details, label => round(t.seconds[k]; sigdigits = 4))
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
