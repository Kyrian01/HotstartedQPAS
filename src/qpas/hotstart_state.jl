# State of the hot-started active-set solver (thesis §4.5).
#
# The QP solved at every column generation iteration is the eliminated regularized master
# problem (thesis eq. 4.6):
#
#     min  1/(2δ) xᵀAᵀAx + c̃ᵀx,   c̃ = c - Aᵀ(π̂ + b/δ),   s.t.  Cx = 1,  x ≥ 0.
#
# Its gradient has the convenient form
#
#     g = (1/δ)AᵀAx + c̃ = c - Aᵀπ,     π = π̂ + (b - Ax)/δ,
#
# i.e. gⱼ is the reduced cost of column j without the convexity dual, and π is exactly the
# dual vector passed to pricing (thesis eq. 4.4). The solver therefore maintains the m-vector π
# instead of the n-vector g, and evaluates gⱼ = cⱼ - a_jᵀπ only where it is needed:
#
#   - on N ∪ S to build the reduced right-hand side (O((|N| + p)·nnz)),
#   - on W only when checking multipliers at a subspace minimizer (O(nnz(A))).
#
# A primal step x += αp updates π -= (α/δ)·Ap with Ap = Σ_{k ∈ N ∪ S} pₖ aₖ.
#
# Between CG iterations everything below is kept: x, the ordering [S, W, N] and the LDLᵀ
# factor. A new column enters W with weight 0 (no factor change), and a new center only
# changes π (recomputed from scratch at the start of each solve).

"""
    QPASState(pool, rhs, delta; options = QPASOptions())

Everything the hot-started active-set solver keeps from one column generation iteration to
the next: the iterate `x`, the working duals `π`, the convexity duals `μ`, the KKT system
(ordering + reduced-Hessian factor), scratch buffers, counters and timers.

`pool` must hold at least one column per block; the first column of each block becomes its
representative with weight 1, all other columns start in the working set.
"""
mutable struct QPASState
    pool::ColumnPool
    options::QPASOptions
    delta::Float64
    rhs::Vector{Float64}
    center::Vector{Float64}
    x::Vector{Float64}
    π::Vector{Float64}
    μ::Vector{Float64}
    Ap::Vector{Float64}
    p::Vector{Float64}
    kkt::KKTSystem
    n_registered::Int
    cost_scale::Float64        # max(1, maxⱼ |cⱼ|), scales the optimality tolerance
    last::QPASCounters
    total::QPASCounters
    timer::SectionTimer
end

function QPASState(pool::ColumnPool, rhs::AbstractVector{Float64}, delta::Real;
                   options::QPASOptions = QPASOptions())
    delta > 0 || throw(ArgumentError("δ must be positive"))
    m, P, cap = pool.n_rows, pool.n_blocks, pool.capacity
    length(rhs) == m || throw(DimensionMismatch("rhs must have length $m"))
    st = QPASState(pool, options, Float64(delta), collect(Float64, rhs), zeros(m),
                   zeros(cap), zeros(m), zeros(P), zeros(m), zeros(cap),
                   KKTSystem(P, m, cap; debug = options.debug), 0, 1.0,
                   QPASCounters(), QPASCounters(), SectionTimer(QPAS_TIMER_SECTIONS))
    _initialize_ordering!(st)
    return st
end

function _initialize_ordering!(st::QPASState)
    pool = st.pool
    ord = st.kkt.ordering
    for j in 1:pool.n
        b = pool.block[j]
        if ord.rep[b] == 0
            make_representative!(ord, pool.block, j)
            st.x[j] = 1.0
        else
            move_to_W!(ord, j)
        end
        st.cost_scale = max(st.cost_scale, abs(pool.cost[j]))
    end
    missing_blocks = findall(==(0), ord.rep)
    isempty(missing_blocks) ||
        throw(ArgumentError("every block needs an initial column; missing: $missing_blocks"))
    st.n_registered = pool.n
    return st
end

"""
    register_columns!(state)

Hot start (thesis §4.5.2): columns appended to the pool since the last call enter the working
set with weight 0. Neither the nullspace basis nor the factor changes.
"""
function register_columns!(st::QPASState)
    ord = st.kkt.ordering
    for j in (st.n_registered + 1):st.pool.n
        move_to_W!(ord, j)
        st.x[j] = 0.0
        st.cost_scale = max(st.cost_scale, abs(st.pool.cost[j]))
    end
    st.n_registered = st.pool.n
    return st
end

"Set the stability center π̂ used by the next solve."
set_center!(st::QPASState, center::AbstractVector{Float64}) = (copyto!(st.center, center); st)

"Recompute `π = π̂ + (b - Ax)/δ` from scratch."
function refresh_duals!(st::QPASState)
    coupling_product!(st.Ap, st.pool, st.x)
    @. st.π = st.center + (st.rhs - st.Ap) / st.delta
    return st
end

"Objective gradient component `gⱼ = cⱼ - a_jᵀπ`."
@inline gradient(st::QPASState, j::Integer) =
    @inbounds st.pool.cost[j] - column_dot(st.pool, j, st.π)

"Convexity duals `μ_b = g_rep(b)` (thesis Theorem 4.3.5: μ = h_S, and h = g at p = 0)."
function update_convexity_duals!(st::QPASState)
    ord = st.kkt.ordering
    @inbounds for b in eachindex(ord.rep)
        st.μ[b] = gradient(st, ord.rep[b])
    end
    return st.μ
end

"`cᵀx` over the registered columns."
cost_term(st::QPASState) = total_cost(st.pool, st.x)

"Stabilized master objective `cᵀx + δ/2‖π‖² - δ/2‖π̂‖²` at the current iterate."
master_objective(st::QPASState) = stabilized_objective(cost_term(st), st.π, st.center, st.delta)
