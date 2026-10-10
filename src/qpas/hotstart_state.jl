# State of the hot-started active-set solver (thesis §4.5).
#
# The QP solved at every column generation iteration is the eliminated regularized master
# problem (thesis eq. 4.6), extended by one slack sσ per inequality coupling row
# (docs/qpas_inequality_extension.md, eq. 2.1). With y = [x; s] and Ā = [A  K] (K places κσ in
# the row of slack σ):
#
#     min  1/(2δ) yᵀĀᵀĀy + c̃ᵀy,   c̃ = [c; 0] - Āᵀ(π̂ + b/δ),   s.t.  Cx = 1,  x ≥ 0,  s ≥ 0.
#
# Its gradient has the convenient form
#
#     g = [c; 0] - Āᵀπ,     π = π̂ + (b - Ax - Ks)/δ,
#
# i.e. g_j = c_j - a_jᵀπ is the reduced cost of column j without the convexity dual, the
# gradient of slack σ is g_σ = -κσ πᵢ (the dual of its row up to sign), and π is exactly the
# dual vector passed to pricing (thesis eq. 4.4). The solver therefore maintains the m-vector π
# instead of the gradient and evaluates g_j = c_j - a_jᵀπ only where it is needed:
#
#   - on N ∪ S to build the reduced right-hand side (O((|N| + p)·nnz)),
#   - on W only when checking multipliers at a subspace minimizer (O(nnz(A))).
#
# A primal step y += αp updates π -= (α/δ)·Āp with Āp = Σ_{k ∈ N ∪ S} pₖ aₖ + Σ_σ κσ p_σ e_row(σ).
#
# Between CG iterations everything below is kept: x, s, the ordering [S, W, N | W_s, N_s] and
# the LDLᵀ factor. A new column enters W with weight 0 (no factor change), and a new center
# only changes π (recomputed from scratch at the start of each solve).

"""
    QPASState(pool, rhs, delta; options = QPASOptions(), slack_rows = Int[], slack_signs = Float64[])

Everything the hot-started active-set solver keeps from one column generation iteration to
the next: the iterate `x` (column weights) and `s` (slacks of the inequality rows), the working
duals `π`, the convexity duals `μ`, the KKT system (ordering + factor), scratch buffers,
counters and timers.

`rhs` is `b`, the right-hand side of every coupling row written as an equality
`aᵢᵀx + κσ sσ = bᵢ`. Row `slack_rows[σ]` has a slack with coefficient `slack_signs[σ]`
(`-1` for a `≥` row, `+1` for a `≤` row); all other rows are equalities.

`pool` must hold at least one column per block; the first column of each block becomes its
representative with weight 1, all other columns start in the working set. All slacks start in
the working set with `sσ = 0`.
"""
mutable struct QPASState
    pool::ColumnPool
    options::QPASOptions
    delta::Float64
    rhs::Vector{Float64}
    center::Vector{Float64}
    x::Vector{Float64}
    s::Vector{Float64}         # slacks of the inequality rows
    π::Vector{Float64}
    μ::Vector{Float64}
    Ap::Vector{Float64}
    p::Vector{Float64}         # direction, column part (support N ∪ S)
    ps::Vector{Float64}        # direction, slack part (support N_s)
    πm::Vector{Float64}        # π with the rows of free slacks zeroed (the duals of A_eff)
    μm::Vector{Float64}        # convexity duals computed from πm
    kkt::KKTSystem
    n_registered::Int
    cost_scale::Float64        # max(1, maxⱼ |cⱼ|), scales the optimality tolerance
    last::QPASCounters
    total::QPASCounters
    timer::SectionTimer
end

function QPASState(pool::ColumnPool, rhs::AbstractVector{Float64}, delta::Real;
                   options::QPASOptions = QPASOptions(), slack_rows::Vector{Int} = Int[],
                   slack_signs::Vector{Float64} = Float64[])
    delta > 0 || throw(ArgumentError("δ must be positive"))
    m, P, cap = pool.n_rows, pool.n_blocks, pool.capacity
    length(rhs) == m || throw(DimensionMismatch("rhs must have length $m"))
    all(r -> 1 <= r <= m, slack_rows) || throw(ArgumentError("slack row out of range"))
    surplus = SurplusSet(slack_rows, slack_signs, m)
    mI = n_surplus(surplus)
    st = QPASState(pool, options, Float64(delta), collect(Float64, rhs), zeros(m),
                   zeros(cap), zeros(mI), zeros(m), zeros(P), zeros(m), zeros(cap), zeros(mI),
                   zeros(m), zeros(P), KKTSystem(P, m, cap, surplus; debug = options.debug), 0,
                   1.0, QPASCounters(), QPASCounters(), SectionTimer(QPAS_TIMER_SECTIONS))
    _initialize_ordering!(st)
    return st
end

function _initialize_ordering!(st::QPASState)
    pool = st.pool
    ord = st.kkt.ordering
    for j in 1:pool.n
        b = pool.block[j]
        if ord.rep[b] == 0
            # first column of a block: representative carrying the whole convexity weight
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

"Recompute `π = π̂ + (b - Ax - Ks)/δ` from scratch."
function refresh_duals!(st::QPASState)
    # Āy = Ax + Σσ κσ sσ e_row(σ)
    coupling_product!(st.Ap, st.pool, st.x)
    sp = st.kkt.surplus
    @inbounds for σ in eachindex(st.s)
        st.Ap[sp.row[σ]] += sp.sign[σ] * st.s[σ]
    end
    @. st.π = st.center + (st.rhs - st.Ap) / st.delta
    return st
end

"Objective gradient component `gⱼ = cⱼ - a_jᵀπ` of column `j`."
@inline gradient(st::QPASState, j::Integer) =
    @inbounds st.pool.cost[j] - column_dot(st.pool, j, st.π)

"Objective gradient component `g_σ = -κσ πᵢ` of slack `σ` (row `i`)."
@inline surplus_gradient(st::QPASState, σ::Integer) =
    @inbounds -st.kkt.surplus.sign[σ] * st.π[st.kkt.surplus.row[σ]]

"Convexity duals `μ_b = g_rep(b)` (thesis Theorem 4.3.5: μ = h_S, and h = g at p = 0)."
function update_convexity_duals!(st::QPASState)
    ord = st.kkt.ordering
    @inbounds for b in eachindex(ord.rep)
        st.μ[b] = gradient(st, ord.rep[b])
    end
    return st.μ
end

"""
    masked_duals!(state) -> (π_eff, μ_eff)

The duals entering the right-hand side of the effective reduced system
(docs/qpas_inequality_extension.md, eq. 3.6): `π` with the rows of free slacks set to zero,
and the convexity duals `μ_b = c_rep(b) - a_rep(b)ᵀπ_eff` computed from it. Without free slacks
this is `(π, μ)` itself.
"""
function masked_duals!(st::QPASState)
    sp = st.kkt.surplus
    any_free(sp) || return st.π, update_convexity_duals!(st)
    copyto!(st.πm, st.π)
    # rows absorbed by a free slack carry no dual pressure on the columns
    @inbounds for σ in items(sp.free)
        st.πm[sp.row[σ]] = 0.0
    end
    ord = st.kkt.ordering
    @inbounds for b in eachindex(ord.rep)
        r = ord.rep[b]
        st.μm[b] = st.pool.cost[r] - column_dot(st.pool, r, st.πm)
    end
    return st.πm, st.μm
end

"`cᵀx` over the registered columns."
cost_term(st::QPASState) = total_cost(st.pool, st.x)

"Stabilized master objective `cᵀx + δ/2‖π‖² - δ/2‖π̂‖²` at the current iterate."
master_objective(st::QPASState) = stabilized_objective(cost_term(st), st.π, st.center, st.delta)
