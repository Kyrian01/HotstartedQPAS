# State of the hot-started active-set solver (thesis §6.6, Hot Start Between Column Generation
# Iterations; Alg. 6.1, Hot-Started Nullspace Active-Set Method).
#
# The QP solved at every column generation iteration is the eliminated regularized master
# problem (thesis eq. 3.5, written as eq. 4.1 in §4.1), with one slack sσ per inequality
# coupling row (thesis §2.4, Surplus Variables and the Unified Formulation). With y = [x; s] and Ā = [A  K] (K places κσ in
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
# dual vector passed to pricing (thesis Def. 3.3.2, Dual estimate, eq. 3.9). The solver
# therefore maintains the m-vector π instead of the gradient and evaluates g_j = c_j - a_jᵀπ only where it is needed:
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
    QPASState(pool, rhs, delta; options = QPASOptions(), slack_rows = Int[], slack_signs = Float64[],
              slack_upper = fill(Inf, length(slack_rows)))

Everything the hot-started active-set solver keeps from one column generation iteration to
the next: the iterate `x` (column weights) and `s` (slacks of the inequality rows), the working
duals `π`, the convexity duals `μ`, the KKT system (ordering + factor), scratch buffers,
counters and timers.

`rhs` is `b`, the right-hand side of every coupling row written as an equality
`aᵢᵀx + κσ sσ = bᵢ`. Row `slack_rows[σ]` has a slack `0 ≤ sσ ≤ slack_upper[σ]` with
coefficient `slack_signs[σ]` (`-1` for a `≥` row, `+1` for a `≤` row; the upper bound is finite
only for range rows); all other rows are equalities. `rhs` and the signs of range rows change
when their slack flips its bound (`flip_bound!`).

`pool` must hold at least one column per block; the first column of each block becomes its
representative with weight 1, all other columns start in the working set. All slacks start in
the working set with `sσ = 0`.
"""
mutable struct QPASState{F<:UpdatableLDLt}
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
    kkt::KKTSystem{F}
    n_registered::Int
    cost_scale::Float64        # max(1, maxⱼ |cⱼ|), scales the optimality tolerance
    last::QPASCounters
    total::QPASCounters
    timer::SectionTimer
end

function QPASState(pool::ColumnPool, rhs::AbstractVector{Float64}, delta::Real;
                   options::QPASOptions = QPASOptions(), slack_rows::Vector{Int} = Int[],
                   slack_signs::Vector{Float64} = Float64[],
                   slack_upper::Vector{Float64} = fill(Inf, length(slack_rows)))
    delta > 0 || throw(ArgumentError("δ must be positive"))
    validate(options)
    m, P, cap = pool.n_rows, pool.n_blocks, pool.capacity
    length(rhs) == m || throw(DimensionMismatch("rhs must have length $m"))
    all(r -> 1 <= r <= m, slack_rows) || throw(ArgumentError("slack row out of range"))
    surplus = SurplusSet(slack_rows, slack_signs, m; upper = slack_upper)
    mI = n_surplus(surplus)
    st = QPASState(pool, options, Float64(delta), collect(Float64, rhs), zeros(m),
                   zeros(cap), zeros(mI), zeros(m), zeros(P), zeros(m), zeros(cap), zeros(mI),
                   zeros(m), zeros(P),
                   KKTSystem(P, m, cap, surplus; debug = options.debug,
                             factorization = options.factorization), 0,
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

Hot start (thesis Thm. 6.6.1, Hot start, case (i)): columns appended to the pool since the
last call enter the working set with weight 0. Neither the nullspace basis nor the factor changes.
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

"""
    set_delta!(state, δ)

Set the stabilization parameter of the next solve. The ordering and the factor of
`H_eff = (A_eff Z)ᵀ(A_eff Z)` do not depend on δ (thesis Rem. 5.2.8); δ only scales the
right-hand side `-δ Zᵀg` and the dual estimate `π = π̂ + (b - Āy)/δ`, which the next solve
recomputes. The iterate stays feasible.
"""
function set_delta!(st::QPASState, δ::Real)
    δ > 0 && isfinite(δ) || throw(ArgumentError("δ must be positive and finite"))
    st.delta = Float64(δ)
    return st
end

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

"""
    flip_bound!(state, σ)

Range-row slack `σ` (free, at its upper bound `Uσ`): substitute `s̄σ = Uσ - sσ`, i.e.

    aᵢᵀx + κσ sσ = bᵢ   ⟺   aᵢᵀx - κσ s̄σ = bᵢ - κσ Uσ,

so `κσ ← -κσ`, `bᵢ ← bᵢ - κσ Uσ` and `sσ ← Uσ - sσ` (≈ 0). The row switches between its `≥`
form (`bᵢ = ℓᵢ`) and its `≤` form (`bᵢ = uᵢ`). `Āy - b`, and with it `π`, the objective and the
row's contribution to `H_eff`, are unchanged; the caller then fixes `s̄σ = 0` like any slack
that reaches zero.
"""
function flip_bound!(st::QPASState, σ::Integer)
    sp = st.kkt.surplus
    U = sp.upper[σ]
    isfinite(U) || throw(ArgumentError("slack $σ has no upper bound"))
    κ = sp.sign[σ]
    i = sp.row[σ]
    st.rhs[i] -= κ * U
    sp.sign[σ] = -κ
    st.s[σ] = U - st.s[σ]
    return st
end

"Objective gradient component `gⱼ = cⱼ - a_jᵀπ` of column `j`."
@inline gradient(st::QPASState, j::Integer) =
    @inbounds st.pool.cost[j] - column_dot(st.pool, j, st.π)

"Objective gradient component `g_σ = -κσ πᵢ` of slack `σ` (row `i`)."
@inline surplus_gradient(st::QPASState, σ::Integer) =
    @inbounds -st.kkt.surplus.sign[σ] * st.π[st.kkt.surplus.row[σ]]

"""
Convexity duals `μ_b = g_rep(b)` (thesis Thm. 5.3.1, Lagrange multipliers: `μ = h_S`, and
`h = g` at `p = 0`; Cor. 5.3.2, Multipliers at a subspace minimizer).
"""
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
(thesis Cor. 5.2.3, Effective linking matrix, eq. 5.10): `π` with the rows of free slacks set
to zero, and the convexity duals `μ_b = c_rep(b) - a_rep(b)ᵀπ_eff` computed from it. Without
free slacks this is `(π, μ)` itself.
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

"""
    qpas_diagnostics(state) -> NamedTuple

Snapshot of the solver's internal size, for traces (`CGOptions(diagnostics_every = k)`):

| field | meaning |
|:------|:--------|
| `N`, `W`, `N_s` | sizes of the nullspace set, the working set and the free-slack set |
| `factor_stored` | off-diagonal entries stored in `L` (incl. zeros left by `rowdel`/`updown`) |
| `factor_effective` | stored off-diagonal entries that are numerically nonzero |
| `factor_dense` | `|N|(|N|-1)/2`, the off-diagonal entries of a dense factor of `H_eff` |
| `factor_fill` | `effective / dense`: how dense the true factor of `H_eff` is |
| `factor_waste` | `(stored - effective) / stored`: share of stored entries that are zeros |
| `factor_allocated` | entries reserved for `L` (CHOLMOD `nzmax`; dense: slots²) |
| `reach` | rows touched by the last sparse solve (`|reach_L(N)|`, ≥ `|N|`) |
| `updates_since_refactor` | row updates since the last rebuild (structural zeros accumulate) |
| `factor_MB`, `factor_peak_MB` | memory held by the factor backend now / at its peak |
| `factor_residual` | `max|(LDLᵀ)[N,N] - H_eff| / max(1, max|H_eff|)`, `H_eff` rebuilt from the pool (only with `residual = true`) |

The time of every factor operation so far is included as `t_<op>` (seconds) and `n_<op>`
(calls) for the sections in `FACTOR_TIMER_SECTIONS`.
"""
function qpas_diagnostics(st::QPASState; residual::Bool = false)
    kkt = st.kkt
    Nset = items(kkt.ordering.N)
    fs = LDLt.factor_stats(kkt.factor, Nset)
    nN = length(Nset)
    dense = nN * (nN - 1) ÷ 2
    base = (N = nN, W = length(kkt.ordering.W), N_s = length(kkt.surplus.free),
            factor_stored = fs.stored, factor_effective = fs.effective, factor_dense = dense,
            factor_fill = dense == 0 ? 0.0 : fs.effective / dense,
            factor_waste = fs.stored == 0 ? 0.0 : (fs.stored - fs.effective) / fs.stored,
            factor_allocated = fs.allocated, reach = LDLt.last_reach(kkt.factor),
            updates_since_refactor = kkt.updates_since_refactor,
            factor_MB = fs.memory / 2^20, factor_peak_MB = fs.memory_peak / 2^20)
    residual && (base = merge(base, (factor_residual = factor_residual(kkt, st.pool),)))
    t = kkt.timer
    ops = NamedTuple{Tuple(Symbol("t_", n) for n in t.names)}(Tuple(t.seconds))
    calls = NamedTuple{Tuple(Symbol("n_", n) for n in t.names)}(Tuple(t.calls))
    return merge(base, ops, calls)
end

"`cᵀx` over the registered columns."
cost_term(st::QPASState) = total_cost(st.pool, st.x)

"Stabilized master objective `cᵀx + δ/2‖π‖² - δ/2‖π̂‖²` at the current iterate."
master_objective(st::QPASState) = stabilized_objective(cost_term(st), st.π, st.center, st.delta)
