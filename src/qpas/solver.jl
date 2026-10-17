# Active-set method with efficient working-set updates, including inequality coupling rows
# (thesis Alg. 6.1, Hot-Started Nullspace Active-Set Method; one iteration is Alg. 5.1, the
# singular case Alg. 7.1).
#
#   repeat
#       if not at a subspace minimizer:
#           solve H_eff y = -δ Zᵀg̃, p_x = Zy, p_s from eq. 5.8 / Rem. 5.3.3   (compute_direction!)
#       if ‖p‖ ≈ 0 (subspace minimizer):
#           λⱼ = gⱼ - μ_ψ(j) on W_x, λσ = -κσ πᵢ on W_s;
#           if min λ ≥ -tol → optimal                               (most_negative_multiplier)
#           else release the most negative one from W               (reduce_working_set!,
#                                                                     reduce_surplus!)
#       else:
#           ratio test, y += αp, π -= (α/δ)Āp                      (ratio_test, take_step!)
#           if blocked: add the blocking index to W                 (expand_working_set!,
#                                                                     expand_surplus!)
#
# After a full step (α = 1) the iterate is the minimizer on the current subspace, so the next
# direction is zero and is not computed; the multipliers are checked directly.

"""
    qpas_solve!(state) -> Symbol

Solve the regularized master problem for the current columns and center, starting from the
iterate, ordering and factorization left by the previous solve. Returns `:optimal` or
`:iteration_limit`. Event counts of this solve are in `state.last`.
"""
function qpas_solve!(st::QPASState)
    opts = st.options
    ctr = st.last
    empty!(ctr)
    ctr.solves = 1
    timer = st.timer
    kkt = st.kkt

    updates_at_start = kkt.total_updates
    # new center / new columns: π = π̂ + (b - Āy)/δ from scratch
    refresh_duals!(st)
    status = :iteration_limit
    at_minimizer = false
    for _ in 1:opts.max_iterations
        ctr.iterations += 1
        if !at_minimizer
            pnorm = @timed_section timer :direction compute_direction!(st)
            if pnorm <= opts.step_tolerance
                clear_direction!(st)
                at_minimizer = true
            end
        end

        if at_minimizer
            # subspace minimizer: KKT multipliers of the working set (thesis Cor. 5.3.2)
            λmin, r, is_slack = @timed_section timer :multipliers most_negative_multiplier(st)
            if r == 0 || λmin >= -opts.dual_tolerance * st.cost_scale
                status = :optimal
                break
            end
            # release the most violated working-set constraint
            if is_slack
                @timed_section timer :reduce reduce_surplus!(st, r)
            else
                @timed_section timer :reduce reduce_working_set!(st, r)
            end
            at_minimizer = false
        else
            α, q, q_slack, q_upper = @timed_section timer :step ratio_test(st, 1.0, 0, 0)
            @timed_section timer :step take_step!(st, α, 0, 0)
            clear_direction!(st)
            ctr.steps += 1
            if q == 0
                # α = 1: minimizer on the current subspace (thesis Prop. 5.2.6, Full step)
                ctr.full_steps += 1
                at_minimizer = true
            elseif q_slack
                @timed_section timer :expand expand_slack!(st, q, q_upper)
            else
                @timed_section timer :expand expand_working_set!(st, q)
            end
        end
        opts.debug && verify_kkt(kkt, st.pool)
    end

    update_convexity_duals!(st)
    ctr.factor_updates = kkt.total_updates - updates_at_start
    accumulate!(st.total, ctr)
    return status
end

# ── direction ────────────────────────────────────────────────────────────────────────────

"""
    compute_direction!(state) -> ‖p‖∞

Solve the reduced KKT system and expand the direction (thesis Thm. 5.2.2, Decoupling of the
reduced system, eqs. 5.7–5.8, and §5.2.4, The search direction):

- column part: `H_eff y = -δ Zᵀ(c - A_effᵀπ_eff)`, i.e. the right-hand side uses the duals
  with the rows of free slacks zeroed. With `zⱼ = eⱼ - e_rep(j)`: `pⱼ = yⱼ`, `p_rep(j) -= yⱼ`;
- slack part (free slacks only): `p_σ = κσ (δ πᵢ - (A p_x)ᵢ)`, which drives πᵢ to zero at a
  full step; slacks in the working set do not move.
"""
function compute_direction!(st::QPASState)
    ord = st.kkt.ordering
    sp = st.kkt.surplus
    Nset = items(ord.N)
    block = st.pool.block
    p = st.p
    pnorm = 0.0
    if !isempty(Nset)
        F = st.kkt.factor
        # duals of A_eff and the convexity duals derived from them
        πe, μe = masked_duals!(st)
        rhs = LDLt.rhs_buffer(F)
        neg_delta = -st.delta
        # reduced gradient (Zᵀg̃)ₖ = gₖ - g_rep(k) with g = c - A_effᵀπ_eff
        @inbounds for k in Nset
            rhs[k] = neg_delta * (st.pool.cost[k] - column_dot(st.pool, k, πe) - μe[block[k]])
        end
        y = _reduced_solve!(st)
        # p = Zy: weight moves from each representative to the columns of its block
        @inbounds for k in Nset
            rhs[k] = 0.0
            yk = y[k]
            p[k] = yk
            p[ord.rep[block[k]]] -= yk
            pnorm = max(pnorm, abs(yk))
        end
        @inbounds for r in ord.rep
            pnorm = max(pnorm, abs(p[r]))
        end
    end
    if any_free(sp)
        # slack part (thesis Rem. 5.3.3, "≤" rows): p_σ = κσ (δπᵢ - (A p_x)ᵢ)
        _column_product!(st, 0)
        @inbounds for σ in items(sp.free)
            i = sp.row[σ]
            st.ps[σ] = sp.sign[σ] * (st.delta * st.π[i] - st.Ap[i])
            pnorm = max(pnorm, abs(st.ps[σ]))
        end
    end
    return pnorm
end

# Solve with the factor for a right-hand side supported on N; the result is valid on N.
@inline function _reduced_solve!(st::QPASState)
    kkt = st.kkt
    F = kkt.factor
    if st.options.sparse_solves
        Nset = items(kkt.ordering.N)
        return @timed_section kkt.timer :solve LDLt.solve!(F, Nset, length(Nset))
    end
    return @timed_section kkt.timer :solve LDLt.solve!(F)
end

# `st.Ap = A p_x` over the support of the column direction (`N ∪ S`, plus column `extra`).
function _column_product!(st::QPASState, extra::Int)
    ord = st.kkt.ordering
    pool = st.pool
    p, Ap = st.p, st.Ap
    fill!(Ap, 0.0)
    @inbounds for k in items(ord.N)
        p[k] == 0.0 || column_axpy!(Ap, p[k], pool, k)
    end
    @inbounds for k in ord.rep
        p[k] == 0.0 || column_axpy!(Ap, p[k], pool, k)
    end
    extra != 0 && p[extra] != 0.0 && column_axpy!(Ap, p[extra], pool, extra)
    return Ap
end

"Zero the direction on its support `N ∪ S ∪ N_s` (and on `extra` / `extra_s` if nonzero)."
function clear_direction!(st::QPASState, extra::Int = 0, extra_s::Int = 0)
    ord = st.kkt.ordering
    p = st.p
    @inbounds for k in items(ord.N)
        p[k] = 0.0
    end
    @inbounds for r in ord.rep
        p[r] = 0.0
    end
    extra != 0 && (p[extra] = 0.0)
    @inbounds for σ in items(st.kkt.surplus.free)
        st.ps[σ] = 0.0
    end
    extra_s != 0 && (st.ps[extra_s] = 0.0)
    return nothing
end

# ── primal step ──────────────────────────────────────────────────────────────────────────

"""
    ratio_test(state, αmax, extra, extra_s) -> (α, q, q_is_slack, q_at_upper)

Largest `α ≤ αmax` keeping `x + αp ≥ 0` and `0 ≤ s + αp_s ≤ U` over the support of the
direction (`N ∪ S`, the free slacks `N_s`, plus column `extra` / slack `extra_s` if nonzero).
`q` is the blocking index (a column, or a slack if `q_is_slack`), or 0 if `α = αmax` is not
blocked. `q_at_upper` is `true` if slack `q` blocks at its upper bound (range rows only).
"""
function ratio_test(st::QPASState, αmax::Float64, extra::Int, extra_s::Int)
    ord = st.kkt.ordering
    tol = st.options.ratio_tolerance
    α = αmax
    q = 0
    q_slack = false
    q_upper = false
    x, p = st.x, st.p
    # columns: αₖ = xₖ / -pₖ for every decreasing component
    @inbounds for k in items(ord.N)
        pk = p[k]
        if pk < -tol
            αk = max(x[k], 0.0) / -pk
            αk < α && ((α, q) = (αk, k))
        end
    end
    @inbounds for k in ord.rep
        pk = p[k]
        if pk < -tol
            αk = max(x[k], 0.0) / -pk
            αk < α && ((α, q) = (αk, k))
        end
    end
    if extra != 0
        @inbounds pk = p[extra]
        if pk < -tol
            αk = max(x[extra], 0.0) / -pk
            αk < α && ((α, q) = (αk, extra))
        end
    end
    # slacks: the same test on s ≥ 0, and on s ≤ U for range rows
    s, ps, U = st.s, st.ps, st.kkt.surplus.upper
    @inbounds for σ in items(st.kkt.surplus.free)
        pσ = ps[σ]
        if pσ < -tol
            ασ = max(s[σ], 0.0) / -pσ
            ασ < α && ((α, q, q_slack, q_upper) = (ασ, σ, true, false))
        elseif pσ > tol && isfinite(U[σ])
            ασ = max(U[σ] - s[σ], 0.0) / pσ
            ασ < α && ((α, q, q_slack, q_upper) = (ασ, σ, true, true))
        end
    end
    if extra_s != 0
        @inbounds pσ = ps[extra_s]
        if pσ < -tol
            ασ = max(s[extra_s], 0.0) / -pσ
            ασ < α && ((α, q, q_slack, q_upper) = (ασ, extra_s, true, false))
        elseif pσ > tol && isfinite(U[extra_s])
            ασ = max(U[extra_s] - s[extra_s], 0.0) / pσ
            ασ < α && ((α, q, q_slack, q_upper) = (ασ, extra_s, true, true))
        end
    end
    return α, q, q_slack, q_upper
end

"""
    take_step!(state, α, extra, extra_s)

`y += αp` and `π -= (α/δ)·Āp` over the support of the direction, where
`Āp = Σₖ pₖ aₖ + Σσ κσ p_σ e_row(σ)` (columns `N ∪ S` and `extra`, slacks `N_s` and `extra_s`).
"""
function take_step!(st::QPASState, α::Float64, extra::Int, extra_s::Int)
    ord = st.kkt.ordering
    sp = st.kkt.surplus
    pool = st.pool
    x, p, Ap = st.x, st.p, st.Ap
    fill!(Ap, 0.0)
    # column part: x += αp and accumulate A p_x
    @inbounds for k in items(ord.N)
        _step_component!(x, Ap, pool, k, p[k], α)
    end
    @inbounds for k in ord.rep
        _step_component!(x, Ap, pool, k, p[k], α)
    end
    extra != 0 && _step_component!(x, Ap, pool, extra, p[extra], α)
    # slack part: s += αp_s and accumulate K p_s
    @inbounds for σ in items(sp.free)
        _step_slack!(st, sp, σ, α)
    end
    extra_s != 0 && _step_slack!(st, sp, extra_s, α)
    # π is affine in y (thesis eq. 3.9, Dual estimate): π -= (α/δ) Āp
    scale = α / st.delta
    @inbounds for i in eachindex(st.π)
        st.π[i] -= scale * Ap[i]
    end
    return nothing
end

@inline function _step_component!(x, Ap, pool, k, pk, α)
    pk == 0.0 && return nothing
    # clamp at 0 so round-off never makes a blocking weight negative
    @inbounds x[k] = max(x[k] + α * pk, 0.0)
    column_axpy!(Ap, pk, pool, k)
    return nothing
end

@inline function _step_slack!(st, sp, σ, α)
    pσ = st.ps[σ]
    pσ == 0.0 && return nothing
    # clamp to [0, U] so round-off never pushes a blocking slack past its bound
    @inbounds st.s[σ] = clamp(st.s[σ] + α * pσ, 0.0, sp.upper[σ])
    @inbounds st.Ap[sp.row[σ]] += sp.sign[σ] * pσ
    return nothing
end

# ── multipliers ──────────────────────────────────────────────────────────────────────────

"""
    most_negative_multiplier(state) -> (λmin, r, r_is_slack)

At a subspace minimizer `h = g`, so by thesis Thm. 5.3.1 (Lagrange multipliers) and
Cor. 5.3.2 (Multipliers at a subspace minimizer) `μ = g_S` and the bound multipliers (thesis
`ν_x`, `ν_s`) are

- `λⱼ = gⱼ - μ_ψ(j)` for columns `j ∈ W_x`: the reduced cost of column `j`;
- `λσ = g_σ = -κσ πᵢ` for slacks `σ ∈ W_s`: the dual of row `i`, signed so that `λσ ≥ 0`
  is the dual sign condition (`πᵢ ≥ 0` for `≥` rows, `πᵢ ≤ 0` for `≤` rows).

Returns the most negative one, its index, and whether it is a slack (`r = 0` if `W` is empty).
"""
function most_negative_multiplier(st::QPASState)
    μ = update_convexity_duals!(st)
    block = st.pool.block
    λmin = Inf
    r = 0
    @inbounds for j in items(st.kkt.ordering.W)
        λj = gradient(st, j) - μ[block[j]]
        λj < λmin && ((λmin, r) = (λj, j))
    end
    r_slack = false
    sp = st.kkt.surplus
    @inbounds for σ in 1:n_surplus(sp)
        is_free(sp, σ) && continue
        λσ = surplus_gradient(st, σ)
        λσ < λmin && ((λmin, r, r_slack) = (λσ, σ, true))
    end
    return λmin, r, r_slack
end

# ── working-set updates: columns ─────────────────────────────────────────────────────────

"""
    expand_working_set!(state, q)

Add the blocking column `q` to the working set (thesis §6.2, Expanding the working set;
Lemma 6.2.2, Existence of a replacement):

- `q ∈ N`: delete row/column `q` from the factor.
- `q ∈ S`: pick a replacement representative `r` among the N-columns of the same block (the
  one with the largest weight), and refresh every affected column `j ∈ ν` of the factor,
  since `zⱼ` changes from `eⱼ - e_q` to `eⱼ - e_r` (thesis Thm. 6.2.3 (ii), Update of the
  nullspace basis; Thm. 6.2.4 (ii), Update of the effective reduced Hessian, expansion).
"""
function expand_working_set!(st::QPASState, q::Int)
    kkt = st.kkt
    ord = kkt.ordering
    if in_N(ord, q)
        deactivate!(kkt, q)
    elseif in_S(ord, q)
        b = st.pool.block[q]
        # ν = N-columns of block b, whose nullspace vectors all reference q
        ν = kkt.affected
        empty!(ν)
        for j in items(ord.N)
            st.pool.block[j] == b && push!(ν, j)
        end
        isempty(ν) &&
            error("QPAS: representative $q of block $b blocks but the block has no free column")
        # the heaviest column is the least likely to block next
        r = ν[argmax(view(st.x, ν))]
        _swap_representative!(st, r, ν)
        st.last.representative_swaps += 1
    else
        error("QPAS: blocking index $q is already in the working set")
    end
    move_to_W!(ord, q)
    st.x[q] = 0.0
    st.last.expansions += 1
    _check_factor!(st)
    return nothing
end

# Make `newrep` the representative of its block. `ν` lists the N-columns of that block (it
# may or may not contain `newrep`); their factor rows are rebuilt for the new representative.
# The old representative is left for the caller to place.
function _swap_representative!(st::QPASState, newrep::Int, ν::Vector{Int})
    kkt = st.kkt
    ord = kkt.ordering
    # u_j changes from a_j - a_q to a_j - a_newrep: remove the old rows ...
    for j in ν
        deactivate!(kkt, j)
    end
    make_representative!(ord, st.pool.block, newrep)
    # ... and add them back assembled against the new representative
    for j in ν
        j == newrep && continue
        activate!(kkt, st.pool, j)
    end
    return nothing
end

"""
    reduce_working_set!(state, r)

Release column `r` (negative multiplier) from the working set into `N` (thesis §6.3, Reducing the
working set; Thm. 6.3.1 (iv)) by adding row/column `r` to the factor.

If the new column `A_eff z_r` depends on the columns of `A_eff Z`, the effective reduced
Hessian becomes singular (thesis Prop. 7.1.1 (i), Singular reductions; the step below is
Thm. 7.3.1, Zero-curvature direction, column, and Alg. 7.1); this shows up as a vanishing
LDLᵀ pivot right after the `rowadd`. The update is then undone (refactorization without `r`), and with `t = H⁻¹h` the
direction `d = z_r - Σ tₖ zₖ` (plus the slack components that keep `Ād = 0`) satisfies
`Ād = 0` and `gᵀd = λ_r < 0`: a zero-curvature descent direction. The solver steps along `d`
until a variable blocks, adds that index to `W` (which removes the dependency), and retries.
Unlike picking `argmax |tₖ|` this keeps every working-set variable exactly at zero.
"""
function reduce_working_set!(st::QPASState, r::Int)
    kkt = st.kkt
    ord = kkt.ordering
    pool = st.pool
    tol = st.options.singular_tolerance
    for _ in 1:(length(ord.N) + n_surplus(kkt.surplus) + 2)
        # column r of H_eff: hₖ = (A_eff zₖ)ᵀ(A_eff z_r), h_rr = ‖A_eff z_r‖²
        nnz, hrr, scale = hessian_column!(kkt, pool, r)
        if hrr > 0.0
            insert_column!(kkt, r, nnz, scale)
            if min_pivot_ratio(kkt) > tol           # invariant: all other pivots are > tol
                st.last.reductions += 1
                _check_refactor_interval!(st)
                return nothing
            end
            # dependent column: undo by rebuilding the factor without r
            move_to_W!(ord, r)
            _refactor!(st)
            nnz, hrr, scale = hessian_column!(kkt, pool, r)
        end
        # t = H_eff⁻¹h: coefficients of A_eff z_r in the basis A_eff Z
        t = nnz > 1 ? _solve_on_N!(st, kkt.col_rows, kkt.col_vals, nnz - 1) : nothing
        _singular_step!(st, r, t) && return nothing
    end
    error("QPAS: could not resolve a singular reduced Hessian when releasing column $r")
end

# Solve H_eff t = v for a sparse right-hand side given by (rows, vals)[1:nnz]; returns t (valid
# on N) and leaves the factor's right-hand-side buffer zero again.
function _solve_on_N!(st::QPASState, rows::Vector{Int}, vals::Vector{Float64}, nnz::Int)
    rhs = LDLt.rhs_buffer(st.kkt.factor)
    @inbounds for s in 1:nnz
        rhs[rows[s]] = vals[s]
    end
    t = _reduced_solve!(st)
    @inbounds for s in 1:nnz
        rhs[rows[s]] = 0.0
    end
    return t
end

# Zero-curvature step along d = z_r - Σₖ tₖ zₖ (see `reduce_working_set!`). Returns `true` if
# the step also completed the reduction (r became the representative of its block).
function _singular_step!(st::QPASState, r::Int, t)
    ord = st.kkt.ordering
    block = st.pool.block
    p = st.p
    # column part of d: z_r - Σ tₖ zₖ written out with zₖ = eₖ - e_rep(k)
    @inbounds begin
        p[r] += 1.0
        p[ord.rep[block[r]]] -= 1.0
        if t !== nothing
            for k in items(ord.N)
                tk = t[k]
                p[k] -= tk
                p[ord.rep[block[k]]] += tk
            end
        end
    end
    # slack part: free slacks absorb the residual of their row so that Ād = 0 there
    _zero_curvature_slacks!(st, r, 0)
    α, q, q_slack, q_upper = ratio_test(st, Inf, r, 0)
    q == 0 && error("QPAS: unbounded zero-curvature direction (infeasible master?)")
    take_step!(st, α, r, 0)
    clear_direction!(st, r)
    st.last.singular_steps += 1

    if q_slack
        expand_slack!(st, q, q_upper)
        return false
    end
    if in_S(ord, q) && block[q] == block[r]
        # The representative of r's block blocked: r (now positive) takes its place.
        b = block[r]
        ν = st.kkt.affected
        empty!(ν)
        for j in items(ord.N)
            block[j] == b && push!(ν, j)
        end
        _swap_representative!(st, r, ν)
        move_to_W!(ord, q)
        st.x[q] = 0.0
        st.last.representative_swaps += 1
        st.last.expansions += 1
        st.last.reductions += 1
        _check_factor!(st)
        return true
    end
    expand_working_set!(st, q)
    return false
end

# Slack components of a zero-curvature direction: p_σ = -κσ (A p_x)ᵢ on the free slacks (and on
# `extra_s`), so that row i of Āp = (A p_x)ᵢ + κσ p_σ vanishes. `extra` is an extra column in
# the support of p_x.
function _zero_curvature_slacks!(st::QPASState, extra::Int, extra_s::Int)
    sp = st.kkt.surplus
    (any_free(sp) || extra_s != 0) || return nothing
    _column_product!(st, extra)
    @inbounds for σ in items(sp.free)
        st.ps[σ] = -sp.sign[σ] * st.Ap[sp.row[σ]]
    end
    extra_s != 0 && (st.ps[extra_s] = -sp.sign[extra_s] * st.Ap[sp.row[extra_s]])
    return nothing
end

# ── working-set updates: slacks of inequality rows ───────────────────────────────────────

"""
    expand_surplus!(state, σ)

The free slack `σ` reached zero and enters the working set (`N_s → W_s`). Its row joins
`A_eff`, so `H_eff ← H_eff + vvᵀ` with `v = Zᵀâᵢ` (thesis Thm. 6.2.4 (iii), Update of the
effective reduced Hessian, expansion; eq. 6.1). An update cannot destroy positive
definiteness.
"""
function expand_surplus!(st::QPASState, σ::Int)
    kkt = st.kkt
    nnz = surplus_vector!(kkt, st.pool, σ)
    fix_surplus!(kkt.surplus, σ)
    nnz > 0 && rank_one_update!(kkt, true, nnz)
    st.s[σ] = 0.0
    st.ps[σ] = 0.0
    st.last.expansions += 1
    st.last.slack_expansions += 1
    _check_factor!(st)
    return nothing
end

"""
    expand_slack!(state, σ, at_upper)

The free slack `σ` blocked. At zero it enters the working set directly; at its upper bound
(range rows) the row first switches to its other form (`flip_bound!`), after which the new
slack is zero. Both cases are the same rank-one update of `H_eff`, since the row's contribution
`vvᵀ` does not depend on the slack's sign.
"""
function expand_slack!(st::QPASState, σ::Int, at_upper::Bool)
    if at_upper
        flip_bound!(st, σ)
        st.last.bound_flips += 1
    end
    expand_surplus!(st, σ)
    return nothing
end

"""
    reduce_surplus!(state, σ)

Release slack `σ` (negative multiplier `-κσ πᵢ < 0`) from the working set (`W_s → N_s`). Its
row leaves `A_eff`, so `H_eff ← H_eff - vvᵀ` (thesis Thm. 6.3.1 (v), Update of the effective
reduced Hessian, reduction; eq. 6.3).

The downdate is singular exactly when `vᵀH_eff⁻¹v = 1` (thesis Prop. 7.1.1 (ii), Singular
reductions); this shows up as a vanishing
pivot. The update is then undone (refactorization with the row kept), and with
`t = H_eff⁻¹v` the direction `d_x = -κσ Σ tₖ zₖ` with slack components `-κ (A d_x)` on the free
rows and on row `i` has `Ād = 0`, `d_σ = 1` and `gᵀd = λσ < 0` (thesis Thm. 7.3.2, Zero-curvature direction, surplus, with the
sign `κ` of Rem. 7.3.3): the solver steps
along it until a variable blocks, adds that variable to the working set and retries.
"""
function reduce_surplus!(st::QPASState, σ::Int)
    kkt = st.kkt
    sp = kkt.surplus
    pool = st.pool
    tol = st.options.singular_tolerance
    for _ in 1:(length(kkt.ordering.N) + n_surplus(sp) + 2)
        # v = row i of A Z: the contribution of row i to H_eff
        nnz = surplus_vector!(kkt, pool, σ)
        free_surplus!(sp, σ)
        if nnz == 0                                  # row i does not touch Z: H_eff unchanged
            st.last.reductions += 1
            st.last.slack_reductions += 1
            return nothing
        end
        ok = rank_one_update!(kkt, false, nnz)
        if ok && min_pivot_ratio(kkt) > tol          # invariant: other pivots were > tol
            st.last.reductions += 1
            st.last.slack_reductions += 1
            _check_refactor_interval!(st)
            return nothing
        end
        # singular downdate: keep the row in A_eff and rebuild the factor
        fix_surplus!(sp, σ)
        _refactor!(st)
        nnz = surplus_vector!(kkt, pool, σ)
        t = _solve_on_N!(st, kkt.v_rows, kkt.v_vals, nnz)
        _surplus_singular_step!(st, σ, t)
    end
    error("QPAS: could not resolve a singular reduced Hessian when releasing slack $σ")
end

# Zero-curvature step for a slack release (see `reduce_surplus!`).
function _surplus_singular_step!(st::QPASState, σ::Int, t)
    ord = st.kkt.ordering
    sp = st.kkt.surplus
    block = st.pool.block
    p = st.p
    # column part d_x = -κσ Σₖ tₖ zₖ, scaled so that the slack component d_σ equals 1
    κ = sp.sign[σ]
    @inbounds for k in items(ord.N)
        tk = -κ * t[k]
        p[k] += tk
        p[ord.rep[block[k]]] -= tk
    end
    _zero_curvature_slacks!(st, 0, σ)
    α, q, q_slack, q_upper = ratio_test(st, Inf, 0, σ)
    q == 0 && error("QPAS: unbounded zero-curvature direction (infeasible master?)")
    take_step!(st, α, 0, σ)
    clear_direction!(st, 0, σ)
    st.last.singular_steps += 1
    q_slack ? expand_slack!(st, q, q_upper) : expand_working_set!(st, q)
    return nothing
end

# ── factor maintenance ───────────────────────────────────────────────────────────────────

# After an expansion: rebuild the factor when the refactorization interval of the backend is
# reached (CHOLMOD: removes structural zeros left by rowdel/updown) or when a pivot has
# degraded. Keeps the invariant that every pivot in N is above `singular_tolerance`.
function _check_factor!(st::QPASState)
    opts = st.options
    kkt = st.kkt
    interval = refactor_interval(opts)
    due = interval > 0 && kkt.updates_since_refactor >= interval
    if due || (opts.check_pivots && min_pivot_ratio(kkt) < opts.singular_tolerance)
        _refactor!(st)
        # A rebuild cannot raise a pivot that is small in exact arithmetic. This happens only if
        # a representative swap made H_eff itself nearly singular (the re-appended rows get new
        # vectors and scales); later releases would then fail the pivot test spuriously.
        if opts.check_pivots && min_pivot_ratio(kkt) < opts.singular_tolerance
            @warn "QPAS: pivot invariant violated after a rebuild; H_eff is nearly singular" maxlog = 1
        end
    end
    return nothing
end

# After a reduction (whose pivots were just checked): only the refactorization interval.
function _check_refactor_interval!(st::QPASState)
    interval = refactor_interval(st.options)
    interval > 0 && st.kkt.updates_since_refactor >= interval && _refactor!(st)
    return nothing
end

function _refactor!(st::QPASState)
    @timed_section st.timer :refactor refactor!(st.kkt, st.pool)
    st.last.refactorizations += 1
    return nothing
end
