# Active-set method with efficient working-set updates (thesis Alg. 4.3).
#
#   repeat
#       if not at a subspace minimizer:
#           solve H y = -δ Zᵀg, p = Zy                          (compute_direction!)
#       if ‖p‖ ≈ 0 (subspace minimizer):
#           λⱼ = gⱼ - μ_ψ(j) for j ∈ W; if min λ ≥ -tol → optimal  (most_negative_multiplier)
#           else release the most negative one from W             (reduce_working_set!)
#       else:
#           ratio test, x += αp, π -= (α/δ)Ap                      (ratio_test, take_step!)
#           if blocked: add the blocking index to W               (expand_working_set!)
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
            λmin, r = @timed_section timer :multipliers most_negative_multiplier(st)
            if r == 0 || λmin >= -opts.dual_tolerance * st.cost_scale
                status = :optimal
                break
            end
            @timed_section timer :reduce reduce_working_set!(st, r)
            at_minimizer = false
        else
            α, q = @timed_section timer :step ratio_test(st, 1.0, 0)
            @timed_section timer :step take_step!(st, α, 0)
            clear_direction!(st)
            ctr.steps += 1
            if q == 0
                ctr.full_steps += 1
                at_minimizer = true
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

Solve the reduced KKT system `H y = -δ Zᵀg` (thesis eq. 4.32) and expand `p = Zy` into
`state.p` (support `N ∪ S`). With `zⱼ = eⱼ - e_rep(j)`: `(Zᵀg)ⱼ = gⱼ - g_rep(j)`, and
`pⱼ = yⱼ`, `p_rep(j) -= yⱼ`.
"""
function compute_direction!(st::QPASState)
    ord = st.kkt.ordering
    Nset = items(ord.N)
    isempty(Nset) && return 0.0
    block = st.pool.block
    p = st.p
    F = st.kkt.factor
    μ = update_convexity_duals!(st)            # gradient at the representatives
    rhs = Cholmod.rhs_buffer(F)
    neg_delta = -st.delta
    @inbounds for k in Nset
        rhs[k] = neg_delta * (gradient(st, k) - μ[block[k]])
    end
    y = _reduced_solve!(st)
    pnorm = 0.0
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
    return pnorm
end

# Solve with the factor for a right-hand side supported on N; the result is valid on N.
@inline function _reduced_solve!(st::QPASState)
    F = st.kkt.factor
    if st.options.sparse_solves
        Nset = items(st.kkt.ordering.N)
        return Cholmod.solve!(F, Nset, length(Nset))
    end
    return Cholmod.solve!(F)
end

"Zero the direction on its support `N ∪ S` (and on `extra` if nonzero)."
function clear_direction!(st::QPASState, extra::Int = 0)
    ord = st.kkt.ordering
    p = st.p
    @inbounds for k in items(ord.N)
        p[k] = 0.0
    end
    @inbounds for r in ord.rep
        p[r] = 0.0
    end
    extra != 0 && (p[extra] = 0.0)
    return nothing
end

# ── primal step ──────────────────────────────────────────────────────────────────────────

"""
    ratio_test(state, αmax, extra) -> (α, q)

Largest `α ≤ αmax` keeping `x + αp ≥ 0` over the support of `p` (`N ∪ S`, plus index `extra`
if nonzero). `q` is the blocking index, or 0 if `α = αmax` is not blocked.
"""
function ratio_test(st::QPASState, αmax::Float64, extra::Int)
    ord = st.kkt.ordering
    tol = st.options.ratio_tolerance
    α = αmax
    q = 0
    x, p = st.x, st.p
    @inbounds for k in items(ord.N)
        pk = p[k]
        if pk < -tol
            αk = max(x[k], 0.0) / -pk
            if αk < α
                α, q = αk, k
            end
        end
    end
    @inbounds for k in ord.rep
        pk = p[k]
        if pk < -tol
            αk = max(x[k], 0.0) / -pk
            if αk < α
                α, q = αk, k
            end
        end
    end
    if extra != 0
        @inbounds pk = p[extra]
        if pk < -tol
            αk = max(x[extra], 0.0) / -pk
            if αk < α
                α, q = αk, extra
            end
        end
    end
    return α, q
end

"`x += αp` and `π -= (α/δ)·Ap` over the support of `p` (`N ∪ S`, plus `extra` if nonzero)."
function take_step!(st::QPASState, α::Float64, extra::Int)
    ord = st.kkt.ordering
    pool = st.pool
    x, p, Ap = st.x, st.p, st.Ap
    fill!(Ap, 0.0)
    @inbounds for k in items(ord.N)
        _step_component!(x, Ap, pool, k, p[k], α)
    end
    @inbounds for k in ord.rep
        _step_component!(x, Ap, pool, k, p[k], α)
    end
    extra != 0 && _step_component!(x, Ap, pool, extra, p[extra], α)
    scale = α / st.delta
    @inbounds for i in eachindex(st.π)
        st.π[i] -= scale * Ap[i]
    end
    return nothing
end

@inline function _step_component!(x, Ap, pool, k, pk, α)
    pk == 0.0 && return nothing
    @inbounds x[k] = max(x[k] + α * pk, 0.0)
    column_axpy!(Ap, pk, pool, k)
    return nothing
end

# ── multipliers ──────────────────────────────────────────────────────────────────────────

"""
    most_negative_multiplier(state) -> (λmin, r)

At a subspace minimizer `h = g`, so by thesis Theorem 4.3.5 `μ = g_S` and
`λⱼ = gⱼ - μ_ψ(j)` for `j ∈ W` — the reduced cost of column `j`. Returns the most negative
one and its index (`r = 0` if `W` is empty).
"""
function most_negative_multiplier(st::QPASState)
    μ = update_convexity_duals!(st)
    block = st.pool.block
    λmin = Inf
    r = 0
    @inbounds for j in items(st.kkt.ordering.W)
        λj = gradient(st, j) - μ[block[j]]
        if λj < λmin
            λmin, r = λj, j
        end
    end
    return λmin, r
end

# ── working-set updates ──────────────────────────────────────────────────────────────────

"""
    expand_working_set!(state, q)

Add the blocking index `q` to the working set (thesis §4.4.1, Lemma 4.4.1):

- `q ∈ N`: delete row/column `q` from the factor.
- `q ∈ S`: pick a replacement representative `r` among the N-columns of the same block (the
  one with the largest weight), and refresh every affected column `j ∈ ν` of the factor,
  since `zⱼ` changes from `eⱼ - e_q` to `eⱼ - e_r` (thesis Theorem 4.4.2, eq. 4.58).
"""
function expand_working_set!(st::QPASState, q::Int)
    kkt = st.kkt
    ord = kkt.ordering
    if in_N(ord, q)
        deactivate!(kkt, q)
    elseif in_S(ord, q)
        b = st.pool.block[q]
        ν = kkt.affected
        empty!(ν)
        for j in items(ord.N)
            st.pool.block[j] == b && push!(ν, j)
        end
        isempty(ν) &&
            error("QPAS: representative $q of block $b blocks but the block has no free column")
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
    for j in ν
        deactivate!(kkt, j)
    end
    make_representative!(ord, st.pool.block, newrep)
    for j in ν
        j == newrep && continue
        activate!(kkt, st.pool, j)
    end
    return nothing
end

"""
    reduce_working_set!(state, r)

Release index `r` (negative multiplier) from the working set into `N` (thesis §4.4.2) by
adding row/column `r` to the factor.

If the new column `A z_r` depends on the columns of `AZ`, the reduced Hessian becomes
singular (thesis §4.4.5); this shows up as a vanishing LDLᵀ pivot right after the `rowadd`.
The update is then undone (refactorization without `r`), and with `t = H⁻¹h` the direction
`d = z_r - Σ tₖ zₖ` satisfies `Ad = 0` and `gᵀd = λ_r < 0`: a zero-curvature descent
direction. The solver steps along `d` until a variable blocks, adds that index to `W` (which
removes the dependency), and retries. Unlike picking `argmax |tₖ|` this keeps every
working-set variable exactly at zero.
"""
function reduce_working_set!(st::QPASState, r::Int)
    kkt = st.kkt
    ord = kkt.ordering
    pool = st.pool
    tol = st.options.singular_tolerance
    for _ in 1:(length(ord.N) + 2)
        nnz, hrr = hessian_column!(kkt, pool, r)
        if hrr > 0.0
            insert_column!(kkt, r, nnz, hrr)
            if min_pivot_ratio(kkt) > tol           # invariant: all other pivots are > tol
                st.last.reductions += 1
                _check_refactor_interval!(st)
                return nothing
            end
            # dependent column: undo by rebuilding the factor without r
            move_to_W!(ord, r)
            _refactor!(st)
            nnz, hrr = hessian_column!(kkt, pool, r)
        end
        t = nothing
        if nnz > 1
            F = kkt.factor
            rhs = Cholmod.rhs_buffer(F)
            @inbounds for s in 1:(nnz - 1)
                rhs[kkt.col_rows[s]] = kkt.col_vals[s]
            end
            t = _reduced_solve!(st)                 # t = H⁻¹h, read on all of N
            @inbounds for s in 1:(nnz - 1)
                rhs[kkt.col_rows[s]] = 0.0
            end
        end
        _singular_step!(st, r, t) && return nothing
    end
    error("QPAS: could not resolve a singular reduced Hessian when releasing column $r")
end

# Zero-curvature step along d = z_r - Σₖ tₖ zₖ (see `reduce_working_set!`). Returns `true`
# if the step also completed the reduction (r became the representative of its block).
function _singular_step!(st::QPASState, r::Int, t)
    ord = st.kkt.ordering
    block = st.pool.block
    p = st.p
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
    α, q = ratio_test(st, Inf, r)
    q == 0 && error("QPAS: unbounded zero-curvature direction (infeasible master?)")
    take_step!(st, α, r)
    clear_direction!(st, r)
    st.last.singular_steps += 1

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

# After an expansion: rebuild the factor when the refactorization interval is reached
# (removes structural zeros left by rowdel) or when a pivot has degraded. Keeps the invariant
# that every pivot in N is above `singular_tolerance`.
function _check_factor!(st::QPASState)
    opts = st.options
    kkt = st.kkt
    due = opts.refactor_interval > 0 && kkt.updates_since_refactor >= opts.refactor_interval
    if due || (opts.check_pivots && min_pivot_ratio(kkt) < opts.singular_tolerance)
        _refactor!(st)
    end
    return nothing
end

# After a reduction (whose pivots were just checked): only the refactorization interval.
function _check_refactor_interval!(st::QPASState)
    interval = st.options.refactor_interval
    interval > 0 && st.kkt.updates_since_refactor >= interval && _refactor!(st)
    return nothing
end

function _refactor!(st::QPASState)
    @timed_section st.timer :refactor refactor!(st.kkt, st.pool)
    st.last.refactorizations += 1
    return nothing
end
