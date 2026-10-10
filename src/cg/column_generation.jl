# Column generation driver (thesis Alg. 1.3 for the LP master, Alg. 4.1 for the regularized
# master). The driver is independent of how the restricted master is solved: it talks to an
# `AbstractMasterSolver` through the interface in `src/master/interface.jl`.

"""
    solve_column_generation(instance, master; options = CGOptions()) -> CGResult

Solve the LP relaxation of `instance` by column generation, using `master` for the restricted
master problem:

- [`HighsLPMaster`](@ref): classical LP master with Big-M slacks
- [`HighsExplicitQPMaster`](@ref): stabilized master, explicit `w` formulation, HiGHS QP
- [`HighsEliminatedQPMaster`](@ref): stabilized master, eliminated formulation, HiGHS QP
- [`QPASMaster`](@ref): stabilized master, eliminated formulation, hot-started QPAS

The initial columns are the cheapest path of every pilot. Each iteration solves the master,
prices the pilots in random order with the master duals, and adds improving columns.

Termination is exact (no heuristic stopping):
- unstabilized: a full pricing round over all pilots finds no reduced cost `< -tol`;
- stabilized: additionally `π = π̂` up to `center_tolerance`. If pricing finds nothing but
  the center has not converged, the center is moved to `π` and the master is re-solved
  without new columns (a proximal step on the current restricted model).
"""
function solve_column_generation(inst::CrewSchedulingInstance, master::AbstractMasterSolver;
                                 options::CGOptions = CGOptions())
    validate(options)
    if has_inequalities(inst) && !supports_inequalities(master)
        throw(ArgumentError("$(method_name(master)) does not support inequality coupling rows"))
    end
    t_start = time_ns()
    T, P = n_tasks(inst), n_pilots(inst)
    lo, hi = inst.task_lower, inst.task_upper
    stab = options.stabilization
    stabilized = is_stabilized(master)
    full_pricing = options.pricing === :all_blocks || options.compute_bound ||
                   (stabilized && stab.center_update === :best_bound)

    stats = CGStats(method_name(master))
    pool = ColumnPool(T, P, options.max_columns)
    pricer = ShortestPathPricer(inst)
    rng = Xoshiro(options.seed)

    # initial columns: cheapest path of every pilot (pricing at zero duals); they make the
    # restricted master feasible for Cλ = 1 (one column per block)
    set_duals!(pricer, zeros(T))
    for k in 1:P
        _, cost = price_block!(pricer, k)
        rows, vals = path_coverage!(pricer)
        add_column!(pool, k, cost, rows, vals, pricer.path)
    end

    spec = MasterSpec(lo, hi, P, options.max_columns, stabilized ? stab.delta : NaN)
    initialize!(master, spec, pool)

    # stability center π̂ of the proximal term δ/2‖π - π̂‖²
    center = stab.initial_center === nothing ? zeros(T) : copy(stab.initial_center)
    length(center) == T || throw(DimensionMismatch("initial_center must have length $T"))
    center_value = -Inf
    if stabilized
        set_center!(master, center)
        if stab.center_update === :best_bound
            center_value = lagrangian_bound(pricer, center, lo, hi, P)
        end
    end

    prev_π = zeros(T)
    price_π = zeros(T)                          # sign-feasible copy of the master duals
    Ax = zeros(T)
    order = collect(1:P)
    status = :iteration_limit
    final = nothing
    stats.setup_time = (time_ns() - t_start) * 1e-9
    options.verbose && _log_header(stats.method)

    for iter in 1:options.max_iterations
        # ── restricted master ────────────────────────────────────────────────────────────
        t0 = time_ns()
        sol = solve!(master)
        t_master = (time_ns() - t0) * 1e-9
        if sol.status !== :optimal
            status = :master_failure
            @warn "column generation: master solve returned $(sol.status) at iteration $iter"
            break
        end
        n_master = pool.n                       # columns present in this master solve

        # ── pricing ──────────────────────────────────────────────────────────────────────
        t1 = time_ns()
        # inequality rows: clip round-off sign violations (πᵢ ≥ -ε) so θ(π) is a valid bound
        project_duals!(price_π, sol.π, lo, hi)
        set_duals!(pricer, price_π)
        randperm!(rng, order)
        # Lagrangian bound θ(π) = min_{lo≤v≤hi} πᵀv + Σₖ (shortest path value of pilot k)
        θ = coupling_dual_term(price_π, lo, hi)
        priced_all = true
        limit_hit = false
        for (pos, k) in enumerate(order)
            value, cost = price_block!(pricer, k)
            θ += value
            # reduced cost of the best path of pilot k: (cₖ - A_taskᵀπ)ᵀx - μₖ
            value - sol.μ[k] < -options.reduced_cost_tolerance || continue
            if contains_column(pool, k, pricer.path)
                stats.duplicate_columns += 1
            elseif is_full(pool)
                limit_hit = true
            else
                rows, vals = path_coverage!(pricer)
                add_column!(pool, k, cost, rows, vals, pricer.path)
                if !full_pricing && options.pricing === :first_improving
                    priced_all = pos == P
                    break
                end
            end
        end
        n_new = pool.n - n_master
        # θ is only a valid Lagrangian bound if every pilot was priced at the same π
        bound = priced_all ? θ : NaN
        t_pricing = (time_ns() - t1) * 1e-9

        # ── convergence and stability center ─────────────────────────────────────────────
        converged = false
        center_moved = false
        if n_new == 0 && !limit_hit
            # no improving column: optimal once π = π̂ (w = 0, the master is feasible)
            if !stabilized || _center_converged(sol.π, center, options.center_tolerance)
                converged = true
            else
                center_moved = true             # proximal step without new columns
                center_value = bound
            end
        elseif stabilized
            # proximal point: always move; bundle: move only on a better Lagrangian bound
            if stab.center_update === :every_iteration || bound > center_value
                center_moved = true
                isnan(bound) || (center_value = bound)
            end
        end
        # hand the changes to the master; nothing is solved until the next iteration
        n_new > 0 && add_columns!(master, pool, (n_master + 1):pool.n)
        if center_moved
            copyto!(center, sol.π)
            set_center!(master, center)
        end

        # ── statistics ───────────────────────────────────────────────────────────────────
        # cᵀλ and the coupling violation of λ in the unstabilized master
        op_cost = _weighted_cost(pool, sol.weights, n_master)
        infeas = coupling_violation!(Ax, pool, sol.weights, n_master, lo, hi)
        record_iteration!(stats;
                          master_time = t_master, pricing_time = t_pricing,
                          master_objective = sol.objective, operational_cost = op_cost,
                          infeasibility = infeas, lower_bound = bound,
                          dual_change = iter == 1 ? NaN : _maxabsdiff(sol.π, prev_π),
                          n_columns = pool.n, inner_iterations = sol.inner_iterations,
                          columns_added = n_new, center_updated = center_moved)
        copyto!(prev_π, sol.π)
        if options.diagnostics_every > 0 && iter % options.diagnostics_every == 0
            _record_diagnostics!(stats, master, pool, iter, (time_ns() - t_start) * 1e-9)
        end
        final = (solution = sol, cost = op_cost, infeasibility = infeas)   # copied after the loop

        if options.verbose && (iter % options.log_every == 0 || converged)
            _log_line(stats, (time_ns() - t_start) * 1e-9)
        end

        if converged
            status = :optimal
            break
        elseif limit_hit && n_new == 0
            status = :column_limit
            break
        elseif (time_ns() - t_start) * 1e-9 > options.time_limit
            status = :time_limit
            break
        end
    end

    # one final snapshot, unless the last iteration was already sampled
    if options.diagnostics_every > 0 &&
       (isempty(stats.diagnostics) || stats.diagnostics[end].iteration != stats.iterations)
        _record_diagnostics!(stats, master, pool, stats.iterations, (time_ns() - t_start) * 1e-9)
    end
    stats.status = status
    stats.total_time = (time_ns() - t_start) * 1e-9
    stats.details = solver_details(master)
    options.verbose && _log_footer(stats)

    if final === nothing
        return CGResult(stats.method, status, NaN, NaN, -Inf, NaN, zeros(pool.n), zeros(T),
                        zeros(P), pool, stats)
    end
    sol = final.solution          # buffers of the last master solve (no solve happened since)
    return CGResult(stats.method, status, final.cost, sol.objective, best_lower_bound(stats),
                    final.infeasibility, sol.weights[1:pool.n], copy(sol.π), copy(sol.μ),
                    pool, stats)
end

# ── helpers ──────────────────────────────────────────────────────────────────────────────

# Snapshot of memory use and solver internals (`CGOptions(diagnostics_every = k)`).
function _record_diagnostics!(stats::CGStats, master::AbstractMasterSolver, pool::ColumnPool,
                              iter::Integer, elapsed::Float64)
    base = (iteration = Int(iter), time = elapsed, columns = pool.n,
            # live Julia heap, peak resident set of the whole process, the column pool
            julia_heap_MB = Base.gc_live_bytes() / 2^20, peak_rss_MB = Sys.maxrss() / 2^20,
            pool_MB = pool_memory(pool) / 2^20)
    d = diagnostics(master)                    # solver internals, e.g. the QPAS factor
    push!(stats.diagnostics, d === nothing ? base : merge(base, d))
    return nothing
end

"""
    coupling_dual_term(π, lower, upper) -> Float64

`min_{lower ≤ v ≤ upper} πᵀv`, the right-hand-side part of the Lagrangian dual function for
range-constrained coupling rows (`πᵢbᵢ` for equalities, `-Inf` if a sign condition fails).
"""
function coupling_dual_term(π::AbstractVector{Float64}, lo::Vector{Float64}, hi::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(π)
        πi = π[i]
        πi == 0.0 && continue
        # min over lo ≤ vᵢ ≤ hi of πᵢvᵢ: lower bound if πᵢ > 0, upper bound otherwise
        v = πi > 0 ? lo[i] : hi[i]
        isfinite(v) || return -Inf
        s += πi * v
    end
    return s
end

"""
    project_duals!(out, π, lower, upper) -> out

Projection of `π` onto the dual sign conditions of range-constrained coupling rows:
`πᵢ ≥ 0` for `≥` rows (`upper = +∞`), `πᵢ ≤ 0` for `≤` rows (`lower = -∞`), unrestricted
otherwise. Master solvers satisfy these conditions only up to their tolerance; without the
projection a tiny wrong-signed entry would make [`coupling_dual_term`](@ref) `-Inf`. For
equality rows this is a copy.
"""
function project_duals!(out::AbstractVector{Float64}, π::AbstractVector{Float64},
                        lo::Vector{Float64}, hi::Vector{Float64})
    @inbounds for i in eachindex(π)
        πi = π[i]
        if hi[i] == Inf && lo[i] > -Inf
            πi = max(πi, 0.0)                   # ≥ row
        elseif lo[i] == -Inf && hi[i] < Inf
            πi = min(πi, 0.0)                   # ≤ row
        elseif lo[i] == -Inf && hi[i] == Inf
            πi = 0.0                            # free row: no constraint, zero dual
        end
        out[i] = πi
    end
    return out
end

"""
    lagrangian_bound(pricer, π, lower, upper, n_pilots) -> Float64

Lagrangian dual value `θ(π) = min_{lower≤v≤upper} πᵀv + Σₖ min_{x∈Pₖ} (cₖ - A_taskᵀπ)ᵀx`, a valid
lower bound on the LP optimum (prices every pilot).
"""
function lagrangian_bound(pricer::ShortestPathPricer, π::AbstractVector{Float64},
                          lo::Vector{Float64}, hi::Vector{Float64}, n_pilots::Integer)
    set_duals!(pricer, π)
    θ = coupling_dual_term(π, lo, hi)
    for k in 1:n_pilots
        value, _ = price_block!(pricer, k)
        θ += value
    end
    return θ
end

"Coupling-row violation `‖max(lo - Aλ, Aλ - hi, 0)‖∞` over the first `n` columns."
function coupling_violation!(Ax::Vector{Float64}, pool::ColumnPool, λ::AbstractVector{Float64},
                             n::Integer, lo::Vector{Float64}, hi::Vector{Float64})
    fill!(Ax, 0.0)
    @inbounds for j in 1:n
        λ[j] == 0.0 || column_axpy!(Ax, λ[j], pool, j)
    end
    v = 0.0
    @inbounds for i in eachindex(Ax)
        v = max(v, lo[i] - Ax[i], Ax[i] - hi[i])
    end
    return v
end

function _weighted_cost(pool::ColumnPool, λ::AbstractVector{Float64}, n::Integer)
    s = 0.0
    @inbounds for j in 1:n
        s += pool.cost[j] * λ[j]
    end
    return s
end

function _maxabsdiff(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    d = 0.0
    @inbounds for i in eachindex(a)
        d = max(d, abs(a[i] - b[i]))
    end
    return d
end

# relative test ‖π - π̂‖∞ ≤ tol·max(1, ‖π‖∞); for the eliminated form π - π̂ = (b - Āy)/δ, so
# this bounds the coupling violation by δ·tol·max(1, ‖π‖∞)
_center_converged(π, center, tol) =
    _maxabsdiff(π, center) <= tol * max(1.0, maximum(abs, π; init = 0.0))

function _log_header(method)
    println("── ", method, " ", "─"^max(0, 100 - length(method)))
    @printf("%7s %9s %15s %15s %15s %9s %9s %6s %5s %7s %9s\n", "iter", "time[s]",
            "cost cᵀλ", "master obj", "Lagr. bound", "infeas", "Δπ", "cols", "added", "inner",
            "master ms")
end

function _log_line(s::CGStats, elapsed)
    k = s.iterations
    @printf("%7d %9.2f %15.6f %15.6f %15.6f %9.2e %9.2e %6d %5d %7d %9.3f\n", k, elapsed,
            s.operational_cost[k], s.master_objective[k], s.lower_bound[k], s.infeasibility[k],
            s.dual_change[k], s.n_columns[k], s.columns_added[k], s.inner_iterations[k],
            1e3 * s.master_time[k])
    flush(stdout)
end

_log_footer(s::CGStats) =
    @printf("── finished: %s after %d iterations, %.3f s\n", s.status, s.iterations, s.total_time)
