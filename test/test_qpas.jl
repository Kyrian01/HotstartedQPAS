# Tests of the hot-started QPAS in isolation: random multi-block regularized master problems,
# hot-started sequences (new columns + moving center), dependent columns that make the reduced
# Hessian singular, and the debug-mode factor verification. Everything runs once per LDLᵀ
# backend (CHOLMOD and dense).

# backend of the testset currently running; `_qopts` builds options for it
const QPAS_BACKEND = Ref(:cholmod)
_qopts(; kwargs...) = QPASOptions(; factorization = QPAS_BACKEND[], kwargs...)

# Random pool: block-structured columns with sparse 0/1 coupling coefficients.
function _random_pool(rng, m, P, n, capacity; density = 0.25)
    pool = HQ.ColumnPool(m, P, capacity)
    for j in 1:n
        _add_random_column!(rng, pool, j <= P ? j : rand(rng, 1:P); density)
    end
    return pool
end

function _add_random_column!(rng, pool, block; density = 0.25, rows = nothing)
    m = pool.n_rows
    if rows === nothing
        rows = sort(unique(rand(rng, 1:m, max(1, round(Int, density * m)))))
    end
    j = pool.n + 1
    HQ.add_column!(pool, block, 1 + 10 * rand(rng), rows, ones(length(rows)), [j])
    return j
end

# Reference: the same QP solved by HiGHS (eliminated formulation).
function _highs_reference(pool, b, δ, center)
    P, cap = pool.n_blocks, pool.capacity
    spec = HQ.MasterSpec(b, b, P, cap, δ)
    master = HighsEliminatedQPMaster()
    HQ.initialize!(master, spec, pool)
    HQ.set_center!(master, center)
    sol = HQ.solve!(master)
    @test sol.status === :optimal
    return (objective = sol.objective, π = copy(sol.π), μ = copy(sol.μ))
end

# Reference with inequality rows lo ≤ AXλ ≤ hi: the explicit form (thesis Def. 3.2.1, eq. 3.3) in HiGHS.
function _highs_ineq_reference(pool, lo, hi, δ, center)
    spec = HQ.MasterSpec(lo, hi, pool.n_blocks, pool.capacity, δ)
    master = HighsExplicitQPMaster()
    HQ.initialize!(master, spec, pool)
    HQ.set_center!(master, center)
    sol = HQ.solve!(master)
    @test sol.status === :optimal
    return (objective = sol.objective, π = copy(sol.π))
end

# QPAS state for coupling rows lo ≤ AXλ ≤ hi (slacks built as in QPASMaster).
function _ineq_state(pool, lo, hi, δ; options = _qopts(debug = true))
    b, rows, signs = HQ.slack_structure(lo, hi)
    return HQ.QPASState(pool, b, δ; options, slack_rows = rows, slack_signs = signs)
end

# First-order optimality of the QPAS iterate (independent of the solver's own bookkeeping),
# including the slacks of inequality rows: s ≥ 0, sign condition -κπᵢ ≥ 0, complementarity.
function _check_kkt(st; tol = 1e-7)
    pool = st.pool
    n = pool.n
    x = st.x[1:n]
    @test all(x .>= 0)
    @test all(st.s .>= 0)
    for b in 1:pool.n_blocks
        @test sum(x[j] for j in 1:n if pool.block[j] == b) ≈ 1 atol = 1e-10
    end
    A = Matrix(HQ.column_matrix(pool))
    sp = st.kkt.surplus
    Ks = zeros(pool.n_rows)
    for σ in eachindex(st.s)
        Ks[sp.row[σ]] += sp.sign[σ] * st.s[σ]
    end
    π = st.center .+ (st.rhs .- A * x .- Ks) ./ st.delta
    @test π ≈ st.π atol = 1e-9
    g = pool.cost[1:n] .- A' * π
    for j in 1:n
        λj = g[j] - st.μ[pool.block[j]]
        @test λj >= -tol
        x[j] > 1e-9 && @test abs(λj) <= tol
    end
    for σ in eachindex(st.s)
        λσ = -sp.sign[σ] * π[sp.row[σ]]
        @test λσ >= -tol
        st.s[σ] > 1e-9 && @test abs(λσ) <= tol
    end
end

# Random row types: equality (b = 1), "≥ 1" or "≤ 1" with the given fractions.
function _random_rows(rng, m; ge = 0.4, le = 0.2)
    lo, hi = ones(m), ones(m)
    for i in 1:m
        r = rand(rng)
        if r < ge
            hi[i] = Inf
        elseif r < ge + le
            lo[i] = -Inf
        end
    end
    return lo, hi
end

@testset "QPAS ($backend)" for backend in HQ.FACTORIZATIONS
    QPAS_BACKEND[] = backend

    @testset "single solve vs HiGHS (m=$m, P=$P, n=$n, δ=$δ)" for (m, P, n, δ) in
            [(8, 2, 12, 1.0), (20, 4, 60, 0.5), (30, 5, 120, 2.0), (15, 1, 40, 1.0)]
        rng = Xoshiro(m + P + n)
        pool = _random_pool(rng, m, P, n, 200)
        b = ones(m)
        center = randn(rng, m)
        st = HQ.QPASState(pool, b, δ; options = _qopts(debug = true))
        HQ.set_center!(st, center)
        @test HQ.qpas_solve!(st) === :optimal
        _check_kkt(st)
        ref = _highs_reference(pool, b, δ, center)
        @test HQ.master_objective(st) ≈ ref.objective rtol = 1e-6
        @test st.π ≈ ref.π atol = 1e-4
    end

    @testset "hot-started sequence" begin
        rng = Xoshiro(7)
        m, P, δ = 25, 4, 1.0
        pool = _random_pool(rng, m, P, P, 400)
        b = ones(m)
        st = HQ.QPASState(pool, b, δ; options = _qopts(debug = true))
        center = zeros(m)
        for it in 1:60
            HQ.set_center!(st, center)
            @test HQ.qpas_solve!(st) === :optimal
            _check_kkt(st)
            if it % 10 == 0
                ref = _highs_reference(pool, b, δ, center)
                @test HQ.master_objective(st) ≈ ref.objective rtol = 1e-6
            end
            center = copy(st.π)                       # proximal center update
            _add_random_column!(rng, pool, rand(rng, 1:P))
            HQ.register_columns!(st)
        end
        @test st.total.reductions > 0 && st.total.expansions > 0
        @test st.total.representative_swaps > 0
    end

    @testset "dependent columns (singular reduced Hessian)" begin
        rng = Xoshiro(11)
        m, P, δ = 6, 2, 1.0
        pool = _random_pool(rng, m, P, 8, 100; density = 0.5)
        # columns whose coupling part duplicates or combines existing ones
        for _ in 1:30
            src = rand(rng, 1:pool.n)
            rows = collect(HQ.column_rows(pool, src))
            _add_random_column!(rng, pool, rand(rng, 1:P); rows)
        end
        st = HQ.QPASState(pool, ones(m), δ; options = _qopts(debug = true))
        HQ.set_center!(st, randn(rng, m))
        @test HQ.qpas_solve!(st) === :optimal
        _check_kkt(st)
        ref = _highs_reference(pool, ones(m), δ, st.center)
        @test HQ.master_objective(st) ≈ ref.objective rtol = 1e-6
        # many more columns than rows: N can never exceed rank(A) ≤ m
        @test HQ.nullspace_dimension(st.kkt.ordering) <= m
    end

    # Constructed hot-start cases that force a dependent nullspace column (thesis Ch. 7, Prop. 7.1.1 / Thm. 7.3.1).
    # One block; rep a₁ = 0, a₂ = e₁, a₃ = e₂. After the first solve N = {2, 3} at π = 0.
    # Column 4 with a₄ = s·(e₁ + e₂) = s·(u₂ + u₃) and negative cost then enters dependent.
    #   s = 1   : d = z₄ - z₂ - z₃ raises the representative; an N-column blocks.
    #   s = 1/4 : d lowers the representative, which blocks first (swap branch).
    @testset "singular reduced Hessian, blocking by $(label)" for (s, label) in
            [(1.0, "an N-column"), (0.25, "the representative")]
        b = s == 1.0 ? [0.3, 0.3] : [0.45, 0.45]
        pool = HQ.ColumnPool(2, 1, 10)
        HQ.add_column!(pool, 1, 0.0, Int[], Float64[], [1])
        HQ.add_column!(pool, 1, 0.0, [1], [1.0], [2])
        HQ.add_column!(pool, 1, 0.0, [2], [1.0], [3])
        st = HQ.QPASState(pool, b, 1.0; options = _qopts(debug = true))
        @test HQ.qpas_solve!(st) === :optimal
        @test sort(HQ.items(st.kkt.ordering.N)) == [2, 3]
        @test st.π ≈ [0.0, 0.0] atol = 1e-12

        HQ.add_column!(pool, 1, -0.1, [1, 2], [s, s], [4])
        HQ.register_columns!(st)
        @test HQ.qpas_solve!(st) === :optimal
        @test st.last.singular_steps >= 1
        s == 0.25 && @test st.last.representative_swaps >= 1
        _check_kkt(st)
        ref = _highs_reference(pool, b, 1.0, zeros(2))
        @test HQ.master_objective(st) ≈ ref.objective rtol = 1e-7
    end

    # ── inequality coupling rows (docs/qpas_inequality_extension.md) ─────────────────────
    @testset "inequality rows: single solve vs HiGHS (m=$m, P=$P, n=$n, δ=$δ, ≥=$ge, ≤=$le)" for
            (m, P, n, δ, ge, le) in [(8, 2, 12, 1.0, 0.5, 0.0), (20, 4, 60, 0.5, 0.4, 0.2),
                                     (30, 5, 120, 2.0, 0.3, 0.3), (15, 1, 40, 1.0, 1.0, 0.0),
                                     (12, 3, 50, 1.0, 0.0, 1.0), (25, 3, 90, 0.1, 0.5, 0.5)]
        rng = Xoshiro(100 + m + P + n)
        pool = _random_pool(rng, m, P, n, 200)
        lo, hi = _random_rows(rng, m; ge, le)
        center = randn(rng, m)
        st = _ineq_state(pool, lo, hi, δ)
        HQ.set_center!(st, center)
        @test HQ.qpas_solve!(st) === :optimal
        _check_kkt(st)
        ref = _highs_ineq_reference(pool, lo, hi, δ, center)
        @test HQ.master_objective(st) ≈ ref.objective rtol = 1e-6
        @test st.π ≈ ref.π atol = 1e-4
    end

    @testset "inequality rows: hot-started sequence (seed $seed)" for seed in 1:3
        rng = Xoshiro(200 + seed)
        m, P, δ = 25, 4, 1.0
        pool = _random_pool(rng, m, P, P, 400)
        lo, hi = _random_rows(rng, m; ge = 0.4, le = 0.3)
        # seed 3 refactors often, which exercises the rebuild under a row mask
        opts = _qopts(debug = true, refactor_interval = seed == 3 ? 7 : 150,
                      dense_refactor_interval = seed == 3 ? 7 : 0)
        st = _ineq_state(pool, lo, hi, δ; options = opts)
        center = zeros(m)
        for it in 1:60
            HQ.set_center!(st, center)
            @test HQ.qpas_solve!(st) === :optimal
            _check_kkt(st)
            if it % 10 == 0
                ref = _highs_ineq_reference(pool, lo, hi, δ, center)
                @test HQ.master_objective(st) ≈ ref.objective rtol = 1e-6
            end
            center = copy(st.π)                       # proximal center update
            _add_random_column!(rng, pool, rand(rng, 1:P))
            HQ.register_columns!(st)
        end
        @test st.total.slack_expansions > 0 && st.total.slack_reductions > 0
    end

    @testset "inequality rows: dependent columns" begin
        rng = Xoshiro(13)
        m, P, δ = 6, 2, 1.0
        pool = _random_pool(rng, m, P, 8, 100; density = 0.5)
        for _ in 1:30
            src = rand(rng, 1:pool.n)
            rows = collect(HQ.column_rows(pool, src))
            _add_random_column!(rng, pool, rand(rng, 1:P); rows)
        end
        lo, hi = _random_rows(rng, m; ge = 0.5, le = 0.2)
        st = _ineq_state(pool, lo, hi, δ)
        HQ.set_center!(st, randn(rng, m))
        @test HQ.qpas_solve!(st) === :optimal
        _check_kkt(st)
        ref = _highs_ineq_reference(pool, lo, hi, δ, st.center)
        @test HQ.master_objective(st) ≈ ref.objective rtol = 1e-6
    end

    # Releasing the slack of the only row removes the only row of A_eff: H_eff - vvᵀ = 0.
    # One block, rep a₁ = 0, a₂ = ±e₁, c₂ = -0.2. Column 2 is released first; at the
    # resulting subspace minimizer the slack multiplier is -0.2 < 0 and the downdate is
    # singular. The zero-curvature step lowers the representative, which blocks (swap branch).
    #   "≥" row  x₂ ≥ 0.5     (κ = -1)
    #   "≤" row -x₂ ≤ -0.5    (κ = +1, the same constraint written the other way round)
    @testset "singular slack downdate, $(label) row" for (label, lo, hi, a2) in
            [("≥", 0.5, Inf, 1.0), ("≤", -Inf, -0.5, -1.0)]
        pool = HQ.ColumnPool(1, 1, 10)
        HQ.add_column!(pool, 1, 0.0, Int[], Float64[], [1])
        HQ.add_column!(pool, 1, -0.2, [1], [a2], [2])
        st = _ineq_state(pool, [lo], [hi], 1.0)
        @test HQ.qpas_solve!(st) === :optimal
        @test st.total.singular_steps >= 1
        @test st.total.slack_reductions >= 1
        @test st.total.representative_swaps >= 1
        _check_kkt(st)
        @test st.x[2] ≈ 1.0 atol = 1e-12
        @test st.s[1] ≈ 0.5 atol = 1e-12
        @test st.π[1] ≈ 0.0 atol = 1e-12
        ref = _highs_ineq_reference(pool, [lo], [hi], 1.0, zeros(1))
        @test HQ.master_objective(st) ≈ ref.objective atol = 1e-9
    end

    # A column released while its only row has a free slack: the masked column is zero
    # (h_rr = 0), so the zero-curvature step along z₂ (plus the slack) is taken until the
    # slack blocks; the row becomes active and the column then enters N normally.
    # First solve (rep only): the "≤ 0.5" row is slack, s = 0.5, π = 0. Hot start with
    # column 2 (c₂ = -0.2 < c_rep): it is released while its row is masked.
    @testset "released column with a fully masked Hessian column" begin
        pool = HQ.ColumnPool(1, 1, 10)
        HQ.add_column!(pool, 1, 0.0, Int[], Float64[], [1])
        st = _ineq_state(pool, [-Inf], [0.5], 1.0)
        @test HQ.qpas_solve!(st) === :optimal
        @test st.s[1] ≈ 0.5 && HQ.any_free(st.kkt.surplus)
        HQ.add_column!(pool, 1, -0.2, [1], [1.0], [2])
        HQ.register_columns!(st)
        @test HQ.qpas_solve!(st) === :optimal
        @test st.last.singular_steps >= 1
        @test st.last.slack_expansions >= 1
        _check_kkt(st)
        ref = _highs_ineq_reference(pool, [-Inf], [0.5], 1.0, zeros(1))
        @test HQ.master_objective(st) ≈ ref.objective atol = 1e-9
        @test st.π ≈ ref.π atol = 1e-6
    end

    @testset "row classification" begin
        b, rows, signs = HQ.slack_structure([1.0, 2.0, -Inf, 0.0], [1.0, Inf, 3.0, 0.0])
        @test b == [1.0, 2.0, 3.0, 0.0] && rows == [2, 3] && signs == [-1.0, 1.0]
        @test_throws ArgumentError HQ.slack_structure([0.0], [1.0])      # range row
        @test_throws ArgumentError HQ.slack_structure([-Inf], [Inf])     # free row
    end

    @testset "periodic refactorization" begin
        rng = Xoshiro(3)
        pool = _random_pool(rng, 20, 3, 80, 100)
        st = HQ.QPASState(pool, ones(20), 1.0;
                          options = _qopts(refactor_interval = 5, dense_refactor_interval = 5))
        HQ.qpas_solve!(st)
        HQ.set_center!(st, st.π)
        @test HQ.qpas_solve!(st) === :optimal
        @test st.total.refactorizations >= 1
        _check_kkt(st)
        @test HQ.verify_kkt(st.kkt, pool) < 1e-8
    end

    @testset "factor type and timers" begin
        rng = Xoshiro(21)
        pool = _random_pool(rng, 15, 3, 50, 100)
        st = HQ.QPASState(pool, ones(15), 1.0; options = _qopts())
        @test HQ.qpas_solve!(st) === :optimal
        F = st.kkt.factor
        @test HQ.LDLt.backend_name(F) == String(backend)
        @test st isa HQ.QPASState{typeof(F)}
        t = st.kkt.timer
        calls(name) = t.calls[findfirst(==(name), t.names)]
        @test calls(:rowadd) >= HQ.nullspace_dimension(st.kkt.ordering)
        @test calls(:solve) >= 1 && calls(:assemble) >= calls(:rowadd)
        # dense: the factor never stores more than the active block
        if backend === :dense
            @test F.k == HQ.nullspace_dimension(st.kkt.ordering)
            @test HQ.LDLt.factor_stats(F, HQ.items(st.kkt.ordering.N)).stored == F.k * (F.k - 1) ÷ 2
        end
        @test HQ.factor_residual(st.kkt, pool) < 1e-10
        # refactor interval: CHOLMOD default 150, dense never (no structural garbage)
        @test HQ.refactor_interval(st.options) == (backend === :dense ? 0 : 150)
    end
end
