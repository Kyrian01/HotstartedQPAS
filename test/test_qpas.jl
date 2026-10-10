# Tests of the hot-started QPAS in isolation: random multi-block regularized master problems,
# hot-started sequences (new columns + moving center), dependent columns that make the reduced
# Hessian singular, and the debug-mode factor verification.

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

# First-order optimality of the QPAS iterate (independent of the solver's own bookkeeping).
function _check_kkt(st; tol = 1e-7)
    pool = st.pool
    n = pool.n
    x = st.x[1:n]
    @test all(x .>= 0)
    for b in 1:pool.n_blocks
        @test sum(x[j] for j in 1:n if pool.block[j] == b) ≈ 1 atol = 1e-10
    end
    A = Matrix(HQ.column_matrix(pool))
    π = st.center .+ (st.rhs .- A * x) ./ st.delta
    @test π ≈ st.π atol = 1e-9
    g = pool.cost[1:n] .- A' * π
    for j in 1:n
        λj = g[j] - st.μ[pool.block[j]]
        @test λj >= -tol
        x[j] > 1e-9 && @test abs(λj) <= tol
    end
end

@testset "QPAS" begin
    @testset "single solve vs HiGHS (m=$m, P=$P, n=$n, δ=$δ)" for (m, P, n, δ) in
            [(8, 2, 12, 1.0), (20, 4, 60, 0.5), (30, 5, 120, 2.0), (15, 1, 40, 1.0)]
        rng = Xoshiro(m + P + n)
        pool = _random_pool(rng, m, P, n, 200)
        b = ones(m)
        center = randn(rng, m)
        st = HQ.QPASState(pool, b, δ; options = QPASOptions(debug = true))
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
        st = HQ.QPASState(pool, b, δ; options = QPASOptions(debug = true))
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
        st = HQ.QPASState(pool, ones(m), δ; options = QPASOptions(debug = true))
        HQ.set_center!(st, randn(rng, m))
        @test HQ.qpas_solve!(st) === :optimal
        _check_kkt(st)
        ref = _highs_reference(pool, ones(m), δ, st.center)
        @test HQ.master_objective(st) ≈ ref.objective rtol = 1e-6
        # many more columns than rows: N can never exceed rank(A) ≤ m
        @test HQ.nullspace_dimension(st.kkt.ordering) <= m
    end

    # Constructed hot-start cases that force a dependent nullspace column (thesis §4.4.5).
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
        st = HQ.QPASState(pool, b, 1.0; options = QPASOptions(debug = true))
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

    @testset "periodic refactorization" begin
        rng = Xoshiro(3)
        pool = _random_pool(rng, 20, 3, 80, 100)
        st = HQ.QPASState(pool, ones(20), 1.0; options = QPASOptions(refactor_interval = 5))
        HQ.qpas_solve!(st)
        HQ.set_center!(st, st.π)
        @test HQ.qpas_solve!(st) === :optimal
        @test st.total.refactorizations >= 1
        _check_kkt(st)
        @test HQ.verify_kkt(st.kkt, pool) < 1e-8
    end
end
