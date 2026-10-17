# Adaptive stabilization (docs/adaptive_stabilization.md): changing δ between hot-started
# solves, and complete column generation runs with the geometric δ rule.

@testset "adaptive stabilization: QPAS ($backend)" for backend in HQ.FACTORIZATIONS
    QPAS_BACKEND[] = backend

    @testset "changing δ keeps the factor and gives the δ' optimum (δ' = $δ2)" for δ2 in
            (0.25, 4.0)
        rng = Xoshiro(3)
        m, P = 20, 4
        pool = _random_pool(rng, m, P, 60, 200)
        b = ones(m)
        center = randn(rng, m)
        st = HQ.QPASState(pool, b, 1.0; options = _qopts(debug = true))
        HQ.set_center!(st, center)
        @test HQ.qpas_solve!(st) === :optimal
        updates = st.kkt.total_updates
        N = sort(HQ.items(st.kkt.ordering.N))
        HQ.set_delta!(st, δ2)
        @test st.kkt.total_updates == updates                 # no factor operation
        @test sort(HQ.items(st.kkt.ordering.N)) == N
        @test HQ.qpas_solve!(st) === :optimal                 # hot start from the δ = 1 optimum
        _check_kkt(st)
        ref = _highs_reference(pool, b, δ2, center)
        @test HQ.master_objective(st) ≈ ref.objective rtol = 1e-6
        @test st.π ≈ ref.π atol = 1e-4
        @test_throws ArgumentError HQ.set_delta!(st, 0.0)
        @test_throws ArgumentError HQ.set_delta!(st, Inf)
    end
end

@testset "adaptive stabilization: column generation" begin
    variants = [
        StabilizationOptions(delta_update = :geometric),
        StabilizationOptions(delta_update = :geometric, delta_increase = Inf, delta_min = 1e-3),
        StabilizationOptions(delta_update = :geometric, center_update = :best_bound),
    ]
    @testset "$(preset) seed $(seed), inequality fraction $(frac)" for (preset, seed, frac) in
            [(:tiny, 1, 0.0), (:small, 1, 0.0), (:small, 2, 0.0), (:small, 3, 0.4),
             (:small, 1, 1.0)]
        inst = test_instance(preset; seed, inequality_fraction = frac)
        ref = solve_exact(inst).objective
        for stab in variants, f in HQ.FACTORIZATIONS
            r = solve_column_generation(inst, QPASMaster(factorization = f);
                                        options = CGOptions(max_columns = 2_000,
                                                            stabilization = stab))
            @test r.status === :optimal
            @test r.objective ≈ ref rtol = 1e-6
            @test r.lower_bound <= ref + 1e-6
            @test r.infeasibility <= 1e-5
            _check_flows(inst, edge_flows(r, inst); tol = 1e-5)
            s = r.stats
            @test length(s.delta) == s.iterations
            @test all(stab.delta_min - 1e-15 .<= s.delta .<= stab.delta)
            @test minimum(s.delta) < stab.delta                  # δ was decreased at some point
        end
    end

    @testset "options and unsupported masters" begin
        inst = test_instance(:tiny; seed = 1)
        geo = CGOptions(stabilization = StabilizationOptions(delta_update = :geometric))
        @test_throws ArgumentError solve_column_generation(inst, HighsExplicitQPMaster(); options = geo)
        @test_throws ArgumentError solve_column_generation(inst, HighsEliminatedQPMaster(); options = geo)
        # the LP master is unstabilized: the stabilization options do not apply
        @test solve_column_generation(inst, HighsLPMaster(); options = geo).status === :optimal
        bad = [StabilizationOptions(delta_update = :linear),
               StabilizationOptions(delta_update = :geometric, delta_decrease = 0.0),
               StabilizationOptions(delta_update = :geometric, delta_increase = 0.5),
               StabilizationOptions(delta = 1.0, delta_min = 2.0)]
        for s in bad
            @test_throws ArgumentError HQ.validate(CGOptions(stabilization = s))
        end
        # traces carry δ
        r = solve_column_generation(inst, QPASMaster(); options = geo)
        path = write_trace_csv(tempname() * ".csv", r.stats)
        @test endswith(readline(path), ",center_updated,delta")
        @test countlines(path) == r.stats.iterations + 1
    end
end
