# End-to-end tests: every column generation variant must reproduce the exact LP optimum.

function _check_flows(inst, flows; tol = 1e-6)
    F = HQ.incidence_matrix(inst.network)
    b = HQ.supply_vector(inst.network)
    for k in 1:n_pilots(inst)
        @test norm(F * flows[:, k] - b, Inf) <= tol
    end
    cover = inst.task_matrix * vec(sum(flows; dims = 2))
    @test all(cover .>= inst.task_lower .- tol)
    @test all(cover .<= inst.task_upper .+ tol)
    @test all(flows .>= -tol) && all(flows .<= 1 + tol)
end

@testset "solvers end-to-end" begin
    opts = CGOptions(max_columns = 2_000)

    @testset "equality instance $(preset) seed $(seed)" for (preset, seed) in
            [(:tiny, 1), (:tiny, 2), (:small, 1)]
        inst = test_instance(preset; seed)
        exact = solve_exact(inst)
        @test exact.status === :optimal
        _check_flows(inst, exact.flows)
        ref = exact.objective

        masters = [HighsLPMaster(), HighsExplicitQPMaster(), HighsEliminatedQPMaster(),
                   QPASMaster(), QPASMaster(factorization = :dense)]
        for master in masters
            r = solve_column_generation(inst, master; options = opts)
            @test r.status === :optimal
            rtol = master isa HighsLPMaster ? 1e-9 : 1e-6
            @test r.objective ≈ ref rtol = rtol
            @test r.lower_bound <= ref + 1e-6
            @test r.lower_bound ≈ ref rtol = 1e-5
            @test r.infeasibility <= 1e-5
            flows = edge_flows(r, inst)
            @test dot(vec(inst.edge_costs), vec(flows)) ≈ r.objective
            _check_flows(inst, flows; tol = 1e-5)
            s = r.stats
            @test length(s.master_time) == s.iterations == length(s.operational_cost)
            @test s.duplicate_columns == 0
        end
    end

    @testset "QPAS debug mode through a full CG run ($f)" for f in HQ.FACTORIZATIONS
        inst = test_instance(:tiny; seed = 4)
        r = solve_column_generation(inst, QPASMaster(debug = true, factorization = f);
                                    options = opts)
        @test r.objective ≈ solve_exact(inst).objective rtol = 1e-6
    end

    # The two backends factor the same matrices; the runs agree up to rounding (iteration
    # paths may differ slightly in degenerate ties) and both reach the exact optimum.
    @testset "CHOLMOD and dense backends agree (seed $seed, ineq $frac)" for (seed, frac) in
            [(1, 0.0), (2, 0.0), (3, 0.0), (1, 0.4), (2, 1.0)]
        inst = test_instance(:small; seed, inequality_fraction = frac)
        ref = solve_exact(inst).objective
        rc = solve_column_generation(inst, QPASMaster(factorization = :cholmod); options = opts)
        rd = solve_column_generation(inst, QPASMaster(factorization = :dense); options = opts)
        @test rc.status === rd.status === :optimal
        @test rd.objective ≈ ref rtol = 1e-6
        @test rd.objective ≈ rc.objective rtol = 1e-7
        @test rd.lower_bound <= ref + 1e-6
        @test rd.infeasibility <= 1e-5
        @test abs(rd.stats.iterations - rc.stats.iterations) <= 0.1 * rc.stats.iterations
        _check_flows(inst, edge_flows(rd, inst); tol = 1e-5)
        @test method_name(rd) == "Hot-start QPAS CG (dense)"
    end

    @testset "pricing and stabilization variants" begin
        inst = test_instance(:small; seed = 2)
        ref = solve_exact(inst).objective
        variants = [
            CGOptions(max_columns = 2_000, pricing = :all_blocks),
            CGOptions(max_columns = 2_000, compute_bound = true),
            CGOptions(max_columns = 2_000,
                      stabilization = StabilizationOptions(delta = 5.0, center_update = :best_bound)),
            CGOptions(max_columns = 2_000, stabilization = StabilizationOptions(delta = 0.1)),
        ]
        for o in variants,
            master in (QPASMaster(), QPASMaster(factorization = :dense), HighsExplicitQPMaster())
            r = solve_column_generation(inst, master; options = o)
            @test r.status === :optimal
            @test r.objective ≈ ref rtol = 1e-6
        end
    end

    @testset "mixed equality / inequality rows (fraction $(frac), seed $(seed))" for
            (frac, seed) in [(0.4, 3), (0.5, 1), (1.0, 2)]
        inst = test_instance(:small; seed, inequality_fraction = frac)
        @test has_inequalities(inst)
        exact = solve_exact(inst)
        @test exact.status === :optimal
        for master in (HighsLPMaster(), HighsExplicitQPMaster(), HighsEliminatedQPMaster(),
                       QPASMaster(), QPASMaster(factorization = :dense))
            r = solve_column_generation(inst, master; options = opts)
            @test r.status === :optimal
            @test r.objective ≈ exact.objective rtol = 1e-6
            @test r.lower_bound <= exact.objective + 1e-6
            @test r.infeasibility <= 1e-5
            _check_flows(inst, edge_flows(r, inst); tol = 1e-5)
        end
    end

    # "≤" rows are handled by a slack with the opposite sign; build an instance with
    # "cover ≤ 1", "cover ≥ 1" and "cover = 1" rows from a generated one.
    @testset "≤ rows" begin
        base = test_instance(:small; seed = 5)
        T = n_tasks(base)
        lo, hi = copy(base.task_lower), copy(base.task_upper)
        lo[1:3:T] .= -Inf                                     # cover ≤ 1
        hi[2:3:T] .= Inf                                      # cover ≥ 1
        inst = HQ.CrewSchedulingInstance(base.network, base.edge_costs, base.task_matrix,
                                         lo, hi, "le_rows")
        exact = solve_exact(inst)
        @test exact.status === :optimal
        for master in (HighsLPMaster(), HighsEliminatedQPMaster(), QPASMaster(), QPASMaster(debug = true),
                       QPASMaster(factorization = :dense),
                       QPASMaster(factorization = :dense, debug = true))
            r = solve_column_generation(inst, master; options = opts)
            @test r.status === :optimal
            @test r.objective ≈ exact.objective rtol = 1e-6
            @test r.lower_bound <= exact.objective + 1e-6
            _check_flows(inst, edge_flows(r, inst); tol = 1e-5)
        end
    end

    # range rows ℓ ≤ cover ≤ u: one slack with an upper bound (bound flips); "cover ∈ [0, 1]"
    # (optional flight) and "cover ∈ [1, 2]" so that both bounds become active
    @testset "range rows" begin
        base = test_instance(:small; seed = 5)
        T = n_tasks(base)
        lo, hi = copy(base.task_lower), copy(base.task_upper)
        lo[1:3:T] .= 0.0                                      # cover ∈ [0, 1]
        hi[2:3:T] .= 2.0                                      # cover ∈ [1, 2]
        inst = HQ.CrewSchedulingInstance(base.network, base.edge_costs, base.task_matrix,
                                         lo, hi, "range_rows")
        exact = solve_exact(inst)
        @test exact.status === :optimal
        for master in (HighsLPMaster(), HighsExplicitQPMaster(), HighsEliminatedQPMaster(),
                       QPASMaster(), QPASMaster(debug = true),
                       QPASMaster(factorization = :dense),
                       QPASMaster(factorization = :dense, debug = true))
            r = solve_column_generation(inst, master; options = opts)
            @test r.status === :optimal
            @test r.objective ≈ exact.objective rtol = 1e-6
            @test r.lower_bound <= exact.objective + 1e-6
            @test r.infeasibility <= 1e-5
            _check_flows(inst, edge_flows(r, inst); tol = 1e-5)
        end
    end

    @testset "inequality rows: pricing and stabilization variants" begin
        inst = test_instance(:small; seed = 4, inequality_fraction = 0.5)
        ref = solve_exact(inst).objective
        variants = [
            CGOptions(max_columns = 2_000, pricing = :all_blocks),
            CGOptions(max_columns = 2_000, compute_bound = true),
            CGOptions(max_columns = 2_000,
                      stabilization = StabilizationOptions(delta = 5.0, center_update = :best_bound)),
            CGOptions(max_columns = 2_000, stabilization = StabilizationOptions(delta = 0.1)),
        ]
        for o in variants, f in HQ.FACTORIZATIONS
            r = solve_column_generation(inst, QPASMaster(factorization = f); options = o)
            @test r.status === :optimal
            @test r.objective ≈ ref rtol = 1e-6
            @test r.lower_bound <= ref + 1e-6
        end
        for f in HQ.FACTORIZATIONS
            r = solve_column_generation(inst, QPASMaster(debug = true, factorization = f);
                                        options = opts)
            @test r.objective ≈ ref rtol = 1e-6
        end
    end

    @testset "diagnostics traces ($f)" for f in HQ.FACTORIZATIONS
        inst = test_instance(:small; seed = 1, inequality_fraction = 0.3)
        o = CGOptions(max_columns = 2_000, diagnostics_every = 25, diagnostics_residual = true)
        r = solve_column_generation(inst, QPASMaster(factorization = f); options = o)
        d = r.stats.diagnostics
        @test length(d) == r.stats.iterations ÷ 25 + (r.stats.iterations % 25 == 0 ? 0 : 1)
        @test d[end].iteration == r.stats.iterations              # final snapshot
        @test all(x -> x.factor_stored >= x.factor_effective, d)
        @test all(x -> x.factor_effective <= x.factor_dense, d)
        @test all(x -> x.reach >= 0 && x.peak_rss_MB > 0 && x.factor_MB > 0, d)
        @test all(x -> x.factor_residual < 1e-9, d)              # no drift between rebuilds
        @test all(x -> x.n_solve > 0 && x.t_solve > 0, d[2:end])
        # dense: exactly the triangle of the active block is stored (its zeros are true zeros
        # of L, not entries left behind by deletions)
        f === :dense && @test all(x -> x.factor_stored == x.factor_dense, d)
        path = write_diagnostics_csv(tempname() * ".csv", r.stats)
        @test countlines(path) == length(d) + 1
        # masters without internals still get the memory columns
        r2 = solve_column_generation(inst, HighsLPMaster(); options = o)
        @test haskey(r2.stats.diagnostics[end], :julia_heap_MB)
        @test !haskey(r2.stats.diagnostics[end], :factor_stored)
        # off by default
        @test isempty(solve_column_generation(inst, QPASMaster()).stats.diagnostics)
    end

    @testset "limits and reporting" begin
        inst = test_instance(:small; seed = 1)
        r = solve_column_generation(inst, HighsLPMaster(); options = CGOptions(max_iterations = 5))
        @test r.status === :iteration_limit && r.stats.iterations == 5
        r = solve_column_generation(inst, QPASMaster(); options = CGOptions(max_columns = 20))
        @test r.status === :column_limit
        path = write_trace_csv(tempname() * ".csv", r.stats)
        @test countlines(path) == r.stats.iterations + 1
        io = IOBuffer()
        compare_results(solve_exact(inst), r; reference = solve_exact(inst), io)
        print_details(r; io)
        @test occursin("Hot-start QPAS", String(take!(io)))
    end
end
