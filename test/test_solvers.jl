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
                   QPASMaster()]
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

    @testset "QPAS debug mode through a full CG run" begin
        inst = test_instance(:tiny; seed = 4)
        r = solve_column_generation(inst, QPASMaster(debug = true); options = opts)
        @test r.objective ≈ solve_exact(inst).objective rtol = 1e-6
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
        for o in variants, master in (QPASMaster(), HighsExplicitQPMaster())
            r = solve_column_generation(inst, master; options = o)
            @test r.status === :optimal
            @test r.objective ≈ ref rtol = 1e-6
        end
    end

    @testset "mixed equality / inequality rows" begin
        inst = test_instance(:small; seed = 3, inequality_fraction = 0.4)
        @test has_inequalities(inst)
        exact = solve_exact(inst)
        @test exact.status === :optimal
        for master in (HighsLPMaster(), HighsExplicitQPMaster())
            r = solve_column_generation(inst, master; options = opts)
            @test r.status === :optimal
            @test r.objective ≈ exact.objective rtol = 1e-6
            _check_flows(inst, edge_flows(r, inst); tol = 1e-5)
        end
        @test_throws ArgumentError solve_column_generation(inst, HighsEliminatedQPMaster())
        @test_throws ArgumentError solve_column_generation(inst, QPASMaster())
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
