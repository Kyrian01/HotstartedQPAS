# Tests of the problem layer: network, generator, pricing, column pool.

@testset "problem data" begin
    @testset "network" begin
        net = FlowNetwork(4, [1, 1, 2, 3, 2], [2, 3, 3, 4, 4])
        @test n_edges(net) == 5
        @test sort(collect(HQ.out_edges(net, 2))) == [3, 5]
        F = HQ.incidence_matrix(net)
        @test F * [1, 0, 1, 1, 0] == HQ.supply_vector(net)      # path 1→2→3→4
        @test_throws ArgumentError FlowNetwork(3, [2], [1])     # not topological
    end

    @testset "generator" begin
        inst = test_instance(:tiny; seed = 3)
        @test n_tasks(inst) > 0
        @test size(inst.edge_costs) == (n_edges(inst), n_pilots(inst))
        @test !has_inequalities(inst)
        @test all(inst.task_matrix.nzval .== 1)
        # equality instances are feasible by construction
        @test solve_exact(inst).status === :optimal
        # deterministic for a fixed seed
        inst2 = test_instance(:tiny; seed = 3)
        @test inst2.task_matrix == inst.task_matrix && inst2.edge_costs == inst.edge_costs

        mixed = test_instance(:tiny; seed = 3, inequality_fraction = 0.5)
        @test has_inequalities(mixed)
        @test count(isinf, mixed.task_upper) == round(Int, 0.5 * n_tasks(mixed))

        path = tempname() * ".jls"
        save_instance(path, inst)
        @test load_instance(path).task_matrix == inst.task_matrix
    end

    @testset "pricing = brute-force shortest path" begin
        inst = test_instance(:tiny; seed = 5)
        pricer = HQ.ShortestPathPricer(inst)
        π = randn(Xoshiro(1), n_tasks(inst))
        HQ.set_duals!(pricer, π)
        reduced = inst.edge_costs .- inst.task_matrix' * π
        F = HQ.incidence_matrix(inst.network)
        b = HQ.supply_vector(inst.network)
        for k in 1:n_pilots(inst)
            value, cost = HQ.price_block!(pricer, k)
            x = zeros(n_edges(inst))
            x[pricer.path] .= 1
            @test F * x == b                                   # valid path
            @test value ≈ dot(reduced[:, k], x)
            @test cost ≈ dot(inst.edge_costs[:, k], x)
            # LP over the flow polytope gives the same optimum (DAG ⇒ integral)
            model = HQ.HighsBackend.HighsModel()
            E = n_edges(inst)
            HQ.HighsBackend.pass_lp!(model, F, Vector(reduced[:, k]), zeros(E), ones(E), b, b)
            HQ.HighsBackend.run!(model)
            @test HQ.HighsBackend.model_objective(model) ≈ value atol = 1e-8
            rows, vals = HQ.path_coverage!(pricer)
            @test sparsevec(rows, vals, n_tasks(inst)) ≈ inst.task_matrix * x
        end
    end

    @testset "column pool" begin
        pool = HQ.ColumnPool(4, 2, 5)
        HQ.add_column!(pool, 1, 3.0, [1, 3], [1.0, 2.0], [7, 8])
        HQ.add_column!(pool, 2, 5.0, [2], [1.0], [9])
        @test HQ.contains_column(pool, 1, [7, 8])
        @test !HQ.contains_column(pool, 2, [7, 8])
        @test Matrix(HQ.column_matrix(pool)) == [1 0; 0 1; 2 0; 0 0]
        y = zeros(4)
        HQ.coupling_product!(y, pool, [2.0, 3.0, 0, 0, 0])
        @test y == [2, 3, 4, 0]
        z = zeros(5)
        HQ.coupling_transpose_product!(z, pool, [1.0, 1.0, 1.0, 1.0])
        @test z[1:2] == [3, 1]
    end
end
