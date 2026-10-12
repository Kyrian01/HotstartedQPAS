# Tests of the TOML experiment configuration and runner.

const ROOT = dirname(@__DIR__)

function _write_config(text)
    path = tempname() * ".toml"
    write(path, text)
    return path
end

@testset "experiment config" begin
    @testset "shipped experiments.toml" begin
        exps = load_experiments(joinpath(ROOT, "experiments.toml"))
        @test [e.name for e in exps][1:3] == ["I1", "I2", "I3"]
        i3 = exps[3]
        @test i3.instance.n_nodes == 175 && i3.instance.n_pilots == 10
        @test i3.solvers == ["exact", "qpas"]                # LP and HiGHS QPs disabled for I3
        @test exps[1].solvers == ["exact", "lp", "explicit_qp", "eliminated_qp", "qpas"]
        @test i3.cg_options.stabilization.delta == 1.0
        @test i3.cg_options.pricing === :all_blocks
        @test i3.qpas_options.refactor_interval == 150
        @test i3.qpas_options.factorization === :dense
        # the backend comparison pairs generate the same instance and differ only in the factor
        byname = Dict(e.name => e for e in exps)
        for inst in ("I3", "L2")
            c, d = byname["cmp_$(inst)_cholmod"], byname["cmp_$(inst)_dense"]
            @test c.qpas_options.factorization === :cholmod
            @test d.qpas_options.factorization === :dense
            @test (c.instance.n_nodes, c.instance.n_tasks, c.instance.seed) ==
                  (d.instance.n_nodes, d.instance.n_tasks, d.instance.seed)
        end
        @test byname["I3_first_improving"].cg_options.pricing === :first_improving
        @test byname["I3_diag"].cg_options.diagnostics_residual
    end

    @testset "factorization backend" begin
        path = _write_config("""
            [defaults.solvers]
            qpas = true
            [[experiment]]
            name = "c"
            [[experiment]]
            name = "d"
            [experiment.qpas]
            factorization = "dense"
            dense_refactor_interval = 500
            """)
        c, d = load_experiments(path)
        @test c.qpas_options.factorization === :cholmod             # default
        @test d.qpas_options.factorization === :dense
        @test HQ.refactor_interval(d.qpas_options) == 500
        @test HQ.refactor_interval(c.qpas_options) == 150
        bad = _write_config("[[experiment]]\nname = \"x\"\n[experiment.solvers]\nqpas = true\n" *
                            "[experiment.qpas]\nfactorization = \"lapack\"\n")
        @test_throws ArgumentError load_experiments(bad)
        @test_throws ArgumentError QPASMaster(factorization = :lapack)
    end

    @testset "defaults are merged and overridden per experiment" begin
        path = _write_config("""
            [defaults.solvers]
            qpas = true
            [defaults.instance]
            n_nodes = 30
            seed = 7
            [defaults.stabilization]
            delta = 2
            [[experiment]]
            name = "a"
            [[experiment]]
            name = "b"
            [experiment.instance]
            seed = 8
            [experiment.stabilization]
            center_update = "best_bound"
            """)
        a, b = load_experiments(path)
        @test a.instance.n_nodes == b.instance.n_nodes == 30
        @test (a.instance.seed, b.instance.seed) == (7, 8)
        @test a.cg_options.stabilization.delta === 2.0        # integer converted to Float64
        @test b.cg_options.stabilization.center_update === :best_bound
        @test a.instance.name == "a"
    end

    @testset "typos are rejected" begin
        bad_key = _write_config("[[experiment]]\nname = \"x\"\n[experiment.instance]\nn_node = 3\n")
        @test_throws ErrorException load_experiments(bad_key)
        bad_solver = _write_config("[[experiment]]\nname = \"x\"\n[experiment.solvers]\nhighs = true\n")
        @test_throws ErrorException load_experiments(bad_solver)
        no_solver = _write_config("[[experiment]]\nname = \"x\"\n")
        @test_throws ErrorException load_experiments(no_solver)
        bad_value = _write_config("[[experiment]]\nname = \"x\"\n[experiment.solvers]\nqpas = true\n" *
                                  "[experiment.column_generation]\npricing = \"nonsense\"\n")
        @test_throws ArgumentError load_experiments(bad_value)
    end

    @testset "end-to-end run" begin
        out = mktempdir()
        path = _write_config("""
            [defaults.solvers]
            exact = true
            lp = true
            qpas = true
            [defaults.instance]
            n_nodes = 20
            edge_probability = 0.2
            n_tasks = 12
            n_pilots = 3
            [defaults.column_generation]
            diagnostics_every = 50
            [defaults.output]
            directory = "$(escape_string(out))"
            save_instance = true
            [[experiment]]
            name = "tiny_eq"
            [[experiment]]
            name = "tiny_ineq"
            [experiment.instance]
            inequality_fraction = 0.5
            """)
        results = run_experiments(path; io = devnull, warmup = false)
        eq = results["tiny_eq"]
        @test length(eq) == 3
        @test eq[2].objective ≈ eq[1].objective rtol = 1e-9
        @test eq[3].objective ≈ eq[1].objective rtol = 1e-6
        ineq = results["tiny_ineq"]
        @test length(ineq) == 3                              # qpas handles inequality rows
        @test ineq[3].objective ≈ ineq[1].objective rtol = 1e-6
        for f in ("summary.txt", "lp.csv", "qpas.csv", "qpas_diagnostics.csv", "instance.jls")
            @test isfile(joinpath(out, "tiny_eq", f))
        end
        @test countlines(joinpath(out, "summary.csv")) == 1 + 3 + 3
        only = run_experiments(path; only = ["tiny_eq"], io = devnull, warmup = false)
        @test collect(keys(only)) == ["tiny_eq"]
    end
end
