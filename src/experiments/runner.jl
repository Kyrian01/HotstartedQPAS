# Running the experiments of a config file and writing their results.

function _make_master(name::String, e::Experiment)
    name == "lp" && return HighsLPMaster(; big_m = e.big_m)
    name == "explicit_qp" && return HighsExplicitQPMaster()
    name == "eliminated_qp" && return HighsEliminatedQPMaster()
    name == "qpas" && return QPASMaster(e.qpas_options)
    throw(ArgumentError("unknown solver $name"))
end

"""
    run_experiment(experiment; io = stdout) -> Vector

Generate the instance, run every enabled solver, print the comparison table and details,
and write the outputs to `<output.directory>/<name>/`:

- `summary.txt`: comparison table and per-solver details
- `<solver>.csv`: per-iteration traces of every column generation run (if `write_traces`)
- `instance.jls`: the generated instance (if `save_instance`)

Solvers that cannot handle the instance (e.g. inequality rows for the eliminated QP and the
QPAS) are skipped with a message. Returns the list of results.
"""
function run_experiment(e::Experiment; io::IO = stdout)
    dir = joinpath(e.output_dir, e.name)
    mkpath(dir)
    println(io, "═"^100)
    println(io, "experiment \"", e.name, "\"")
    inst = generate_instance(e.instance)
    println(io, "  ", inst)
    e.save_instance && save_instance(joinpath(dir, "instance.jls"), inst)
    flush(io)

    results = Any[]
    reference = nothing
    for solver in e.solvers
        if solver == "exact"
            print(io, "  running exact … ")
            flush(io)
            reference = solve_exact(inst)
            push!(results, reference)
            @printf(io, "%s, %.3f s\n", reference.status, total_time(reference))
            continue
        end
        master = _make_master(solver, e)
        # the eliminated HiGHS QP needs equality rows to eliminate w
        if has_inequalities(inst) && !supports_inequalities(master)
            println(io, "  skipping ", solver, ": inequality rows not supported")
            continue
        end
        print(io, "  running ", solver, " … ")
        e.cg_options.verbose && println(io)
        flush(io)
        r = solve_column_generation(inst, master; options = e.cg_options)
        release!(master)
        push!(results, r)
        @printf(io, "%s, %d iterations, %.3f s\n", r.status, r.stats.iterations, total_time(r))
        flush(io)
        e.write_traces && write_trace_csv(joinpath(dir, "$solver.csv"), r.stats)
        # sampled memory / factor diagnostics, if enabled with diagnostics_every > 0
        e.write_traces && !isempty(r.stats.diagnostics) &&
            write_diagnostics_csv(joinpath(dir, "$(solver)_diagnostics.csv"), r.stats)
    end

    report = sprint() do s
        println(s, inst)
        println(s)
        compare_results(results...; reference, io = s)
        for r in results
            println(s)
            print_details(r; io = s)
        end
    end
    write(joinpath(dir, "summary.txt"), report)
    println(io)
    compare_results(results...; reference, io)
    flush(io)
    return results
end

"""
    run_experiments(path; only = nothing, io = stdout) -> Dict{String,Vector}

Run all experiments of the TOML file `path` (or only those whose names are in `only`). After
every experiment, `<output.directory>/summary.csv` is rewritten with one row per
(experiment, solver), so partial results survive an interrupted run.
"""
function run_experiments(path::AbstractString; only = nothing, io::IO = stdout,
                          warmup::Bool = true)
    experiments = load_experiments(path)
    if only !== nothing
        unknown = setdiff(only, [e.name for e in experiments])
        isempty(unknown) || error("unknown experiment(s) $(unknown) in $path")
        experiments = filter(e -> e.name in only, experiments)
    end
    warmup && _warmup(experiments)

    all_results = Dict{String,Vector}()
    rows = String[]
    for e in experiments
        results = run_experiment(e; io)
        all_results[e.name] = results
        append!(rows, _summary_rows(e, results))
        _write_summary_csv(joinpath(e.output_dir, "summary.csv"), rows)
    end
    return all_results
end

# Run every solver once on a tiny instance so that JIT compilation is not in the timings.
function _warmup(experiments)
    tiny = generate_instance(InstanceConfig(n_nodes = 20, edge_probability = 0.2, n_tasks = 12,
                                            n_pilots = 3, name = "warmup"))
    solve_exact(tiny)
    for name in unique(Iterators.flatten(e.solvers for e in experiments))
        name == "exact" && continue
        solve_column_generation(tiny, _make_master(name, experiments[1]))
    end
    return nothing
end

const _SUMMARY_HEADER = "experiment,method,status,iterations,total_time,avg_master_ms," *
                        "avg_pricing_ms,inner_iterations,columns,objective,gap,infeasibility"

function _summary_rows(e::Experiment, results)
    ref = findfirst(r -> r isa ExactResult, results)
    refobj = ref === nothing ? NaN : objective_value(results[ref])
    rows = String[]
    for r in results
        cols = r isa CGResult ? n_columns(r) : 0
        push!(rows, join(Any[e.name, "\"$(method_name(r))\"", _status(r), _iterations(r),
                             total_time(r), _avg_master_ms(r), _avg_pricing_ms(r),
                             _inner_iterations(r), cols, objective_value(r),
                             objective_value(r) - refobj, _infeasibility(r)], ','))
    end
    return rows
end

function _write_summary_csv(path, rows)
    open(path, "w") do f
        println(f, _SUMMARY_HEADER)
        foreach(r -> println(f, r), rows)
    end
    return path
end
