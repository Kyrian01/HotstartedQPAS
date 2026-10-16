# Benchmark of the LDLᵀ backends of the QPAS: CHOLMOD (sparse, n_max × n_max) vs dense (|N| × |N|).
#
#   julia --project scripts/benchmark_factorization.jl                 # small + large + micro
#   julia --project scripts/benchmark_factorization.jl --suite small   # small | large | micro | all | report
#   julia --project scripts/benchmark_factorization.jl --reps 3 --out results/factorization
#   julia --project scripts/benchmark_factorization.jl --only I3,L2    # instance selection
#
# small suite: I1–I3 and I2_ineq, both pricing rules, checked against the exact LP optimum.
# large suite: larger generated instances (no exact LP; the backends are compared with each
#              other: same status, same objective up to the stabilization tolerance).
# micro:       time of every factor operation as a function of the active size k.
#
# Every CG configuration is run `reps` times per backend (timings: minimum and median), plus one
# diagnostics run per backend that samples memory, factor contents and the factor residual
# ‖LDLᵀ − H_eff‖ about 60 times along the run. BLAS uses one thread (CHOLMOD's simplicial
# factor is single-threaded; multi-threaded BLAS gave no gain at these sizes).
#
# Output (in --out): runs.csv (one row per run), diag_<instance>_<pricing>_<backend>.csv,
# micro.csv, and report.md (the tables that docs/dense_ldlt_backend.md quotes).

using HotstartedQPAS
using LinearAlgebra
using Printf
using Random
using Statistics

const HQ = HotstartedQPAS
const LDLt = HQ.LDLt

BLAS.set_num_threads(1)

# ── instances ────────────────────────────────────────────────────────────────────────────
const SMALL = [
    (name = "I1", n_nodes = 50, edge_probability = 0.098, n_tasks = 40, n_pilots = 4, ineq = 0.0),
    (name = "I2", n_nodes = 150, edge_probability = 0.09, n_tasks = 230, n_pilots = 7, ineq = 0.0),
    (name = "I3", n_nodes = 175, edge_probability = 0.091, n_tasks = 390, n_pilots = 10, ineq = 0.0),
    (name = "I2_ineq", n_nodes = 150, edge_probability = 0.09, n_tasks = 230, n_pilots = 15, ineq = 0.5),
]
const LARGE = [
    (name = "L1", n_nodes = 225, edge_probability = 0.09, n_tasks = 500, n_pilots = 13, ineq = 0.0),
    (name = "L2", n_nodes = 300, edge_probability = 0.08, n_tasks = 800, n_pilots = 18, ineq = 0.0),
    (name = "L3", n_nodes = 400, edge_probability = 0.07, n_tasks = 1200, n_pilots = 24, ineq = 0.0),
    (name = "L2_ineq", n_nodes = 300, edge_probability = 0.08, n_tasks = 800, n_pilots = 18, ineq = 0.3),
]
const BACKENDS = (:cholmod, :dense)

function parse_args(args)
    opts = Dict{String,Any}("suite" => "all", "reps" => 3, "out" => "results/factorization",
                            "only" => nothing, "time_limit" => 3600.0)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--suite"
            opts["suite"] = args[i += 1]
        elseif a == "--reps"
            opts["reps"] = parse(Int, args[i += 1])
        elseif a == "--out"
            opts["out"] = args[i += 1]
        elseif a == "--only"
            opts["only"] = split(args[i += 1], ',')
        elseif a == "--time-limit"
            opts["time_limit"] = parse(Float64, args[i += 1])
        else
            error("unknown argument $a")
        end
        i += 1
    end
    return opts
end

make_instance(s) = generate_instance(InstanceConfig(; n_nodes = s.n_nodes,
                                                    edge_probability = s.edge_probability,
                                                    n_tasks = s.n_tasks, n_pilots = s.n_pilots,
                                                    inequality_fraction = s.ineq, name = s.name))

# ── one CG run → one row ─────────────────────────────────────────────────────────────────
function run_cg(inst, backend, pricing; time_limit, diagnostics_every = 0)
    master = QPASMaster(factorization = backend)
    o = CGOptions(pricing = pricing, time_limit = time_limit,
                  diagnostics_every = diagnostics_every,
                  diagnostics_residual = diagnostics_every > 0)
    GC.gc()
    timed = @timed solve_column_generation(inst, master; options = o)
    r = timed.value
    st = master.state
    row = _row(inst, backend, pricing, r, st, timed)
    HQ.release!(master)
    return r, row
end

function _row(inst, backend, pricing, r, st, timed)
    s = r.stats
    c = st.total
    t = st.timer
    ft = st.kkt.timer
    dg = HQ.qpas_diagnostics(st)
    sec(tm, name) = tm.seconds[findfirst(==(name), tm.names)]
    calls(tm, name) = tm.calls[findfirst(==(name), tm.names)]
    row = Dict{String,Any}(
        "instance" => inst.name, "m" => n_tasks(inst), "edges" => n_edges(inst),
        "pilots" => n_pilots(inst), "ineq_rows" => count(i -> inst.task_lower[i] != inst.task_upper[i], 1:n_tasks(inst)),
        "pricing" => String(pricing), "backend" => String(backend),
        "status" => String(r.status), "iterations" => s.iterations, "columns" => HQ.n_columns(r),
        "total_s" => s.total_time, "master_s" => sum(s.master_time), "pricing_s" => sum(s.pricing_time),
        "inner_iterations" => sum(s.inner_iterations),
        "objective" => r.objective, "master_objective" => s.master_objective[end],
        "lower_bound" => r.lower_bound, "infeasibility" => r.infeasibility,
        "final_N" => HQ.nullspace_dimension(st.kkt.ordering),
        "refactorizations" => c.refactorizations, "factor_updates" => c.factor_updates,
        "expansions" => c.expansions, "reductions" => c.reductions,
        "singular_steps" => c.singular_steps,
        "allocated_MB" => timed.bytes / 2^20, "gc_s" => timed.gctime,
        "factor_MB" => dg.factor_MB, "factor_peak_MB" => dg.factor_peak_MB,
        "factor_stored" => dg.factor_stored, "factor_effective" => dg.factor_effective,
        "factor_dense" => dg.factor_dense)
    for name in t.names
        row["t_$(name)"] = sec(t, name)
    end
    for name in ft.names
        row["op_$(name)_s"] = sec(ft, name)
        row["op_$(name)_calls"] = calls(ft, name)
    end
    for (k, v) in pairs(LDLt.backend_stats(st.kkt.factor))
        row["dense_$(k)"] = v
    end
    return row
end

const COLUMNS = ["instance", "m", "edges", "pilots", "ineq_rows", "pricing", "backend", "rep",
                 "status", "iterations", "columns", "total_s", "master_s", "pricing_s",
                 "inner_iterations", "objective", "master_objective", "lower_bound",
                 "infeasibility", "exact_objective", "gap_exact", "diff_cholmod", "final_N",
                 "refactorizations", "factor_updates", "expansions", "reductions",
                 "singular_steps", "allocated_MB", "gc_s", "factor_MB", "factor_peak_MB",
                 "factor_stored", "factor_effective", "factor_dense",
                 "t_direction", "t_multipliers", "t_step", "t_expand", "t_reduce", "t_refactor",
                 "op_assemble_s", "op_assemble_calls", "op_rowadd_s", "op_rowadd_calls",
                 "op_rowdel_s", "op_rowdel_calls", "op_updown_s", "op_updown_calls",
                 "op_solve_s", "op_solve_calls", "dense_capacity", "dense_peak_active",
                 "dense_grows", "dense_deletions", "dense_mean_trailing_fraction"]

function write_rows(path, rows)
    open(path, "w") do f
        println(f, join(COLUMNS, ','))
        for r in rows
            println(f, join((let v = get(r, c, "")
                                 v isa AbstractString ? "\"$v\"" : string(v)
                             end for c in COLUMNS), ','))
        end
    end
end

function write_diag(path, stats)
    write_diagnostics_csv(path, stats)
end

# ── CG suites ────────────────────────────────────────────────────────────────────────────
function run_suite(specs, pricings, opts, rows, report; exact::Bool)
    out = opts["out"]
    for spec in specs
        opts["only"] === nothing || spec.name in opts["only"] || continue
        inst = make_instance(spec)
        println("═"^100)
        println(inst)
        exact_obj = NaN
        if exact
            ex = solve_exact(inst)
            exact_obj = ex.objective
            @printf("  exact LP: %s, objective %.9f, %.2f s\n", ex.status, ex.objective, total_time(ex))
        end
        for pricing in pricings
            results = Dict{Symbol,Vector{Dict{String,Any}}}()
            for backend in BACKENDS
                results[backend] = Dict{String,Any}[]
                for rep in 1:opts["reps"]
                    r, row = run_cg(inst, backend, pricing; time_limit = opts["time_limit"])
                    row["rep"] = rep
                    row["exact_objective"] = exact_obj
                    row["gap_exact"] = r.objective - exact_obj
                    push!(results[backend], row)
                    @printf("  %-15s %-7s rep %d: %-10s it %6d  total %8.3f s  master %8.3f s  obj %.9f  gap %.2e\n",
                            pricing, backend, rep, r.status, r.stats.iterations, r.stats.total_time,
                            sum(r.stats.master_time), r.objective, r.objective - exact_obj)
                    flush(stdout)
                    # a run that hit the time limit is not repeated
                    r.status === :time_limit && break
                end
            end
            ref = results[:cholmod][1]["objective"]
            for backend in BACKENDS, row in results[backend]
                row["diff_cholmod"] = row["objective"] - ref
                push!(rows, row)
            end
            write_rows(joinpath(out, "runs.csv"), rows)

            # one diagnostics run per backend: ~60 samples, with factor residual
            iters = results[:cholmod][1]["iterations"]
            every = max(1, iters ÷ 60)
            for backend in BACKENDS
                r, _ = run_cg(inst, backend, pricing; time_limit = opts["time_limit"],
                              diagnostics_every = every)
                write_diag(joinpath(out, "diag_$(spec.name)_$(pricing)_$(backend).csv"), r.stats)
                d = r.stats.diagnostics
                @printf("  diagnostics %-7s: %d samples, max residual %.2e, max |N| %d, factor peak %.2f MB\n",
                        backend, length(d), maximum(x -> x.factor_residual, d),
                        maximum(x -> x.N, d), maximum(x -> x.factor_peak_MB, d))
                push!(report, (instance = spec.name, pricing = pricing, backend = backend,
                               max_residual = maximum(x -> x.factor_residual, d),
                               final_residual = d[end].factor_residual,
                               max_N = maximum(x -> x.N, d),
                               max_waste = maximum(x -> x.factor_waste, d),
                               mean_waste = mean(x -> x.factor_waste, d),
                               min_fill = minimum(x -> x.factor_fill, d[2:end]),
                               max_stored = maximum(x -> x.factor_stored, d),
                               max_dense = maximum(x -> x.factor_dense, d),
                               peak_MB = maximum(x -> x.factor_peak_MB, d)))
            end
        end
    end
end

# ── micro benchmark of the factor operations ─────────────────────────────────────────────
# Gram matrix of k random sparse 0/±1 "nullspace columns" with m = k rows, like H_eff.
function micro(out; sizes = (50, 100, 200, 350, 500, 750, 1000, 1500, 2000), n_max = 20_000)
    rows = String["k,backend,op,median_us,min_us,samples"]
    rng = Xoshiro(1)
    for k in sizes
        m = k + 10
        U = Float64.(rand(rng, m, k + 1) .< 0.08) .- Float64.(rand(rng, m, k + 1) .< 0.04)
        H = U' * U + 1e-3I
        idx = collect(1:(k + 1))            # pool indices of the k+1 columns
        for backend in BACKENDS
            F = HQ.make_factor(backend, n_max, m)
            # activate 1:k in order (both backends append)
            for j in 1:k
                LDLt.rowadd!(F, idx[j], idx[1:j], H[1:j, j], j)
            end
            samples = Dict{String,Vector{Float64}}()
            rec(op, t) = push!(get!(samples, op, Float64[]), t * 1e-3)
            reps = k <= 500 ? 60 : 15
            for rep in 1:reps
                # append one more index and remove it again
                j = k + 1
                t0 = time_ns(); LDLt.rowadd!(F, idx[j], idx[1:j], H[1:j, j], j); rec("rowadd_append", time_ns() - t0)
                t0 = time_ns(); LDLt.rowdel!(F, idx[j]); rec("rowdel_last", time_ns() - t0)
                # delete the first / a middle position of the factor and re-activate that index
                # (CHOLMOD: natural order, so position = index; dense: slot order, and the
                # re-activated index is appended, so the victim is chosen by slot each time)
                for (label, pos) in (("first", 1), ("middle", k ÷ 2))
                    q = backend === :dense ? F.index[pos] : idx[pos]
                    t0 = time_ns(); LDLt.rowdel!(F, idx[q]); rec("rowdel_$(label)", time_ns() - t0)
                    active = [i for i in 1:k if i != q]
                    t0 = time_ns(); LDLt.rowadd!(F, idx[q], vcat(active, q), vcat(H[active, q], H[q, q]), k)
                    rec("rowadd_reinsert_$(label)", time_ns() - t0)
                end
                # rank-one update + downdate with a dense v
                v = 0.1 .* randn(rng, k)
                t0 = time_ns(); LDLt.updown!(F, true, idx[1:k], v, k); rec("update", time_ns() - t0)
                t0 = time_ns(); LDLt.updown!(F, false, idx[1:k], v, k); rec("downdate", time_ns() - t0)
                # sparse-pattern solve on the active set
                rhs = LDLt.rhs_buffer(F)
                rhs[idx[1:k]] .= randn(rng, k)
                t0 = time_ns(); LDLt.solve!(F, idx[1:k], k); rec("solve", time_ns() - t0)
                rhs[idx[1:k]] .= 0.0
            end
            # one full rebuild (reset + k appends)
            t0 = time_ns()
            LDLt.reset!(F)
            for j in 1:k
                LDLt.rowadd!(F, idx[j], idx[1:j], H[1:j, j], j)
            end
            rec("rebuild", time_ns() - t0)
            for (op, v) in sort(collect(samples))
                push!(rows, @sprintf("%d,%s,%s,%.3f,%.3f,%d", k, backend, op, median(v), minimum(v), length(v)))
            end
            # memory of the factor with k active indices (MB, in the median column)
            fs = LDLt.factor_stats(F, idx[1:k])
            push!(rows, @sprintf("%d,%s,memory_MB,%.4f,%.4f,1", k, backend, fs.memory / 2^20, fs.memory / 2^20))
            close(F)
            @printf("  micro k=%5d %-7s solve %8.1f µs  append %8.1f µs  del-first %8.1f µs  update %8.1f µs\n",
                    k, backend, median(samples["solve"]), median(samples["rowadd_append"]),
                    median(samples["rowdel_first"]), median(samples["update"]))
            flush(stdout)
        end
    end
    write(joinpath(out, "micro.csv"), join(rows, '\n') * '\n')
end

# ── report (rebuilt from the CSV files, so separate invocations end up in one report) ──
function _parse(v)
    v = strip(v)
    startswith(v, '"') && return String(strip(v, '"'))
    x = tryparse(Int, v)
    x === nothing || return x
    x = tryparse(Float64, v)
    return x === nothing ? String(v) : x
end

function load_rows(out)
    path = joinpath(out, "runs.csv")
    isfile(path) || return Dict{String,Any}[]
    lines = readlines(path)
    header = split(lines[1], ',')
    return [Dict{String,Any}(String(h) => _parse(v) for (h, v) in zip(header, split(l, ',')))
            for l in lines[2:end] if !isempty(l)]
end

# diagnostics summaries from diag_<instance>_<pricing>_<backend>.csv
function load_diag(out)
    report = NamedTuple[]
    for f in sort(readdir(out))
        m = match(r"^diag_(.+)_(first_improving|all_blocks)_(cholmod|dense)\.csv$", f)
        m === nothing && continue
        lines = filter(!isempty, readlines(joinpath(out, f)))
        header = split(lines[1], ',')
        col(name) = [_parse(split(l, ',')[findfirst(==(name), header)]) for l in lines[2:end]]
        push!(report, (instance = m[1], pricing = m[2], backend = m[3],
                       max_residual = maximum(col("factor_residual")),
                       final_residual = col("factor_residual")[end], max_N = maximum(col("N")),
                       max_waste = maximum(col("factor_waste")),
                       mean_waste = mean(col("factor_waste")),
                       min_fill = minimum(col("factor_fill")[2:end]),
                       max_stored = maximum(col("factor_stored")),
                       max_dense = maximum(col("factor_dense")),
                       peak_MB = maximum(col("factor_peak_MB"))))
    end
    return report
end

_fmt_gap(x) = x isa Real && isfinite(x) ? @sprintf("%.1e", x) : "–"

function write_report(out)
    rows = load_rows(out)
    report = load_diag(out)
    io = IOBuffer()
    println(io, "# Factorization benchmark\n")
    println(io, "Machine: ", Sys.CPU_NAME, ", ", Sys.CPU_THREADS, " threads; Julia ", VERSION,
            "; BLAS threads ", BLAS.get_num_threads(), "\n")
    keys_ = unique((r["instance"], r["pricing"]) for r in rows)
    sel(inst, pr, b) = filter(r -> r["instance"] == inst && r["pricing"] == pr &&
                                   r["backend"] == String(b), rows)
    best(inst, pr, b) = (rs = sel(inst, pr, b); rs[argmin([r["total_s"] for r in rs])])
    median_total(inst, pr, b) = median([r["total_s"] for r in sel(inst, pr, b)])
    if !isempty(rows)
        println(io, "## CG runs (best of the repetitions; median total in parentheses)\n")
        println(io, "| instance | m | pricing | backend | status | iter | total [s] | master [s] | master speedup | inner it. | final N | refact. | cᵀλ − exact | master obj − exact | infeas. | dense − CHOLMOD (cᵀλ) |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (inst, pr) in keys_
            rc = best(inst, pr, :cholmod)
            for b in BACKENDS
                r = best(inst, pr, b)
                exact = r["exact_objective"]
                mgap = exact isa Real && isfinite(exact) ? r["master_objective"] - exact : NaN
                @printf(io, "| %s | %d | %s | %s | %s | %d | %.3f (%.3f) | %.3f | %.2f× | %d | %d | %d | %s | %s | %.1e | %s |\n",
                        inst, r["m"], pr, b, r["status"], r["iterations"], r["total_s"],
                        median_total(inst, pr, b), r["master_s"], rc["master_s"] / r["master_s"],
                        r["inner_iterations"], r["final_N"], r["refactorizations"],
                        _fmt_gap(r["gap_exact"]), _fmt_gap(mgap), r["infeasibility"],
                        b === :dense ? _fmt_gap(r["diff_cholmod"]) : "")
            end
        end
        println(io, "\n## Time per factor operation (best run)\n")
        println(io, "| instance | pricing | backend | assemble µs (calls) | rowadd µs (calls) | rowdel µs (calls) | updown µs (calls) | solve µs (calls) | refactor [s] | trailing frac. |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|")
        for (inst, pr) in keys_, b in BACKENDS
            r = best(inst, pr, b)
            cell(op) = r["op_$(op)_calls"] == 0 ? "–" :
                       @sprintf("%.2f (%d)", 1e6 * r["op_$(op)_s"] / r["op_$(op)_calls"], r["op_$(op)_calls"])
            tf = get(r, "dense_mean_trailing_fraction", "")
            @printf(io, "| %s | %s | %s | %s | %s | %s | %s | %s | %.3f | %s |\n", inst, pr, b,
                    cell("assemble"), cell("rowadd"), cell("rowdel"), cell("updown"), cell("solve"),
                    r["t_refactor"], tf isa Real ? @sprintf("%.2f", tf) : "")
        end
        println(io, "\n## QPAS time breakdown (best run) [s], Julia allocations\n")
        println(io, "| instance | pricing | backend | direction | multipliers | step | expand | reduce | (refactor) | allocated MB | GC [s] |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|")
        for (inst, pr) in keys_, b in BACKENDS
            r = best(inst, pr, b)
            @printf(io, "| %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.0f | %.3f |\n",
                    inst, pr, b, r["t_direction"], r["t_multipliers"], r["t_step"], r["t_expand"],
                    r["t_reduce"], r["t_refactor"], r["allocated_MB"], r["gc_s"])
        end
    end
    if !isempty(report)
        println(io, "\n## Factor contents, memory and accuracy (diagnostics runs, ~60 samples each)\n")
        println(io, "| instance | pricing | backend | max N | max stored | dense would be | max waste | mean waste | min fill | peak factor MB | max residual | final residual |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|")
        for d in report
            @printf(io, "| %s | %s | %s | %d | %d | %d | %.3f | %.3f | %.3f | %.2f | %.1e | %.1e |\n",
                    d.instance, d.pricing, d.backend, d.max_N, d.max_stored, d.max_dense,
                    d.max_waste, d.mean_waste, d.min_fill, d.peak_MB, d.max_residual,
                    d.final_residual)
        end
    end
    micro_path = joinpath(out, "micro.csv")
    if isfile(micro_path)
        recs = [split(l, ',') for l in readlines(micro_path)[2:end] if !isempty(l)]
        ops = ["solve", "rowadd_append", "rowdel_last", "rowdel_middle", "rowdel_first",
               "rowadd_reinsert_first", "update", "downdate", "rebuild", "memory_MB"]
        ks = sort(unique(parse(Int, r[1]) for r in recs))
        println(io, "\n## Micro benchmark: median µs per operation (memory in MB) vs active size k\n")
        println(io, "| op | backend | ", join(("k=$k" for k in ks), " | "), " |")
        println(io, "|---|---|", join(("---" for _ in ks), "|"), "|")
        for op in ops, b in BACKENDS
            vals = map(ks) do k
                i = findfirst(r -> r[1] == string(k) && r[2] == String(b) && r[3] == op, recs)
                i === nothing && return "–"
                v = parse(Float64, recs[i][4])
                op == "memory_MB" ? @sprintf("%.2f", v) : @sprintf("%.1f", v)
            end
            println(io, "| ", op, " | ", b, " | ", join(vals, " | "), " |")
        end
    end
    path = joinpath(out, "report.md")
    write(path, take!(io))
    return path
end

function main(args)
    opts = parse_args(args)
    out = opts["out"]
    mkpath(out)
    suite = opts["suite"]
    # warm-up: compile both backends on a tiny instance
    tiny = generate_instance(InstanceConfig(n_nodes = 20, edge_probability = 0.2, n_tasks = 12, n_pilots = 3))
    for b in BACKENDS
        run_cg(tiny, b, :all_blocks; time_limit = 60.0)
        run_cg(tiny, b, :all_blocks; time_limit = 60.0, diagnostics_every = 5)
    end
    if suite == "report"
        println("report: ", write_report(out))
        return
    end
    if suite in ("micro", "all")
        println("═"^100, "\nmicro benchmark")
        micro(out)
    end
    # rows of earlier invocations (e.g. the small suite) are kept in runs.csv
    rows = load_rows(out)
    report = NamedTuple[]
    if suite in ("small", "all")
        run_suite(SMALL, (:first_improving, :all_blocks), opts, rows, report; exact = true)
    end
    if suite in ("large", "all")
        run_suite(LARGE, (:all_blocks,), opts, rows, report; exact = false)
    end
    println("report: ", write_report(out))
end

main(ARGS)
