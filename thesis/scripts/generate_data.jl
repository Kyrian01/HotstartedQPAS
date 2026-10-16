# Generates every raw data file used in the results chapter of the thesis.
#
#   julia --project thesis/scripts/generate_data.jl                 (from the repository root)
#   julia --project thesis/scripts/generate_data.jl A B             (only the given sections)
#
# Sections: A comparison of all methods, both factorizations of the active-set method
#           B refactorization interval (CHOLMOD) and factor contents of both backends
#           C constraint matrix D_W before / after the reordering (spy plot)
#           D scaling to larger instances (S5, S6)
#           E cost of the individual factor operations against the active size
#
# Output: thesis/data/raw/*.csv. The plotting tables are derived from these files by
# thesis/scripts/make_plot_data.jl, so plots can be changed without rerunning experiments.

using HotstartedQPAS, LinearAlgebra, Printf, Random, Statistics
const HQ = HotstartedQPAS
const LDLt = HQ.LDLt

BLAS.set_num_threads(1)              # all experiments single-threaded

# HQ_SMOKE=<dir>: run every section on tiny instances and write to <dir> (checks the script)
const SMOKE = get(ENV, "HQ_SMOKE", "")
const RAW = isempty(SMOKE) ? joinpath(@__DIR__, "..", "data", "raw") : SMOKE
mkpath(RAW)
const SECTIONS = isempty(ARGS) ? ["A", "B", "C", "D", "E"] : uppercase.(ARGS)

#              name  |V|   p      tasks pilots
const SIZES = isempty(SMOKE) ?
              [("I1",  50, 0.098,  40,  4),
               ("I2", 150, 0.090, 230,  7),
               ("I3", 175, 0.091, 390, 10),
               ("I4", 225, 0.090, 500, 13)] :
              [("I1", 30, 0.15, 20, 3), ("I2", 40, 0.12, 30, 4)]
const LARGE = isempty(SMOKE) ?
              [("S5", 300, 0.080, 701, 16),
               ("S6", 400, 0.070, 999, 20)] :
              [("S5", 45, 0.12, 35, 4)]
const FRACS = [0.0, 0.3]
const QP_LIMIT = 600.0        # time limit for the HiGHS QP masters on I1 (they finish)
const QP_LIMIT_I2 = 120.0     # ... on I2: enough to compare the path and the time per master
const LP_LIMIT = 900.0        # ... for the LP master
const EXACT_LIMIT = 300.0     # ... for the direct LP solve of the large instances

# the two factorizations of the active-set method (solver names in the raw files)
const BACKENDS = (("qpas", :cholmod), ("qpas_dense", :dense))

instance(name, V, p, T, P, frac) =
    generate_instance(InstanceConfig(n_nodes = V, edge_probability = p, n_tasks = T,
                                     n_pilots = P, inequality_fraction = frac,
                                     name = "$(name)_$(round(Int, 100frac))"))

tag(name, frac) = "$(name)_$(round(Int, 100frac))"

function append_row(path, header, fields)
    isfile(path) || open(io -> println(io, header), path, "w")
    open(io -> println(io, join(fields, ",")), path, "a")
end

const SUMMARY = joinpath(RAW, "comparison.csv")
const SUMMARY_HDR = "instance,frac,solver,status,iterations,columns,total_s,master_s,pricing_s," *
                    "inner,objective,master_objective,bound,infeasibility,exact_objective"
const DETAILS = joinpath(RAW, "qpas_details.csv")
const DETAILS_HDR = "instance,frac,solver,key,value"
const INSTANCES = joinpath(RAW, "instances.csv")

function cg_row(name, frac, solver, r, zstar)
    s = r.stats
    return [name, frac, solver, r.status, s.iterations, length(r.weights),
            @sprintf("%.4f", s.total_time), @sprintf("%.4f", sum(s.master_time)),
            @sprintf("%.4f", sum(s.pricing_time)), sum(s.inner_iterations),
            @sprintf("%.10f", r.objective), @sprintf("%.10f", s.master_objective[end]),
            @sprintf("%.10f", r.lower_bound), @sprintf("%.3e", r.infeasibility),
            @sprintf("%.10f", zstar)]
end

function exact_row(name, frac, ex)
    zstar = ex.objective
    return [name, frac, "exact", ex.status, ex.stats.simplex_iterations, 0,
            @sprintf("%.4f", ex.stats.setup_time + ex.stats.solve_time),
            @sprintf("%.4f", ex.stats.solve_time), 0, ex.stats.simplex_iterations,
            @sprintf("%.10f", zstar), @sprintf("%.10f", zstar), @sprintf("%.10f", zstar),
            "0", @sprintf("%.10f", zstar)]
end

function write_details(name, frac, solver, r)
    for (k, v) in r.stats.details
        append_row(DETAILS, DETAILS_HDR,
                   [name, frac, solver, "\"" * strip(k) * "\"", "\"$(v)\""])
    end
end

# One backend of the active-set method on one instance: `reps` timed runs without diagnostics
# (the fastest is kept; a run longer than `single_above` seconds is not repeated), then one run
# with diagnostics every `every` iterations (with the factor residual if `residual`).
function run_qpas(inst, name, frac, solver, backend, zstar; reps = 2, single_above = Inf,
                  every = 20, residual = true)
    t = tag(name, frac)
    best = nothing
    for rep in 1:reps
        GC.gc()
        r = solve_column_generation(inst, QPASMaster(factorization = backend))
        (best === nothing || r.stats.total_time < best.stats.total_time) && (best = r)
        r.stats.total_time > single_above && break
    end
    append_row(SUMMARY, SUMMARY_HDR, cg_row(name, frac, solver, best, zstar))
    write_trace_csv(joinpath(RAW, "trace_$(t)_$(solver).csv"), best.stats)
    write_details(name, frac, solver, best)
    if every > 0
        GC.gc()
        rd = solve_column_generation(inst, QPASMaster(factorization = backend);
                                     options = CGOptions(diagnostics_every = every,
                                                         diagnostics_residual = residual))
        write_diagnostics_csv(joinpath(RAW, "diag_$(t)_$(solver).csv"), rd.stats)
    end
    println("$t $solver: $(best.status) $(round(best.stats.total_time; digits = 2)) s")
    flush(stdout)
    return best
end

# ── warm-up (compilation) ────────────────────────────────────────────────────────────────
let w = instance("W", 20, 0.2, 12, 3, 0.5)
    for m in (HighsLPMaster(), HighsExplicitQPMaster(), QPASMaster(),
              QPASMaster(factorization = :dense))
        solve_column_generation(w, m; options = CGOptions(diagnostics_every = 5,
                                                          diagnostics_residual = true))
    end
    solve_column_generation(instance("W", 20, 0.2, 12, 3, 0.0), HighsEliminatedQPMaster())
    solve_exact(w)
end

# ── A. instances, solver comparison, traces, details of the active-set method ───────────
if "A" in SECTIONS
    for f in (SUMMARY, DETAILS, INSTANCES)
        isfile(f) && rm(f)
    end
    for (name, V, p, T, P) in SIZES, frac in FRACS
        inst = instance(name, V, p, T, P, frac)
        t = tag(name, frac)
        nI = count(isinf, inst.task_upper)
        append_row(INSTANCES, "instance,frac,V,E,m,m_ineq,pilots,N",
                   [name, frac, n_nodes(inst), n_edges(inst), n_tasks(inst), nI, n_pilots(inst),
                    HQ.n_flow_variables(inst)])

        GC.gc()
        ex = solve_exact(inst)
        zstar = ex.objective
        append_row(SUMMARY, SUMMARY_HDR, exact_row(name, frac, ex))
        println("$t exact: $(round(ex.stats.solve_time; digits = 2)) s"); flush(stdout)

        for (solver, backend) in BACKENDS
            run_qpas(inst, name, frac, solver, backend, zstar)
        end

        # the other column generation masters (where they finish in reasonable time)
        others = Tuple{String,Any,Float64}[]
        name in ("I1", "I2", "I3") && push!(others, ("lp", HighsLPMaster, LP_LIMIT))
        # HiGHS QP masters: full runs on I1, time-limited runs on the equality variant of I2
        qp_limit = name == "I1" ? QP_LIMIT : QP_LIMIT_I2
        (name == "I1" || (name == "I2" && frac == 0.0)) &&
            push!(others, ("explicit_qp", HighsExplicitQPMaster, qp_limit))
        (name in ("I1", "I2") && frac == 0.0) &&
            push!(others, ("eliminated_qp", HighsEliminatedQPMaster, qp_limit))
        for (solver, M, limit) in others
            GC.gc()
            r = solve_column_generation(inst, M(); options = CGOptions(time_limit = limit))
            append_row(SUMMARY, SUMMARY_HDR, cg_row(name, frac, solver, r, zstar))
            write_trace_csv(joinpath(RAW, "trace_$(t)_$(solver).csv"), r.stats)
            println("$t $solver: $(r.status) $(round(r.stats.total_time; digits = 2)) s")
            flush(stdout)
        end
    end
end

# ── B. refactorization interval (CHOLMOD) and factor contents of both backends ──────────
# The dense backend needs no periodic refactorization; its row (backend "dense", τ = 0) is the
# default configuration. A second dense row with τ = 150 shows what a periodic rebuild costs it.
if "B" in SECTIONS
    REFACTOR = joinpath(RAW, "refactor.csv")
    isfile(REFACTOR) && rm(REFACTOR)
    configs = vcat([(:cholmod, τ) for τ in (0, 25, 50, 100, 150, 300, 600)],
                   [(:dense, 0), (:dense, 150)])
    for (name, V, p, T, P) in SIZES[2:min(3, end)]
        inst = instance(name, V, p, T, P, 0.0)
        for (backend, τ) in configs
            master() = backend === :cholmod ? QPASMaster(refactor_interval = τ) :
                       QPASMaster(factorization = :dense, dense_refactor_interval = τ)
            best = Inf
            r = nothing
            for _ in 1:2
                GC.gc()
                r = solve_column_generation(inst, master())
                best = min(best, r.stats.total_time)
            end
            d = Dict(r.stats.details)
            append_row(REFACTOR,
                       "instance,backend,tau,total_s,master_s,iterations,refactorizations,updates",
                       [name, backend, τ, @sprintf("%.4f", best),
                        @sprintf("%.4f", sum(r.stats.master_time)), r.stats.iterations,
                        d["refactorizations"], d["factor updates (row add/del, rank-one)"]])
            GC.gc()
            rd = solve_column_generation(inst, master(); options = CGOptions(diagnostics_every = 25))
            suffix = backend === :cholmod ? "tau$(τ)" : "dense_tau$(τ)"
            write_diagnostics_csv(joinpath(RAW, "diag_$(name)_$(suffix).csv"), rd.stats)
            println("$name $backend τ=$τ: $(round(best; digits = 2)) s"); flush(stdout)
        end
    end
end

# ── C. constraint matrix D_W before / after the reordering (spy plot data) ──────────────
if "C" in SECTIONS
    let (name, V, p, T, P) = SIZES[2], frac = 0.3
        inst = instance(name, V, p, T, P, frac)
        m = QPASMaster()
        solve_column_generation(inst, m)
        st = m.state
        ord, sp, pool = st.kkt.ordering, st.kkt.surplus, st.pool
        n, mI, P_ = pool.n, HQ.n_surplus(sp), pool.n_blocks
        Wx = sort(collect(HQ.items(ord.W)))
        Nx = sort(collect(HQ.items(ord.N)))
        Ws = [σ for σ in 1:mI if !HQ.is_free(sp, σ)]
        Ns = [σ for σ in 1:mI if HQ.is_free(sp, σ)]
        # entries (row, column) of D_W = [C 0; W_x 0; 0 W_s] in the original column order
        entries = Tuple{Int,Int}[]
        for j in 1:n
            push!(entries, (pool.block[j], j))
        end
        for (t, j) in enumerate(Wx)
            push!(entries, (P_ + t, j))
        end
        for (t, σ) in enumerate(Ws)
            push!(entries, (P_ + length(Wx) + t, n + σ))
        end
        # permutation [S, W_x, N_x | W_s, N_s]
        order = vcat(ord.rep, Wx, Nx, n .+ Ws, n .+ Ns)
        newpos = zeros(Int, n + mI)
        newpos[order] = 1:(n + mI)
        open(joinpath(RAW, "spy_DW.csv"), "w") do io
            println(io, "row,col,col_perm")
            for (r, c) in entries
                println(io, r, ",", c, ",", newpos[c])
            end
        end
        open(joinpath(RAW, "spy_DW_sizes.csv"), "w") do io
            println(io, "p,Wx,Nx,Ws,Ns,n,mI,rows")
            println(io, join([P_, length(Wx), length(Nx), length(Ws), length(Ns), n, mI,
                              P_ + length(Wx) + length(Ws)], ","))
        end
        println("spy data written")
    end
end

# ── D. scaling: larger instances, direct LP with a time limit, both backends ────────────
if "D" in SECTIONS
    SCALING = joinpath(RAW, "scaling.csv")
    isfile(SCALING) && rm(SCALING)
    for (name, V, p, T, P) in LARGE, frac in FRACS
        inst = instance(name, V, p, T, P, frac)
        nI = count(isinf, inst.task_upper)
        GC.gc()
        ex = solve_exact(inst; time_limit = EXACT_LIMIT)
        zstar = ex.status === :optimal ? ex.objective : NaN
        println("$(tag(name, frac)) exact: $(ex.status) $(round(ex.stats.solve_time; digits = 2)) s")
        flush(stdout)
        for (solver, backend) in BACKENDS
            # a run of more than a minute is timed once; no diagnostics run
            r = run_qpas(inst, name, frac, solver, backend, zstar; single_above = 60.0,
                         every = 0)
            d = Dict(r.stats.details)
            append_row(SCALING,
                       "instance,frac,solver,V,E,m,m_ineq,pilots,status,iterations,columns," *
                       "total_s,master_s,pricing_s,inner,objective,bound,infeasibility," *
                       "exact_status,exact_s,exact_objective,refactorizations,final_N",
                       [name, frac, solver, n_nodes(inst), n_edges(inst), n_tasks(inst), nI,
                        n_pilots(inst), r.status, r.stats.iterations, length(r.weights),
                        @sprintf("%.4f", r.stats.total_time), @sprintf("%.4f", sum(r.stats.master_time)),
                        @sprintf("%.4f", sum(r.stats.pricing_time)), sum(r.stats.inner_iterations),
                        @sprintf("%.10f", r.objective), @sprintf("%.10f", r.lower_bound),
                        @sprintf("%.3e", r.infeasibility), ex.status,
                        @sprintf("%.4f", ex.stats.setup_time + ex.stats.solve_time),
                        @sprintf("%.10f", zstar), d["refactorizations"],
                        d["final |N| (nullspace dim.)"]])
        end
    end
end

# ── E. cost of the factor operations against the active size k ──────────────────────────
# H is the Gram matrix of k+1 random sparse 0/±1 columns with k+10 rows (dense, like H_eff).
# Both backends start from the identity of size n_max and activate 1..k in increasing order.
if "E" in SECTIONS
    rows = String["k,backend,op,median_us,min_us,samples"]
    rng = Xoshiro(1)
    n_max = 20_000
    for k in (isempty(SMOKE) ? (50, 100, 200, 350, 500, 750, 1000, 1500, 2000) : (20, 40))
        m = k + 10
        U = Float64.(rand(rng, m, k + 1) .< 0.08) .- Float64.(rand(rng, m, k + 1) .< 0.04)
        H = U' * U + 1e-3I
        for backend in (:cholmod, :dense)
            F = HQ.make_factor(backend, n_max, m)
            for j in 1:k
                LDLt.rowadd!(F, j, collect(1:j), H[1:j, j], j)
            end
            samples = Dict{String,Vector{Float64}}()
            rec(op, t) = push!(get!(samples, op, Float64[]), t * 1e-3)
            for rep in 1:(k <= 500 ? 60 : 15)
                # append index k+1 and remove it again (the newest position)
                j = k + 1
                t0 = time_ns(); LDLt.rowadd!(F, j, collect(1:j), H[1:j, j], j); rec("append", time_ns() - t0)
                t0 = time_ns(); LDLt.rowdel!(F, j); rec("delete_last", time_ns() - t0)
                # delete the oldest / a middle position and re-activate that index
                for (label, pos) in (("first", 1), ("middle", k ÷ 2))
                    q = backend === :dense ? F.index[pos] : pos
                    t0 = time_ns(); LDLt.rowdel!(F, q); rec("delete_$(label)", time_ns() - t0)
                    active = [i for i in 1:k if i != q]
                    LDLt.rowadd!(F, q, vcat(active, q), vcat(H[active, q], H[q, q]), k)
                end
                # rank-one update and downdate with a dense vector
                v = 0.1 .* randn(rng, k)
                t0 = time_ns(); LDLt.updown!(F, true, collect(1:k), v, k); rec("update", time_ns() - t0)
                t0 = time_ns(); LDLt.updown!(F, false, collect(1:k), v, k); rec("downdate", time_ns() - t0)
                # solve on the active set
                rhs = LDLt.rhs_buffer(F)
                rhs[1:k] .= randn(rng, k)
                t0 = time_ns(); LDLt.solve!(F, collect(1:k), k); rec("solve", time_ns() - t0)
                rhs[1:k] .= 0.0
            end
            # one rebuild: reset and k appends in increasing order
            t0 = time_ns()
            LDLt.reset!(F)
            for j in 1:k
                LDLt.rowadd!(F, j, collect(1:j), H[1:j, j], j)
            end
            rec("rebuild", time_ns() - t0)
            for (op, v) in sort(collect(samples))
                push!(rows, @sprintf("%d,%s,%s,%.3f,%.3f,%d", k, backend, op, median(v),
                                     minimum(v), length(v)))
            end
            fs = LDLt.factor_stats(F, collect(1:k))
            push!(rows, @sprintf("%d,%s,memory_MB,%.4f,%.4f,1", k, backend, fs.memory / 2^20,
                                 fs.memory / 2^20))
            close(F)
            @printf("micro k=%5d %-7s solve %8.1f µs  append %8.1f µs  delete-first %8.1f µs\n",
                    k, backend, median(samples["solve"]), median(samples["append"]),
                    median(samples["delete_first"]))
            flush(stdout)
        end
    end
    write(joinpath(RAW, "micro.csv"), join(rows, '\n') * '\n')
end

println("done")
