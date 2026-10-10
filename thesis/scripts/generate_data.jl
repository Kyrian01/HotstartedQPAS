# Generates every raw data file used in the results chapter of the thesis.
#
#   julia --project thesis/scripts/generate_data.jl            (from the repository root)
#
# Output: thesis/data/raw/*.csv. The plotting tables are derived from these files by
# thesis/scripts/make_plot_data.jl, so plots can be changed without rerunning experiments.

using HotstartedQPAS, Printf, Statistics
const HQ = HotstartedQPAS

const RAW = joinpath(@__DIR__, "..", "data", "raw")
mkpath(RAW)

#              name  |V|   p      tasks pilots
const SIZES = [("I1",  50, 0.098,  40,  4),
               ("I2", 150, 0.090, 230,  7),
               ("I3", 175, 0.091, 390, 10),
               ("I4", 225, 0.090, 500, 13)]
const FRACS = [0.0, 0.3]
const QP_LIMIT = 600.0        # time limit for the HiGHS QP masters
const LP_LIMIT = 900.0

instance(name, V, p, T, P, frac) =
    generate_instance(InstanceConfig(n_nodes = V, edge_probability = p, n_tasks = T,
                                     n_pilots = P, inequality_fraction = frac,
                                     name = "$(name)_$(round(Int, 100frac))"))

tag(name, frac) = "$(name)_$(round(Int, 100frac))"

function append_row(path, header, fields)
    isfile(path) || open(io -> println(io, header), path, "w")
    open(io -> println(io, join(fields, ",")), path, "a")
end

# ── warm-up (compilation) ────────────────────────────────────────────────────────────────
let w = instance("W", 20, 0.2, 12, 3, 0.5)
    for m in (HighsLPMaster(), HighsExplicitQPMaster(), QPASMaster())
        solve_column_generation(w, m; options = CGOptions(diagnostics_every = 5))
    end
    solve_column_generation(instance("W", 20, 0.2, 12, 3, 0.0), HighsEliminatedQPMaster())
    solve_exact(w)
end

# ── A. instances, solver comparison, traces, QPAS details ───────────────────────────────
const SUMMARY = joinpath(RAW, "comparison.csv")
const SUMMARY_HDR = "instance,frac,solver,status,iterations,columns,total_s,master_s,pricing_s," *
                    "inner,objective,master_objective,bound,infeasibility,exact_objective"
const DETAILS = joinpath(RAW, "qpas_details.csv")
const INSTANCES = joinpath(RAW, "instances.csv")
for f in (SUMMARY, DETAILS, INSTANCES)
    isfile(f) && rm(f)
end

function cg_row(name, frac, solver, r, zstar)
    s = r.stats
    return [name, frac, solver, r.status, s.iterations, length(r.weights),
            @sprintf("%.4f", s.total_time), @sprintf("%.4f", sum(s.master_time)),
            @sprintf("%.4f", sum(s.pricing_time)), sum(s.inner_iterations),
            @sprintf("%.10f", r.objective), @sprintf("%.10f", s.master_objective[end]),
            @sprintf("%.10f", r.lower_bound), @sprintf("%.3e", r.infeasibility),
            @sprintf("%.10f", zstar)]
end

for (name, V, p, T, P) in SIZES, frac in FRACS
    inst = instance(name, V, p, T, P, frac)
    t = tag(name, frac)
    nI = count(isinf, inst.task_upper)
    append_row(INSTANCES, "instance,frac,V,E,m,m_ineq,pilots,N",
               [name, frac, n_nodes(inst), n_edges(inst), n_tasks(inst), nI, n_pilots(inst),
                HQ.n_flow_variables(inst)])

    ex = solve_exact(inst)
    zstar = ex.objective
    append_row(SUMMARY, SUMMARY_HDR,
               [name, frac, "exact", ex.status, ex.stats.simplex_iterations, 0,
                @sprintf("%.4f", ex.stats.setup_time + ex.stats.solve_time),
                @sprintf("%.4f", ex.stats.solve_time), 0, ex.stats.simplex_iterations,
                @sprintf("%.10f", zstar), @sprintf("%.10f", zstar), @sprintf("%.10f", zstar),
                "0", @sprintf("%.10f", zstar)])
    println("$t exact: $(round(ex.stats.solve_time; digits = 2)) s"); flush(stdout)

    # QPAS: timed twice without diagnostics (keep the faster), then once with diagnostics
    best = nothing
    for _ in 1:2
        GC.gc()
        r = solve_column_generation(inst, QPASMaster())
        (best === nothing || r.stats.total_time < best.stats.total_time) && (best = r)
    end
    append_row(SUMMARY, SUMMARY_HDR, cg_row(name, frac, "qpas", best, zstar))
    write_trace_csv(joinpath(RAW, "trace_$(t)_qpas.csv"), best.stats)
    for (k, v) in best.stats.details
        append_row(DETAILS, "instance,frac,key,value",
                   [name, frac, "\"" * strip(k) * "\"", "\"$(v)\""])
    end
    rd = solve_column_generation(inst, QPASMaster(); options = CGOptions(diagnostics_every = 20))
    write_diagnostics_csv(joinpath(RAW, "diag_$(t)_qpas.csv"), rd.stats)
    println("$t qpas: $(round(best.stats.total_time; digits = 2)) s"); flush(stdout)

    # the other column generation masters (where they finish in reasonable time)
    others = Tuple{String,Any,Float64}[]
    name in ("I1", "I2", "I3") && push!(others, ("lp", HighsLPMaster, LP_LIMIT))
    name in ("I1", "I2") && push!(others, ("explicit_qp", HighsExplicitQPMaster, QP_LIMIT))
    name in ("I1", "I2") && frac == 0.0 &&
        push!(others, ("eliminated_qp", HighsEliminatedQPMaster, QP_LIMIT))
    for (solver, M, limit) in others
        GC.gc()
        r = solve_column_generation(inst, M(); options = CGOptions(time_limit = limit))
        append_row(SUMMARY, SUMMARY_HDR, cg_row(name, frac, solver, r, zstar))
        write_trace_csv(joinpath(RAW, "trace_$(t)_$(solver).csv"), r.stats)
        println("$t $solver: $(r.status) $(round(r.stats.total_time; digits = 2)) s")
        flush(stdout)
    end
end

# ── B. refactorization interval sweep with factor diagnostics ───────────────────────────
const REFACTOR = joinpath(RAW, "refactor.csv")
isfile(REFACTOR) && rm(REFACTOR)
for (name, V, p, T, P) in SIZES[2:3]
    inst = instance(name, V, p, T, P, 0.0)
    for τ in (0, 25, 50, 100, 150, 300, 600)
        best = Inf
        r = nothing
        for _ in 1:2
            GC.gc()
            r = solve_column_generation(inst, QPASMaster(refactor_interval = τ))
            best = min(best, r.stats.total_time)
        end
        d = Dict(r.stats.details)
        append_row(REFACTOR, "instance,tau,total_s,iterations,refactorizations,row_updates",
                   [name, τ, @sprintf("%.4f", best), r.stats.iterations,
                    d["refactorizations"], d["CHOLMOD row updates"]])
        rd = solve_column_generation(inst, QPASMaster(refactor_interval = τ);
                                     options = CGOptions(diagnostics_every = 25))
        write_diagnostics_csv(joinpath(RAW, "diag_$(name)_tau$(τ).csv"), rd.stats)
        println("$name τ=$τ: $(round(best; digits = 2)) s"); flush(stdout)
    end
end

# ── C. constraint matrix D_W before / after the reordering (spy plot data) ──────────────
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

println("done")
