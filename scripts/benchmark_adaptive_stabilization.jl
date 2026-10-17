# Benchmark of the geometric δ rule (docs/adaptive_stabilization.md).
#
#   julia --project scripts/benchmark_adaptive_stabilization.jl            (from the repository root)
#   julia --project scripts/benchmark_adaptive_stabilization.jl I1 I2      (only these instances)
#   HQ_PRICING=all_blocks julia --project scripts/benchmark_adaptive_stabilization.jl
#
# Instances I1–I4 of the thesis (Table 9.1), equality variant and 30 % inequality rows, solved
# by the hot-started QPAS with the dense factor and default CGOptions, i.e. first-improving
# pricing as in thesis Chapter 9 (HQ_PRICING=all_blocks: one column per pilot, the default of
# experiments.toml). Every variant is run `REPS` times, the fastest run is kept; the exact LP is
# the reference optimum.
#
# Output: results/adaptive_stabilization/<pricing>/summary.csv and trace_<instance>_<variant>.csv

using HotstartedQPAS, LinearAlgebra, Printf
const HQ = HotstartedQPAS

BLAS.set_num_threads(1)

const PRICING = Symbol(get(ENV, "HQ_PRICING", "first_improving"))
const OUT = joinpath(@__DIR__, "..", "results", "adaptive_stabilization", String(PRICING))
mkpath(OUT)
const REPS = 3

#              name  |V|   p      tasks pilots
const SIZES = [("I1",  50, 0.098,  40,  4),
               ("I2", 150, 0.090, 230,  7),
               ("I3", 175, 0.091, 390, 10),
               ("I4", 225, 0.090, 500, 13)]
const ONLY = isempty(ARGS) ? [s[1] for s in SIZES] : ARGS
const FRACS = [0.0, 0.3]

const VARIANTS = [
    # thesis method and constant-δ controls
    ("fixed",            StabilizationOptions()),
    ("fixed_d0.1",       StabilizationOptions(delta = 0.1)),
    ("fixed_d0.01",      StabilizationOptions(delta = 0.01)),
    ("fixed_d3",         StabilizationOptions(delta = 3.0)),
    ("fixed_d10",        StabilizationOptions(delta = 10.0)),
    # geometric rule: halve per center-only iteration; reset to δ₀ or double on a column
    ("geometric",        StabilizationOptions(delta_update = :geometric, delta_increase = Inf)),
    ("geometric_d3",     StabilizationOptions(delta = 3.0, delta_update = :geometric,
                                              delta_increase = Inf)),
    ("geometric_d10",    StabilizationOptions(delta = 10.0, delta_update = :geometric,
                                              delta_increase = Inf)),
    ("geometric_x2_d0.3", StabilizationOptions(delta = 0.3, delta_update = :geometric,
                                               delta_increase = 2.0)),
    ("geometric_x2",     StabilizationOptions(delta_update = :geometric, delta_increase = 2.0)),
    ("geometric_x2_d3",  StabilizationOptions(delta = 3.0, delta_update = :geometric,
                                              delta_increase = 2.0)),
    ("geometric_x2_d10", StabilizationOptions(delta = 10.0, delta_update = :geometric,
                                              delta_increase = 2.0)),
]

const SUMMARY = joinpath(OUT, "summary.csv")
const HEADER = "instance,frac,variant,status,iterations,column_iters,center_only_iters," *
               "columns,total_s,master_s,pricing_s,inner,objective_relerr," *
               "bound_gap,infeasibility,delta_min_used,exact_objective,exact_s"
isempty(ARGS) && isfile(SUMMARY) && rm(SUMMARY)
isfile(SUMMARY) || open(io -> println(io, HEADER), SUMMARY, "w")

instance(name, V, p, T, P, frac) =
    generate_instance(InstanceConfig(n_nodes = V, edge_probability = p, n_tasks = T,
                                     n_pilots = P, inequality_fraction = frac,
                                     name = "$(name)_$(round(Int, 100frac))"))

# warm-up (compilation)
let w = instance("W", 20, 0.2, 12, 3, 0.3)
    for (_, s) in VARIANTS
        solve_column_generation(w, QPASMaster(factorization = :dense);
                                options = CGOptions(pricing = PRICING, stabilization = s))
    end
    solve_exact(w)
end

for (name, V, p, T, P) in SIZES, frac in FRACS
    name in ONLY || continue
    inst = instance(name, V, p, T, P, frac)
    tag = "$(name)_$(round(Int, 100frac))"
    ex = solve_exact(inst)
    zstar = ex.objective
    for (label, stab) in VARIANTS
        best = nothing
        for _ in 1:REPS
            GC.gc()
            r = solve_column_generation(inst, QPASMaster(factorization = :dense);
                                        options = CGOptions(pricing = PRICING, stabilization = stab))
            (best === nothing || r.stats.total_time < best.stats.total_time) && (best = r)
        end
        r, s = best, best.stats
        write_trace_csv(joinpath(OUT, "trace_$(tag)_$(label).csv"), s)
        col_iters = count(>(0), s.columns_added)
        row = [name, frac, label, r.status, s.iterations, col_iters, s.iterations - col_iters,
               length(r.weights),
               @sprintf("%.4f", s.total_time), @sprintf("%.4f", total_master_time(s)),
               @sprintf("%.4f", HQ.total_pricing_time(s)), HQ.total_inner_iterations(s),
               @sprintf("%.3e", abs(r.objective - zstar) / abs(zstar)),
               @sprintf("%.3e", (zstar - r.lower_bound) / abs(zstar)),
               @sprintf("%.3e", r.infeasibility), @sprintf("%.3e", minimum(s.delta)),
               @sprintf("%.10f", zstar), @sprintf("%.4f", ex.stats.solve_time)]
        open(io -> println(io, join(row, ",")), SUMMARY, "a")
        @printf("%-6s %-16s %-8s it %6d (col %5d)  %8.3f s  relerr %.1e\n",
                tag, label, r.status, s.iterations, col_iters,
                s.total_time, abs(r.objective - zstar) / abs(zstar))
        flush(stdout)
    end
end
