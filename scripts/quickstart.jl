# Minimal tour of the API. Run from the project root:
#
#   julia --project scripts/quickstart.jl
#
# For running experiments use run.jl with experiments.toml instead; this script shows how to
# call the solvers directly from Julia.

using HotstartedQPAS

# 1. Generate an instance (all InstanceConfig fields are listed in experiments.toml)
instance = generate_instance(InstanceConfig(n_nodes = 50, edge_probability = 0.098, n_tasks = 40,
                                            n_pilots = 4, seed = 1, name = "I1"))
println(instance)

# 2. Reference: the full LP solved directly
exact = solve_exact(instance)

# 3. Column generation with the four master problem solvers
options = CGOptions(stabilization = StabilizationOptions(delta = 1.0))
lp = solve_column_generation(instance, HighsLPMaster(); options)
explicit = solve_column_generation(instance, HighsExplicitQPMaster(); options)
eliminated = solve_column_generation(instance, HighsEliminatedQPMaster(); options)
qpas = solve_column_generation(instance, QPASMaster(); options)

# 4. Compare (thesis Table 5.5 style) and inspect the QPAS run
compare_results(exact, lp, explicit, eliminated, qpas; reference = exact)
println()
print_details(qpas)

# 5. Solutions: column weights → per-pilot edge flows, and the active schedules
flows = edge_flows(qpas, instance)            # n_edges × n_pilots, comparable to exact.flows
for col in active_columns(qpas)[1:min(5, end)]
    println("pilot $(col.pilot): weight $(round(col.weight; digits = 4)), cost $(round(col.cost; digits = 2)), path $(col.path)")
end

# 6. Per-iteration traces (master time, objective, bound, dual oscillation, ...)
write_trace_csv(joinpath("results", "quickstart_qpas.csv"), qpas.stats)
