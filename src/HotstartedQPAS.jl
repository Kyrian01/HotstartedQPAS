"""
    HotstartedQPAS

Column generation for the multi-block crew scheduling LP, with quadratic proximal
stabilization and a hot-started nullspace active-set QP solver (QPAS) whose reduced-Hessian
LDLᵀ factor is maintained by row updates: with CHOLMOD (sparse, fixed size `n_max`) or as a
dense factor of the active block (`QPASOptions(factorization = :dense)`). Master's thesis of
K. Bäumer, "Hot-Started Active-Set Method for Regularized Column Generation".

Solvers:

| solver                                         | master problem                          |
|:-----------------------------------------------|:----------------------------------------|
| [`solve_exact`](@ref)                          | full edge-based LP, HiGHS               |
| [`HighsLPMaster`](@ref)                        | LP master, HiGHS simplex                |
| [`HighsExplicitQPMaster`](@ref)                | stabilized QP with `w`, HiGHS QP        |
| [`HighsEliminatedQPMaster`](@ref)              | stabilized QP without `w`, HiGHS QP     |
| [`QPASMaster`](@ref)                           | stabilized QP without `w`, hot-started QPAS |

The column generation solvers all run through [`solve_column_generation`](@ref). See README.md
for the architecture and the mapping between the code and the thesis.
"""
module HotstartedQPAS

using LinearAlgebra
using SparseArrays
using Random
using Printf
using Serialization
import TOML

# ── low-level backends: the only code that calls into C libraries ───────────────────────
# the LDLᵀ factor interface and its two implementations (sparse CHOLMOD, dense)
include("backends/ldlt_interface.jl")
include("backends/cholmod/Cholmod.jl")
include("backends/dense/DenseLDLt.jl")
include("backends/highs/HighsBackend.jl")
import .LDLtInterface
import .Cholmod
import .DenseLDLt
using .LDLtInterface: UpdatableLDLt
using .HighsBackend
# factor operations are called qualified (`LDLt.solve!`), since `solve!` also names the
# master-problem solve
const LDLt = LDLtInterface

# ── utilities ────────────────────────────────────────────────────────────────────────────
include("utils/index_set.jl")
include("utils/timer.jl")

# ── problem data ─────────────────────────────────────────────────────────────────────────
include("problem/network.jl")
include("problem/instance.jl")
include("problem/formulation.jl")
include("problem/generator.jl")
include("problem/io.jl")

# ── column generation building blocks ────────────────────────────────────────────────────
include("columns/column_pool.jl")
include("pricing/shortest_path.jl")
include("stats/run_stats.jl")

# ── solver 1: exact LP ───────────────────────────────────────────────────────────────────
include("exact/exact_lp.jl")

# ── restricted master problem solvers ────────────────────────────────────────────────────
include("master/interface.jl")
include("master/highs_lp.jl")
include("master/highs_qp_explicit.jl")
include("master/highs_qp_eliminated.jl")

# ── hot-started QPAS ─────────────────────────────────────────────────────────────────────
include("qpas/options.jl")
include("qpas/ordering.jl")
include("qpas/nullspace.jl")
include("qpas/kkt_system.jl")
include("qpas/hotstart_state.jl")
include("qpas/solver.jl")
include("master/qpas_master.jl")

# ── column generation driver and reporting ───────────────────────────────────────────────
include("cg/options.jl")
include("cg/result.jl")
include("cg/column_generation.jl")
include("stats/report.jl")

# ── experiment config files ──────────────────────────────────────────────────────────────
include("experiments/config.jl")
include("experiments/runner.jl")

# problem
export FlowNetwork, CrewSchedulingInstance, InstanceConfig
export generate_instance, save_instance, load_instance
export n_nodes, n_edges, n_pilots, n_tasks, n_flow_variables, has_inequalities, equality_rows
# solvers
export solve_exact, ExactResult
export solve_column_generation, CGOptions, StabilizationOptions, CGResult
export AbstractMasterSolver, HighsLPMaster, HighsExplicitQPMaster, HighsEliminatedQPMaster
export QPASMaster, QPASOptions
# results and statistics
export objective_value, method_name, total_time, edge_flows, active_columns
export CGStats, ExactStats, compare_results, print_details, write_trace_csv, write_diagnostics_csv
export mean_master_time, total_master_time, first_feasible_iteration, best_lower_bound
# experiments
export Experiment, load_experiments, run_experiment, run_experiments

end # module HotstartedQPAS
