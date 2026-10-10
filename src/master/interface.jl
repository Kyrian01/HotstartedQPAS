# Interface between the column generation driver and the restricted master problem solvers.
#
# A master solver owns its own copy of whatever it needs (a HiGHS instance, a QPAS state, …)
# and is told about new columns incrementally. The driver never looks inside it.
#
# Required methods for `M <: AbstractMasterSolver`:
#
#   method_name(m)                 -> String
#   is_stabilized(m)               -> Bool    (true: solves the proximal QP master, needs a center)
#   supports_inequalities(m)       -> Bool    (inequality coupling rows supported?)
#   initialize!(m, spec, pool)     build the master over the initial columns in `pool`
#   add_columns!(m, pool, cols)    columns `cols` (a range) were appended to `pool`
#   solve!(m)                      -> MasterSolution
#
# Stabilized masters additionally implement
#
#   set_center!(m, center)         set the stability center π̂
#
# Optional:
#
#   solver_details(m)              -> Vector{Pair{String,Any}} of solver-specific statistics
#   release!(m)                    free external resources early

"""
Supertype of all restricted master problem solvers. See `src/master/interface.jl` for the
methods a subtype must implement.
"""
abstract type AbstractMasterSolver end

"""
    MasterSpec

Static data of the restricted master problem, handed to [`initialize!`](@ref):
coupling row bounds `row_lower ≤ AXλ ≤ row_upper`, number of blocks (convexity rows),
column capacity `n_max`, and the stabilization parameter `delta` (NaN when unstabilized).
"""
struct MasterSpec
    row_lower::Vector{Float64}
    row_upper::Vector{Float64}
    n_blocks::Int
    capacity::Int
    delta::Float64
end

n_rows(spec::MasterSpec) = length(spec.row_lower)
equality_only(spec::MasterSpec) = spec.row_lower == spec.row_upper

"""
    MasterSolution

Result of one restricted master solve. The vectors are reused between iterations.

- `weights`: column weights `λ` (length = capacity; entries beyond the column count are 0)
- `π`: coupling duals passed to pricing. For stabilized masters `π = π̂ + w`
  (thesis eq. 4.4).
- `μ`: convexity duals, one per block
- `objective`: master objective (including Big-M penalties or the proximal term)
- `inner_iterations`: simplex / QP / active-set iterations of this solve
- `status`: `:optimal` or a failure symbol
"""
mutable struct MasterSolution
    weights::Vector{Float64}
    π::Vector{Float64}
    μ::Vector{Float64}
    objective::Float64
    inner_iterations::Int
    status::Symbol
end

MasterSolution(spec::MasterSpec) =
    MasterSolution(zeros(spec.capacity), zeros(n_rows(spec)), zeros(spec.n_blocks), NaN, 0,
                   :not_solved)

function initialize! end
function add_columns! end
function set_center! end
function method_name end
function is_stabilized end
function supports_inequalities end

solver_details(::AbstractMasterSolver) = Pair{String,Any}[]
release!(::AbstractMasterSolver) = nothing

set_center!(m::AbstractMasterSolver, ::AbstractVector) =
    throw(ArgumentError("$(method_name(m)) is not a stabilized master"))

"Stabilized objective `cᵀλ + δ/2‖π‖² - δ/2‖π̂‖²` given `π = π̂ + (b - Aλ)/δ` (thesis eq. 4.3a)."
function stabilized_objective(cost_term::Float64, π::AbstractVector{Float64},
                              center::AbstractVector{Float64}, delta::Float64)
    return cost_term + 0.5 * delta * (sum(abs2, π) - sum(abs2, center))
end
