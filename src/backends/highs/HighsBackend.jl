"""
    HighsBackend

Small object-oriented wrapper around a persistent HiGHS instance, built on the C API exposed
by `HiGHS.jl`. It provides what the master problems and the exact solver need: passing an
LP, appending rows and columns, passing a Hessian, changing costs, solving and reading
primal/dual solutions and iteration counts.

All index arguments are 1-based Julia indices; conversion to HiGHS' 0-based `HighsInt`
arrays happens here, using reusable buffers.
"""
module HighsBackend

import HiGHS
using SparseArrays: SparseMatrixCSC

export HighsModel, set_option!, pass_lp!, append_rows!, append_columns!, pass_hessian!
export change_costs!, run!, get_solution!, num_cols, num_rows, model_objective
export iteration_counts, model_status_symbol

const HighsInt = HiGHS.HighsInt

"""
    HighsModel(; output = false)

Owning handle to a HiGHS instance. Freed by a finalizer or by `close`.
"""
mutable struct HighsModel
    ptr::Ptr{Cvoid}
    # reusable conversion buffers
    starts::Vector{HighsInt}
    index::Vector{HighsInt}
    values::Vector{Float64}
end

function HighsModel(; output::Bool = false)
    ptr = HiGHS.Highs_create()
    ptr == C_NULL && error("Highs_create failed")
    model = HighsModel(ptr, HighsInt[], HighsInt[], Float64[])
    finalizer(close, model)
    set_option!(model, "output_flag", output)
    return model
end

function Base.close(model::HighsModel)
    if model.ptr != C_NULL
        HiGHS.Highs_destroy(model.ptr)
        model.ptr = C_NULL
    end
    return nothing
end

@inline function _check(ret, what::AbstractString)
    # kHighsStatusError = -1; warnings (1) are accepted
    ret == HiGHS.kHighsStatusError && error("HiGHS: $what returned an error")
    return nothing
end

"""
    set_option!(model, name, value)

Set a HiGHS option; the C setter is chosen from the Julia type of `value`.
"""
set_option!(m::HighsModel, name::AbstractString, value::Bool) =
    _check(HiGHS.Highs_setBoolOptionValue(m.ptr, name, HighsInt(value)), "set option $name")
set_option!(m::HighsModel, name::AbstractString, value::Integer) =
    _check(HiGHS.Highs_setIntOptionValue(m.ptr, name, HighsInt(value)), "set option $name")
set_option!(m::HighsModel, name::AbstractString, value::AbstractFloat) =
    _check(HiGHS.Highs_setDoubleOptionValue(m.ptr, name, Float64(value)), "set option $name")
set_option!(m::HighsModel, name::AbstractString, value::AbstractString) =
    _check(HiGHS.Highs_setStringOptionValue(m.ptr, name, value), "set option $name")

num_cols(m::HighsModel) = Int(HiGHS.Highs_getNumCol(m.ptr))
num_rows(m::HighsModel) = Int(HiGHS.Highs_getNumRow(m.ptr))
model_objective(m::HighsModel) = HiGHS.Highs_getObjectiveValue(m.ptr)

# Fill the reusable 0-based buffers from a 1-based CSC slice.
function _load_csc!(m::HighsModel, colptr::AbstractVector{<:Integer},
                    rowval::AbstractVector{<:Integer}, nzval::AbstractVector{<:Real})
    ncol = length(colptr) - 1
    base = colptr[1]
    nnz = colptr[end] - base
    resize!(m.starts, max(ncol, 1))
    resize!(m.index, max(nnz, 1))
    resize!(m.values, max(nnz, 1))
    @inbounds for j in 1:ncol
        m.starts[j] = colptr[j] - base
    end
    @inbounds for t in 1:nnz
        m.index[t] = rowval[base + t - 1] - 1
        m.values[t] = nzval[base + t - 1]
    end
    return ncol, nnz
end

"""
    pass_lp!(model, A, cost, col_lower, col_upper, row_lower, row_upper; offset = 0.0)

Replace the model by the minimisation LP
`min cost'x + offset  s.t.  row_lower ≤ A x ≤ row_upper,  col_lower ≤ x ≤ col_upper`.
"""
function pass_lp!(m::HighsModel, A::SparseMatrixCSC, cost::Vector{Float64},
                  col_lower::Vector{Float64}, col_upper::Vector{Float64},
                  row_lower::Vector{Float64}, row_upper::Vector{Float64}; offset::Real = 0.0)
    nrow, ncol = size(A)
    length(cost) == length(col_lower) == length(col_upper) == ncol ||
        throw(DimensionMismatch("column data must have length $ncol"))
    length(row_lower) == length(row_upper) == nrow ||
        throw(DimensionMismatch("row bounds must have length $nrow"))
    _, nnz = _load_csc!(m, A.colptr, A.rowval, A.nzval)
    ret = HiGHS.Highs_passLp(m.ptr, ncol, nrow, nnz, HiGHS.kHighsMatrixFormatColwise,
                             HiGHS.kHighsObjSenseMinimize, Float64(offset),
                             cost, col_lower, col_upper, row_lower, row_upper,
                             m.starts, m.index, m.values)
    _check(ret, "Highs_passLp")
    return m
end

"""
    append_rows!(model, lower, upper)

Append rows without coefficients and with bounds `lower ≤ row ≤ upper`.
"""
function append_rows!(m::HighsModel, lower::Vector{Float64}, upper::Vector{Float64})
    n = length(lower)
    length(upper) == n || throw(DimensionMismatch("row bounds"))
    n == 0 && return m
    ret = HiGHS.Highs_addRows(m.ptr, n, lower, upper, 0, C_NULL, C_NULL, C_NULL)
    _check(ret, "Highs_addRows")
    return m
end

"""
    append_columns!(model, cost, lower, upper, colptr, rowval, nzval)

Append `length(cost)` columns given in 1-based CSC form (`colptr` has one more entry than
there are columns; it may start at any offset).
"""
function append_columns!(m::HighsModel, cost::AbstractVector{Float64},
                      lower::AbstractVector{Float64}, upper::AbstractVector{Float64},
                      colptr::AbstractVector{<:Integer}, rowval::AbstractVector{<:Integer},
                      nzval::AbstractVector{<:Real})
    ncol, nnz = _load_csc!(m, colptr, rowval, nzval)
    ncol == length(cost) == length(lower) == length(upper) ||
        throw(DimensionMismatch("append_columns!: inconsistent column data"))
    ncol == 0 && return m
    ret = HiGHS.Highs_addCols(m.ptr, ncol, cost, lower, upper, nnz,
                              m.starts, m.index, m.values)
    _check(ret, "Highs_addCols")
    return m
end

"""
    pass_hessian!(model, dim, colptr, rowval, nzval)

Set the Hessian `Q` of the objective `½ xᵀQx + cᵀx`, given as the lower triangle in 1-based
CSC form. `dim` must equal the current number of columns.
"""
function pass_hessian!(m::HighsModel, dim::Integer, colptr::AbstractVector{<:Integer},
                       rowval::AbstractVector{<:Integer}, nzval::AbstractVector{<:Real})
    length(colptr) == dim + 1 || throw(DimensionMismatch("Hessian colptr must have dim+1 entries"))
    _, nnz = _load_csc!(m, colptr, rowval, nzval)
    ret = HiGHS.Highs_passHessian(m.ptr, dim, nnz, HiGHS.kHighsHessianFormatTriangular,
                                  m.starts, m.index, m.values)
    _check(ret, "Highs_passHessian")
    return m
end

"""
    change_costs!(model, first, cost)

Overwrite the costs of columns `first : first + length(cost) - 1`.
"""
function change_costs!(m::HighsModel, first::Integer, cost::AbstractVector{Float64})
    isempty(cost) && return m
    ret = HiGHS.Highs_changeColsCostByRange(m.ptr, first - 1, first + length(cost) - 2, cost)
    _check(ret, "Highs_changeColsCostByRange")
    return m
end

"""
    run!(model) -> Symbol

Solve the current model and return its model status as a `Symbol` (e.g. `:optimal`).
"""
function run!(m::HighsModel)
    ret = HiGHS.Highs_run(m.ptr)
    _check(ret, "Highs_run")
    return model_status_symbol(HiGHS.Highs_getModelStatus(m.ptr))
end

"""
    get_solution!(model, col_value, col_dual, row_value, row_dual)

Copy the primal and dual solution into preallocated vectors of length `num_cols` /
`num_rows`. Duals follow the HiGHS convention `col_dual = c + Qx - Aᵀ row_dual`.
"""
function get_solution!(m::HighsModel, col_value::Vector{Float64}, col_dual::Vector{Float64},
                       row_value::Vector{Float64}, row_dual::Vector{Float64})
    nc, nr = num_cols(m), num_rows(m)
    (length(col_value) >= nc && length(col_dual) >= nc &&
     length(row_value) >= nr && length(row_dual) >= nr) ||
        throw(DimensionMismatch("solution buffers too small"))
    _check(HiGHS.Highs_getSolution(m.ptr, col_value, col_dual, row_value, row_dual),
           "Highs_getSolution")
    return nothing
end

function _int_info(m::HighsModel, name::AbstractString)
    v = Ref{HighsInt}(0)
    HiGHS.Highs_getIntInfoValue(m.ptr, name, v)
    return Int(max(v[], 0))
end

"""
    iteration_counts(model) -> NamedTuple

Iteration counters of the last run: `simplex`, `ipm`, `qp` and `crossover`.
"""
iteration_counts(m::HighsModel) = (
    simplex = _int_info(m, "simplex_iteration_count"),
    ipm = _int_info(m, "ipm_iteration_count"),
    qp = _int_info(m, "qp_iteration_count"),
    crossover = _int_info(m, "crossover_iteration_count"),
)

const _MODEL_STATUS = Dict(
    HiGHS.kHighsModelStatusNotset => :not_set,
    HiGHS.kHighsModelStatusLoadError => :load_error,
    HiGHS.kHighsModelStatusModelError => :model_error,
    HiGHS.kHighsModelStatusPresolveError => :presolve_error,
    HiGHS.kHighsModelStatusSolveError => :solve_error,
    HiGHS.kHighsModelStatusPostsolveError => :postsolve_error,
    HiGHS.kHighsModelStatusModelEmpty => :model_empty,
    HiGHS.kHighsModelStatusOptimal => :optimal,
    HiGHS.kHighsModelStatusInfeasible => :infeasible,
    HiGHS.kHighsModelStatusUnboundedOrInfeasible => :unbounded_or_infeasible,
    HiGHS.kHighsModelStatusUnbounded => :unbounded,
    HiGHS.kHighsModelStatusObjectiveBound => :objective_bound,
    HiGHS.kHighsModelStatusObjectiveTarget => :objective_target,
    HiGHS.kHighsModelStatusTimeLimit => :time_limit,
    HiGHS.kHighsModelStatusIterationLimit => :iteration_limit,
    HiGHS.kHighsModelStatusUnknown => :unknown,
    HiGHS.kHighsModelStatusSolutionLimit => :solution_limit,
    HiGHS.kHighsModelStatusInterrupt => :interrupt,
)

model_status_symbol(code) = get(_MODEL_STATUS, code, :unknown)

end # module HighsBackend
