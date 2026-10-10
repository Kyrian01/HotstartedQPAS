"""
    ColumnPool(n_rows, n_blocks, capacity)

Append-only store of the columns of the restricted master problem. Column `j` represents an
extreme point (a path) of block `block[j]` and holds

- its coupling coefficients `a_j = A_task xⱼ` (sparse, CSC storage),
- its cost `c_j = cₖᵀ xⱼ`,
- the path itself (edge indices) for solution recovery.

`capacity` is the fixed maximum number of columns `n_max` (thesis §4.5.1); all solver
structures are sized against it so no resizing happens during column generation.
"""
mutable struct ColumnPool
    n_rows::Int
    n_blocks::Int
    capacity::Int
    n::Int
    colptr::Vector{Int}        # length capacity + 1
    rowval::Vector{Int}
    nzval::Vector{Float64}
    cost::Vector{Float64}      # length capacity
    block::Vector{Int}         # length capacity
    paths::Vector{Vector{Int}}
    signatures::Dict{UInt64,Vector{Int}}   # hash(block, path) → columns, duplicate detection
end

function ColumnPool(n_rows::Integer, n_blocks::Integer, capacity::Integer)
    capacity >= n_blocks ||
        throw(ArgumentError("column capacity ($capacity) must be at least the number of blocks ($n_blocks)"))
    colptr = zeros(Int, capacity + 1)
    colptr[1] = 1
    return ColumnPool(n_rows, n_blocks, capacity, 0, colptr, Int[], Float64[],
                      zeros(capacity), zeros(Int, capacity), Vector{Int}[],
                      Dict{UInt64,Vector{Int}}())
end

n_columns(pool::ColumnPool) = pool.n
is_full(pool::ColumnPool) = pool.n >= pool.capacity

"Coupling-row indices and values of column `j` (views)."
@inline column_rows(pool::ColumnPool, j::Integer) =
    @inbounds view(pool.rowval, pool.colptr[j]:(pool.colptr[j + 1] - 1))
@inline column_vals(pool::ColumnPool, j::Integer) =
    @inbounds view(pool.nzval, pool.colptr[j]:(pool.colptr[j + 1] - 1))

_signature(block::Integer, path::AbstractVector{Int}) = hash(path, hash(block))

"`true` if a column with the same block and path is already in the pool."
function contains_column(pool::ColumnPool, block::Integer, path::AbstractVector{Int})
    bucket = get(pool.signatures, _signature(block, path), nothing)
    bucket === nothing && return false
    return any(j -> pool.block[j] == block && pool.paths[j] == path, bucket)
end

"""
    add_column!(pool, block, cost, rows, vals, path) -> j

Append a column and return its index. `rows` must be sorted and duplicate-free.
"""
function add_column!(pool::ColumnPool, block::Integer, cost::Real,
                     rows::AbstractVector{Int}, vals::AbstractVector{Float64},
                     path::AbstractVector{Int})
    is_full(pool) && throw(ArgumentError("column pool is full (capacity $(pool.capacity))"))
    1 <= block <= pool.n_blocks || throw(BoundsError(1:pool.n_blocks, block))
    length(rows) == length(vals) || throw(DimensionMismatch("rows and vals"))
    # append column j in CSC form: rows/values at the end, colptr[j+1] closes it
    j = pool.n + 1
    append!(pool.rowval, rows)
    append!(pool.nzval, vals)
    pool.colptr[j + 1] = pool.colptr[j] + length(rows)
    pool.cost[j] = cost
    pool.block[j] = block
    p = collect(Int, path)
    push!(pool.paths, p)
    # register (block, path) for the duplicate check during pricing
    push!(get!(Vector{Int}, pool.signatures, _signature(block, p)), j)
    pool.n = j
    return j
end

# ── sparse kernels over the first n columns ─────────────────────────────────────────────

"`a_jᵀ v` for a dense vector `v` of length `n_rows`."
@inline function column_dot(pool::ColumnPool, j::Integer, v::AbstractVector{Float64})
    s = 0.0
    @inbounds for t in pool.colptr[j]:(pool.colptr[j + 1] - 1)
        s += pool.nzval[t] * v[pool.rowval[t]]
    end
    return s
end

"`y += α a_j`."
@inline function column_axpy!(y::AbstractVector{Float64}, α::Float64, pool::ColumnPool, j::Integer)
    @inbounds for t in pool.colptr[j]:(pool.colptr[j + 1] - 1)
        y[pool.rowval[t]] += α * pool.nzval[t]
    end
    return y
end

"`(a_j)ᵢ`, the coefficient of column `j` in coupling row `i` (0 if absent)."
@inline function column_entry(pool::ColumnPool, j::Integer, i::Integer)
    lo, hi = pool.colptr[j], pool.colptr[j + 1] - 1
    # row indices of a column are sorted (add_column!): binary search for i
    t = searchsortedfirst(view(pool.rowval, lo:hi), i)
    @inbounds return t <= hi - lo + 1 && pool.rowval[lo + t - 1] == i ? pool.nzval[lo + t - 1] : 0.0
end

"Bytes held by the column pool (coefficients, costs, blocks, stored paths, signatures)."
function pool_memory(pool::ColumnPool)
    b = sizeof(pool.colptr) + sizeof(pool.rowval) + sizeof(pool.nzval) + sizeof(pool.cost) +
        sizeof(pool.block)
    # each path is a separate vector of edge indices
    b += sum(sizeof, pool.paths; init = 0)
    return b
end

"`‖a_j‖²`."
@inline function column_norm2(pool::ColumnPool, j::Integer)
    s = 0.0
    @inbounds for t in pool.colptr[j]:(pool.colptr[j + 1] - 1)
        s += pool.nzval[t]^2
    end
    return s
end

"`y = A x` over the first `n` columns."
function coupling_product!(y::AbstractVector{Float64}, pool::ColumnPool, x::AbstractVector{Float64})
    fill!(y, 0.0)
    @inbounds for j in 1:pool.n
        xj = x[j]
        xj == 0.0 && continue
        column_axpy!(y, xj, pool, j)
    end
    return y
end

"`y[j] = a_jᵀ v` for `j = 1:n`."
function coupling_transpose_product!(y::AbstractVector{Float64}, pool::ColumnPool,
                                     v::AbstractVector{Float64})
    @inbounds for j in 1:pool.n
        y[j] = column_dot(pool, j, v)
    end
    return y
end

"Total cost `cᵀx` over the first `n` columns."
function total_cost(pool::ColumnPool, x::AbstractVector{Float64})
    s = 0.0
    @inbounds for j in 1:pool.n
        s += pool.cost[j] * x[j]
    end
    return s
end

"""
    column_matrix(pool) -> SparseMatrixCSC

The coupling matrix `AX` of the current columns as a `n_rows × n` sparse matrix (a copy).
"""
column_matrix(pool::ColumnPool) =
    SparseMatrixCSC(pool.n_rows, pool.n, pool.colptr[1:(pool.n + 1)],
                    pool.rowval[1:(pool.colptr[pool.n + 1] - 1)],
                    pool.nzval[1:(pool.colptr[pool.n + 1] - 1)])

Base.show(io::IO, pool::ColumnPool) =
    print(io, "ColumnPool(", pool.n, "/", pool.capacity, " columns, ", pool.n_rows, " rows, ",
          pool.n_blocks, " blocks)")
