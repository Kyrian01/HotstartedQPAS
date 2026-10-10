# Implicit nullspace basis of the working-set constraint matrix D_W (thesis §4.3.1).
#
# By Corollary 4.3.4 every nullspace vector is zⱼ = eⱼ - e_rep(j) for j ∈ N, so Z is fully
# determined by the ordering [S, W, N] and never stored. The same holds for AZ, whose column
# j is a_j - a_rep(j). The solver works with these implicit forms; the explicit matrices
# below exist for debugging, tests and documentation.

"""
    nullspace_basis(ordering, pool, n) -> SparseMatrixCSC

Explicit `n × n` nullspace basis in the fixed-size representation of thesis §4.4:
column `j ∈ N` is `eⱼ - e_rep(j)`, all other columns are zero.
"""
function nullspace_basis(ord::ColumnOrdering, pool::ColumnPool, n::Integer)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for j in items(ord.N)
        push!(rows, j, representative(ord, pool.block, j))
        push!(cols, j, j)
        push!(vals, 1.0, -1.0)
    end
    return sparse(rows, cols, vals, n, n)
end

"""
    reduced_hessian(ordering, pool; row_free = nothing) -> (indices, H)

The reduced Hessian `H = (AZ)ᵀ(AZ)` (thesis eq. 4.31) recomputed from scratch, restricted to
the nullspace set: `indices` is `N` in increasing order and `H` the dense `|N| × |N|` matrix
with `H[s, t] = u_iᵀu_j`, `u_j = a_j - a_rep(j)`. Outside `N` the fixed-size reduced Hessian
is the identity. Cost `O(m·|N|²)` with `|N| ≤ m`, independent of the column capacity.

With a row mask `row_free` (rows whose inequality slack is free), those rows are left out and
the result is the effective reduced Hessian `H_eff = (A_eff Z)ᵀ(A_eff Z)`.
"""
function reduced_hessian(ord::ColumnOrdering, pool::ColumnPool; row_free = nothing)
    indices = sort(items(ord.N))
    # U = AZ restricted to N: column t is a_j - a_rep(j)
    U = zeros(pool.n_rows, length(indices))
    for (t, j) in enumerate(indices)
        column_axpy!(view(U, :, t), 1.0, pool, j)
        column_axpy!(view(U, :, t), -1.0, pool, representative(ord, pool.block, j))
    end
    # drop the rows that are not part of A_eff
    row_free === nothing || (U[row_free, :] .= 0.0)
    return indices, U' * U
end
