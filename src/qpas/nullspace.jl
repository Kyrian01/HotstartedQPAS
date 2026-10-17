# Implicit nullspace basis of the working-set constraint matrix D_W (thesis §5.1, The Nullspace
# Basis).
#
# By thesis Cor. 5.1.2 (Explicit nullspace vectors) every column nullspace vector is
# zⱼ = eⱼ - e_rep(j) for j ∈ N (and e_{n+i} for each free slack), so Z is fully
# determined by the ordering [S, W, N] and never stored. The same holds for AZ, whose column
# j is a_j - a_rep(j). The solver works with these implicit forms; the explicit matrices
# below exist for debugging, tests and documentation.

"""
    nullspace_basis(ordering, pool, n) -> SparseMatrixCSC

Explicit `n × n` column part `Z_x` of the nullspace basis in the fixed-size representation of
thesis §6.1 (Fixed-length representation): column `j ∈ N` is `eⱼ - e_rep(j)`, all other columns
are zero. The slack block `Z_s` (unit vectors of the free slacks, thesis eq. 5.2) is omitted.
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

The reduced Hessian `H = (AZ_x)ᵀ(AZ_x)` (thesis Lemma 5.2.1, Blocks of the reduced system)
recomputed from scratch, restricted to the nullspace set: `indices` is `N` in increasing order
and `H` the dense `|N| × |N|` matrix with `H[s, t] = u_{j_s}ᵀu_{j_t}`, `j_s = indices[s]`,
`u_j = a_j - a_rep(j)`. Outside `N` the fixed-size reduced Hessian
is the identity. Cost `O(m·|N|²)` with `|N| ≤ m`, independent of the column capacity.

With a row mask `row_free` (rows whose inequality slack is free), those rows are left out and
the result is the effective reduced Hessian `H_eff = (A_eff Z_x)ᵀ(A_eff Z_x)` (thesis eq. 5.9 /
Cor. 5.2.3, eq. 5.10).
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
