# KKT system of the regularized master problem in nullspace form (thesis §4.3–4.4).
#
# The KKT system (thesis eq. 4.17) is never formed. With p = Zy it reduces to
#
#     H y = -δ Zᵀg,     H = (AZ)ᵀ(AZ),
#
# where H is kept as an LDLᵀ factor of fixed size n_max (identity on indices outside N) and
# updated with CHOLMOD row additions/deletions whenever N changes:
#
#   index j enters N  →  rowadd(j, H[:, j])           (reduction, thesis §4.4.2)
#   index j leaves N  →  rowdel(j)                    (expansion, thesis §4.4.1)
#   representative of block b changes  →  rowdel + rowadd for every j ∈ N of block b
#
# Column j of H is assembled on demand from the column pool:
#   u = a_j - a_rep(j),   H[k, j] = (a_k - a_rep(k))ᵀ u   (k ∈ N),   H[j, j] = uᵀu,
# which costs O((|N| + p) · nnz per column), independent of the total number of columns.

"""
    KKTSystem(n_blocks, n_rows, capacity; debug = false)

Linear-algebra state of the nullspace KKT system:

- `ordering::ColumnOrdering`: the partition [S, W, N]; it fully determines the nullspace basis
  `Z` and `AZ` (see `nullspace.jl`), which are therefore implicit
- `factor::Cholmod.LDLtFactor`: LDLᵀ of the fixed-size reduced Hessian
- `hdiag`: `H[j, j]` at activation, the scale for pivot and singularity tests
- `updates_since_refactor`, `total_updates`: CHOLMOD row additions + deletions
- scratch buffers for Hessian column assembly
- `debug_H`: explicit mirror of the reduced Hessian (active entries only, `(k, j) => H[k, j]`),
  kept only in debug mode
"""
mutable struct KKTSystem
    ordering::ColumnOrdering
    factor::Cholmod.LDLtFactor
    hdiag::Vector{Float64}
    u::Vector{Float64}                 # dense n_rows scatter of a_j - a_rep(j); zero between uses
    rep_dot::Vector{Float64}           # (a_rep(b))ᵀ u per block
    rep_dot_valid::Vector{Bool}
    col_rows::Vector{Int}              # assembled Hessian column (indices, values)
    col_vals::Vector{Float64}
    affected::Vector{Int}              # ν: N-columns of a block whose representative changes
    updates_since_refactor::Int
    total_updates::Int
    debug_H::Union{Nothing,Dict{Tuple{Int,Int},Float64}}
end

function KKTSystem(n_blocks::Integer, n_rows::Integer, capacity::Integer; debug::Bool = false)
    return KKTSystem(ColumnOrdering(n_blocks, capacity), Cholmod.LDLtFactor(capacity),
                     zeros(capacity), zeros(n_rows), zeros(n_blocks), fill(false, n_blocks),
                     zeros(Int, capacity), zeros(capacity), Int[], 0, 0,
                     debug ? Dict{Tuple{Int,Int},Float64}() : nothing)
end

"""
    hessian_column!(kkt, pool, j) -> (nnz, hjj)

Assemble column `j` of the reduced Hessian against the current set `N` (which must not
contain `j`) into `kkt.col_rows[1:nnz]`, `kkt.col_vals[1:nnz]`. The last entry is the
diagonal `hjj = ‖a_j - a_rep(j)‖²`.
"""
function hessian_column!(kkt::KKTSystem, pool::ColumnPool, j::Int)
    ord = kkt.ordering
    block = pool.block
    u = kkt.u
    r = representative(ord, block, j)

    column_axpy!(u, 1.0, pool, j)
    column_axpy!(u, -1.0, pool, r)
    fill!(kkt.rep_dot_valid, false)

    nnz = 0
    rows, vals = kkt.col_rows, kkt.col_vals
    @inbounds for k in items(ord.N)
        b = block[k]
        if !kkt.rep_dot_valid[b]
            kkt.rep_dot[b] = column_dot(pool, ord.rep[b], u)
            kkt.rep_dot_valid[b] = true
        end
        h = column_dot(pool, k, u) - kkt.rep_dot[b]
        if h != 0.0
            nnz += 1
            rows[nnz] = k
            vals[nnz] = h
        end
    end
    hjj = column_dot(pool, j, u) - column_dot(pool, r, u)
    nnz += 1
    rows[nnz] = j
    vals[nnz] = hjj

    @inbounds for i in column_rows(pool, j)
        u[i] = 0.0
    end
    @inbounds for i in column_rows(pool, r)
        u[i] = 0.0
    end
    return nnz, hjj
end

"""
    insert_column!(kkt, j, nnz, hjj)

Activate index `j` in the factor with the column previously assembled by
[`hessian_column!`](@ref) and move `j` into `N`.
"""
function insert_column!(kkt::KKTSystem, j::Int, nnz::Int, hjj::Float64)
    Cholmod.rowadd!(kkt.factor, j, kkt.col_rows, kkt.col_vals, nnz)
    kkt.hdiag[j] = hjj
    kkt.updates_since_refactor += 1
    kkt.total_updates += 1
    if kkt.debug_H !== nothing
        H = kkt.debug_H
        for t in 1:nnz
            k = kkt.col_rows[t]
            H[(k, j)] = kkt.col_vals[t]
            H[(j, k)] = kkt.col_vals[t]
        end
    end
    move_to_N!(kkt.ordering, j)
    return nothing
end

"Assemble column `j` and activate it (no singularity check)."
function activate!(kkt::KKTSystem, pool::ColumnPool, j::Int)
    nnz, hjj = hessian_column!(kkt, pool, j)
    insert_column!(kkt, j, nnz, hjj)
    return nothing
end

"""
    deactivate!(kkt, j)

Reset row/column `j` of the factor to the identity and remove `j` from `N`. The caller
decides where `j` goes next (W or S).
"""
function deactivate!(kkt::KKTSystem, j::Int)
    Cholmod.rowdel!(kkt.factor, j)
    kkt.updates_since_refactor += 1
    kkt.total_updates += 1
    if kkt.debug_H !== nothing
        for k in items(kkt.ordering.N)
            delete!(kkt.debug_H, (k, j))
            delete!(kkt.debug_H, (j, k))
        end
    end
    delete!(kkt.ordering.N, j)
    return nothing
end

"""
    min_pivot_ratio(kkt) -> Float64

Smallest `D[k] / H[k, k]` over `k ∈ N` (`Inf` if `N` is empty). Values near zero indicate a
(numerically) singular reduced Hessian.
"""
min_pivot_ratio(kkt::KKTSystem) =
    Cholmod.min_scaled_pivot(kkt.factor, items(kkt.ordering.N), kkt.hdiag)

"""
    refactor!(kkt, pool)

Rebuild the factorization from scratch: reset to the identity and re-activate every index
of `N`. Used for periodic refactorization and as a fallback when pivots degrade.

Indices are re-activated in increasing order: with CHOLMOD's natural ordering every
`rowadd` then appends the last row of the factor (one triangular solve, no update of
trailing columns).
"""
function refactor!(kkt::KKTSystem, pool::ColumnPool)
    Cholmod.reset!(kkt.factor)
    ord = kkt.ordering
    order = sort!(copy(items(ord.N)))
    empty!(ord.N)
    kkt.debug_H === nothing || empty!(kkt.debug_H)
    for j in order
        activate!(kkt, pool, j)
    end
    kkt.updates_since_refactor = 0
    return nothing
end

"""
    verify_kkt(kkt, pool; tol = 1e-8) -> Float64

Debug check, run after every QPAS iteration in debug mode:

1. the ordering invariants hold;
2. the reduced Hessian on `N`, recomputed from scratch as `(AZ)ᵀ(AZ)`, equals the
   incrementally maintained mirror `debug_H` (if present);
3. the CHOLMOD factor reproduces it: `(L D Lᵀ)[N, N] = H`, and every inactive index has
   `D[k] = 1` and no coupling to `N` in `L`.

All work is restricted to `N` (`|N| ≤ m`), so this stays usable on full-size instances.
Throws on failure; returns the relative factorization error.
"""
function verify_kkt(kkt::KKTSystem, pool::ColumnPool; tol::Real = 1e-8)
    ord = kkt.ordering
    check_ordering(ord, pool, pool.n)
    idx, H = reduced_hessian(ord, pool)
    scale = max(1.0, maximum(abs, H; init = 0.0))

    if kkt.debug_H !== nothing
        pos = Dict(j => t for (t, j) in enumerate(idx))
        mirror = zeros(length(idx), length(idx))
        for ((k, j), v) in kkt.debug_H
            (haskey(pos, k) && haskey(pos, j)) ||
                error("KKT verification: mirror holds entry ($k, $j) outside N")
            mirror[pos[k], pos[j]] = v
        end
        err = maximum(abs, mirror - H; init = 0.0) / scale
        err <= tol || error("KKT verification: tracked reduced Hessian is off by $err")
    end

    L, d = Cholmod.factor_matrices(kkt.factor)
    LN = L[idx, :]
    err = maximum(abs, Matrix(LN * Diagonal(d) * LN') - H; init = 0.0) / scale
    err <= tol || error("KKT verification: LDLᵀ factor is off by $err on N")
    active = falses(length(d))
    active[idx] .= true
    inactive = findall(!, active)
    maximum(abs, d[inactive] .- 1.0; init = 0.0) <= tol ||
        error("KKT verification: an inactive pivot differs from 1")
    maximum(abs, L[idx, inactive]; init = 0.0) <= tol * scale ||
        error("KKT verification: L couples N to inactive indices")
    return err
end
