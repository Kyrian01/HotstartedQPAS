"""
    DenseLDLt

Dense, updatable LDLᵀ factorization of the active block of the effective reduced Hessian. It
implements the same interface as the CHOLMOD backend (`LDLtInterface`), but stores only the
active indices, packed into *slots* `1:k` in the order they were activated:

    M[active, active] = P L D Lᵀ Pᵀ,   L unit lower triangular k × k,  D = diag(d),

where slot `s` holds index `index[s]` (`slot[index[s]] = s`). Every other index of the `n × n`
matrix `M` is an identity row and is not stored.

Because `|N| ≤ rank(A_eff) ≤ m` (the reduced Hessian stays positive definite), the factor
never exceeds `m × m`, independent of the column capacity `n_max`. In the crew scheduling
instances `H_eff` is ≈ 100 % dense, so a dense factor stores nothing a sparse one would not,
and it never accumulates structural zeros.

Update rules (derivations in `docs/dense_ldlt_backend.md`), with `k` active slots:

| operation               | rule                                                           | cost        |
|:------------------------|:---------------------------------------------------------------|:------------|
| `rowadd!` (activate)    | append slot `k+1`: `L w = h`, `ℓ = D⁻¹w`, `d = h_jj - wᵀD⁻¹w`  | `O(k²)`     |
| `rowdel!` (deactivate)  | drop slot `p`, trailing block `+ d_p ℓℓᵀ` (method C1), shift   | `O((k-p)²) + O(p(k-p))` |
| `updown!` (`± vvᵀ`)     | method C1 from the first nonzero of `v`                        | `O(k²)`     |
| `solve!`                | `L z = b`, `z ./= d`, `Lᵀx = z`                                | `O(k²)`     |

The triangular solves call BLAS `trsv` on the leading `k × k` block of a column-major
`cap × cap` array; the rank-one kernels are column-oriented Julia loops over the same array.
"""
module DenseLDLt

using LinearAlgebra
using LinearAlgebra: BLAS
using SparseArrays: SparseMatrixCSC, sparse

import ..LDLtInterface: UpdatableLDLt, rowadd!, rowdel!, updown!, solve!, rhs_buffer, pivot,
                        min_scaled_pivot, reset!, dimension, factor_matrices, factor_stats,
                        last_reach, backend_name, needs_periodic_refactor, backend_stats

export DenseLDLtFactor

"""
    DenseLDLtFactor(n; capacity = min(n, 64))

LDLᵀ factor of a symmetric `n × n` matrix that starts as the identity, storing only the
active indices. `capacity` is the initial number of slots; it grows (doubling, at most `n`)
when more indices are active.
"""
mutable struct DenseLDLtFactor <: UpdatableLDLt
    n::Int                      # dimension of the index space (n_max)
    k::Int                      # active slots 1:k
    cap::Int                    # allocated slots
    L::Matrix{Float64}          # cap × cap; strictly lower part of the leading k × k block is L
    d::Vector{Float64}          # D[1:k]
    slot::Vector{Int}           # slot of index j (0: inactive, identity row)
    index::Vector{Int}          # index held by slot s
    rhs::Vector{Float64}        # right-hand side, length n (indexed like M)
    sol::Vector{Float64}        # solution, length n
    w::Vector{Float64}          # slot-space scratch (solves, new row)
    z::Vector{Float64}          # slot-space scratch (rank-one vector)
    reach::Int                  # rows touched by the last sparse solve
    # statistics
    peak_k::Int
    grows::Int
    deletions::Int
    deleted_trailing::Int       # Σ (k - p) over deletions: trailing block sizes updated
    deleted_size::Int           # Σ k over deletions
    peak_bytes::Int
    closed::Bool
end

function DenseLDLtFactor(n::Integer; capacity::Integer = min(n, 64))
    n > 0 || throw(ArgumentError("DenseLDLtFactor dimension must be positive, got $n"))
    cap = clamp(Int(capacity), 1, Int(n))
    F = DenseLDLtFactor(Int(n), 0, cap, zeros(cap, cap), zeros(cap), zeros(Int, n),
                        zeros(Int, cap), zeros(n), zeros(n), zeros(cap), zeros(cap), 0,
                        0, 0, 0, 0, 0, 0, false)
    F.peak_bytes = _bytes(F)
    return F
end

dimension(F::DenseLDLtFactor) = F.n
backend_name(::DenseLDLtFactor) = "dense"
needs_periodic_refactor(::DenseLDLtFactor) = false     # deletions compact the factor
rhs_buffer(F::DenseLDLtFactor) = F.rhs
last_reach(F::DenseLDLtFactor) = F.reach

"Number of active indices (slots in use)."
active_count(F::DenseLDLtFactor) = F.k

_bytes(F::DenseLDLtFactor) =
    sizeof(F.L) + sizeof(F.d) + sizeof(F.slot) + sizeof(F.index) + sizeof(F.rhs) +
    sizeof(F.sol) + sizeof(F.w) + sizeof(F.z)

# Make room for `need` slots: grow to max(need, 2·cap), at most n; keeps the leading k × k block.
function _ensure_capacity!(F::DenseLDLtFactor, need::Int)
    need <= F.cap && return nothing
    need <= F.n || throw(ArgumentError("more than n = $(F.n) active indices"))
    cap = min(F.n, max(need, 2 * F.cap))
    k = F.k
    L = zeros(cap, cap)
    @inbounds for c in 1:k, r in (c + 1):k
        L[r, c] = F.L[r, c]
    end
    F.L = L
    F.d = copyto!(zeros(cap), 1, F.d, 1, k)
    F.index = copyto!(zeros(Int, cap), 1, F.index, 1, k)
    F.w = zeros(cap)
    F.z = zeros(cap)
    F.cap = cap
    F.grows += 1
    F.peak_bytes = max(F.peak_bytes, _bytes(F))
    return nothing
end

# Slot of active index j; errors if j is an identity row (inactive).
@inline function _active_slot(F::DenseLDLtFactor, j::Integer, what::String)
    s = F.slot[j]
    s == 0 && throw(ArgumentError("$what: index $j is not active (identity row)"))
    return s
end

# ── rowadd: append a slot ───────────────────────────────────────────────────────────────

"""
    rowadd!(F, j, rows, vals, nnz) -> Bool

Activate index `j`: row/column `j` of the factorized matrix becomes the sparse vector
`(rows, vals)[1:nnz]`, which must contain `j` itself and otherwise only active indices. The
new index is appended as the last slot `s = k+1`:

    [L₁₁ 0; ℓᵀ 1] [D₁ 0; 0 dₛ] [L₁₁ᵀ ℓ; 0 1] = [M₁₁ h; hᵀ h_jj]
    ⇒  L₁₁ w = h,  ℓ = D₁⁻¹ w,  dₛ = h_jj - wᵀ D₁⁻¹ w   (the Schur complement).

Nothing else changes. Returns `dₛ > 0` (the extended matrix is positive definite).
"""
function rowadd!(F::DenseLDLtFactor, j::Integer, rows::AbstractVector{<:Integer},
                 vals::AbstractVector{Float64}, nnz::Integer)
    _check_open(F)
    1 <= j <= F.n || throw(BoundsError(F, j))
    F.slot[j] == 0 || throw(ArgumentError("rowadd!: index $j is already active"))
    s = F.k + 1
    _ensure_capacity!(F, s)
    w = F.w
    km = s - 1
    @inbounds for t in 1:km
        w[t] = 0.0
    end
    hjj = 0.0
    @inbounds for t in 1:nnz
        i = rows[t]
        if i == j
            hjj += vals[t]
        else
            w[_active_slot(F, i, "rowadd!")] = vals[t]
        end
    end
    L, d = F.L, F.d
    if km > 0
        # w ← L₁₁⁻¹ h (unit lower triangular)
        BLAS.trsv!('L', 'N', 'U', view(L, 1:km, 1:km), view(w, 1:km))
    end
    ds = hjj
    @inbounds for t in 1:km
        ℓ = w[t] / d[t]
        L[s, t] = ℓ                     # row s of L (strided write, O(k))
        ds -= w[t] * ℓ
    end
    @inbounds begin
        d[s] = ds
        F.index[s] = j
        F.slot[j] = s
    end
    F.k = s
    F.peak_k = max(F.peak_k, s)
    return ds > 0.0
end

# ── rowdel: remove a slot ───────────────────────────────────────────────────────────────

"""
    rowdel!(F, j)

Deactivate index `j` (row/column `j` becomes the identity). With `p = slot[j]` and the
partition into slots before `p`, `p`, after `p`:

    M₃₃ = L₃₁D₁L₃₁ᵀ + d_p ℓ₃₂ℓ₃₂ᵀ + L₃₃D₃L₃₃ᵀ,

so removing slot `p` keeps `L₁₁`, `L₃₁` and replaces the trailing factor by that of
`L₃₃D₃L₃₃ᵀ + d_p ℓ₃₂ℓ₃₂ᵀ` — a rank-one *update* (always stable, `d_p > 0`). The update
(method C1) writes its result one slot up-left, so deletion and compaction are one pass.
"""
function rowdel!(F::DenseLDLtFactor, j::Integer)
    _check_open(F)
    1 <= j <= F.n || throw(BoundsError(F, j))
    p = F.slot[j]
    p == 0 && return nothing                     # already an identity row
    k = F.k
    L, d, cap = F.L, F.d, F.cap
    m = k - p                                    # size of the trailing block
    z = F.z
    @inbounds for t in 1:m
        z[t] = L[p + t, p]                       # ℓ₃₂
    end
    # leading columns: drop row p of L₃₁ (shift rows p+1:k up by one)
    @inbounds for c in 1:(p - 1)
        off = (c - 1) * cap
        for r in p:(k - 1)
            L[off + r] = L[off + r + 1]
        end
    end
    _rank_one_shift!(L, d, z, p, m, d[p])
    # slots p+1:k move to p:k-1
    idx = F.index
    @inbounds for s in p:(k - 1)
        idx[s] = idx[s + 1]
        F.slot[idx[s]] = s
    end
    @inbounds F.slot[j] = 0
    F.k = k - 1
    F.deletions += 1
    F.deleted_trailing += m
    F.deleted_size += k
    return nothing
end

# Method C1 (Gill, Golub, Murray, Saunders 1974) for L₃₃D₃L₃₃ᵀ + α zzᵀ on the trailing block
# that starts at slot p+1 (size m), writing slot p+t of the input to slot p+t-1 of the output.
# Column p+t of the input has been read completely before column p+t-1 is overwritten.
function _rank_one_shift!(L::Matrix{Float64}, d::Vector{Float64}, z::Vector{Float64},
                          p::Int, m::Int, α::Float64)
    a = α
    @inbounds for t in 1:m
        js = p + t                               # source column/slot
        jd = js - 1                              # destination
        pt = z[t]
        dj = d[js]
        dn = dj + a * pt * pt
        β = pt * a / dn
        a = dj * a / dn
        d[jd] = dn
        if pt == 0.0
            # no change to column js: only shift it
            @simd for r in (t + 1):m
                L[p + r - 1, jd] = L[p + r, js]
            end
        else
            @simd for r in (t + 1):m
                lr = L[p + r, js]
                zr = z[r] - pt * lr
                z[r] = zr
                L[p + r - 1, jd] = lr + β * zr
            end
        end
    end
    return nothing
end

# ── updown: rank-one update / downdate ──────────────────────────────────────────────────

"""
    updown!(F, update, rows, vals, nnz) -> Bool

`M ← M + vvᵀ` (`update`) or `M ← M - vvᵀ`, `v = (rows, vals)[1:nnz]` on active indices.
Method C1 runs from the first slot where `v` is nonzero (earlier columns are unchanged).
Returns `false` as soon as a downdated pivot is `≤ 0`; the factor is then invalid and must be
rebuilt by the caller, exactly as with the CHOLMOD backend.
"""
function updown!(F::DenseLDLtFactor, update::Bool, rows::AbstractVector{<:Integer},
                 vals::AbstractVector{Float64}, nnz::Integer)
    _check_open(F)
    nnz == 0 && return true
    k = F.k
    z = F.z
    @inbounds for t in 1:k
        z[t] = 0.0
    end
    first = k + 1
    @inbounds for t in 1:nnz
        s = _active_slot(F, rows[t], "updown!")
        z[s] += vals[t]
        first = min(first, s)
    end
    L, d = F.L, F.d
    a = update ? 1.0 : -1.0
    @inbounds for j in first:k
        pt = z[j]
        pt == 0.0 && continue                    # column j and a are unchanged
        dj = d[j]
        dn = dj + a * pt * pt
        dn > 0.0 || (d[j] = dn; return false)    # downdate lost positive definiteness
        β = pt * a / dn
        a = dj * a / dn
        d[j] = dn
        @simd for r in (j + 1):k
            lr = L[r, j]
            zr = z[r] - pt * lr
            z[r] = zr
            L[r, j] = lr + β * zr
        end
    end
    return true
end

# ── solves ──────────────────────────────────────────────────────────────────────────────

# x ← (L D Lᵀ)⁻¹ x on the first k slots of w.
@inline function _solve_slots!(F::DenseLDLtFactor)
    k = F.k
    k == 0 && return nothing
    Lk = view(F.L, 1:k, 1:k)
    w = view(F.w, 1:k)
    BLAS.trsv!('L', 'N', 'U', Lk, w)
    @inbounds @simd for t in 1:k
        w[t] /= F.d[t]
    end
    BLAS.trsv!('L', 'T', 'U', Lk, w)
    return nothing
end

"""
    solve!(F) -> Vector{Float64}

Solve `M x = rhs_buffer(F)` over all `n` indices (identity rows copy the right-hand side).
`O(n + k²)`; the QPAS uses the pattern solve below.
"""
function solve!(F::DenseLDLtFactor)
    _check_open(F)
    rhs, sol, w = F.rhs, F.sol, F.w
    copyto!(sol, rhs)
    @inbounds for s in 1:F.k
        w[s] = rhs[F.index[s]]
    end
    _solve_slots!(F)
    @inbounds for s in 1:F.k
        sol[F.index[s]] = w[s]
    end
    F.reach = F.n
    return sol
end

"""
    solve!(F, pattern, npattern) -> Vector{Float64}

Solve with a right-hand side that is zero outside `pattern[1:npattern]`. The solution is
written on all active indices and on the pattern; other entries are not defined. `O(k²)`.
"""
function solve!(F::DenseLDLtFactor, pattern::AbstractVector{<:Integer}, npattern::Integer)
    _check_open(F)
    rhs, sol, w = F.rhs, F.sol, F.w
    k = F.k
    @inbounds for s in 1:k
        w[s] = 0.0
    end
    extra = 0
    @inbounds for t in 1:npattern
        j = pattern[t]
        s = F.slot[j]
        if s == 0
            sol[j] = rhs[j]                      # identity row
            extra += 1
        else
            w[s] = rhs[j]
        end
    end
    _solve_slots!(F)
    @inbounds for s in 1:k
        sol[F.index[s]] = w[s]
    end
    F.reach = k + extra
    return sol
end

# ── pivots, reset, statistics ───────────────────────────────────────────────────────────

pivot(F::DenseLDLtFactor, j::Integer) = (s = F.slot[j]; s == 0 ? 1.0 : F.d[s])

function min_scaled_pivot(F::DenseLDLtFactor, indices::AbstractVector{<:Integer},
                          scale::AbstractVector{Float64})
    _check_open(F)
    m = Inf
    @inbounds for j in indices
        s = F.slot[j]
        dj = s == 0 ? 1.0 : F.d[s]
        m = min(m, dj / max(scale[j], floatmin(Float64)))
    end
    return m
end

"Back to the identity. `O(k)`: the stored entries are overwritten before they are read again."
function reset!(F::DenseLDLtFactor)
    _check_open(F)
    @inbounds for s in 1:F.k
        F.slot[F.index[s]] = 0
    end
    F.k = 0
    return F
end

"""
    factor_stats(F, active) -> NamedTuple

Same fields as the CHOLMOD backend. `stored` is the strictly lower triangle of the active
`k × k` block (`k(k-1)/2`), `effective` its numerically nonzero entries, `allocated` the
reserved `cap²` entries, `memory` the bytes of all arrays of the factor (no hidden workspace).
"""
function factor_stats(F::DenseLDLtFactor, active::AbstractVector{<:Integer})
    _check_open(F)
    k = F.k
    is_active = falses(k)
    for j in active
        s = F.slot[j]
        s > 0 && (is_active[s] = true)
    end
    effective = aeffective = astored = 0
    @inbounds for c in 1:k
        nz = 0
        for r in (c + 1):k
            F.L[r, c] != 0.0 && (nz += 1)
        end
        effective += nz
        if is_active[c]
            astored += k - c
            aeffective += nz
        end
    end
    bytes = _bytes(F)
    F.peak_bytes = max(F.peak_bytes, bytes)
    return (allocated = F.cap * F.cap, stored = k * (k - 1) ÷ 2, effective = effective,
            active_stored = astored, active_effective = aeffective,
            memory = bytes, memory_peak = F.peak_bytes)
end

"""
    backend_stats(F) -> NamedTuple

- `capacity`, `peak_active`, `grows`: allocated slots, most slots ever in use, reallocations
- `deletions`: `rowdel!` calls on active indices
- `mean_trailing_fraction`: mean of `(k - p)/k` over deletions, i.e. the share of the factor
  that the rank-one part of a deletion touches (1 = always the oldest slot)
"""
backend_stats(F::DenseLDLtFactor) =
    (capacity = F.cap, peak_active = F.peak_k, grows = F.grows, deletions = F.deletions,
     mean_trailing_fraction = F.deleted_size == 0 ? 0.0 : F.deleted_trailing / F.deleted_size)

"""
    factor_matrices(F) -> (G::SparseMatrixCSC, d::Vector{Float64})

The factorization in index space: `M = G Diagonal(d) Gᵀ` with `G[index[r], index[c]] = L[r, c]`
(a row/column permutation of a unit lower-triangular matrix) and identity rows elsewhere.
Debugging and tests only.
"""
function factor_matrices(F::DenseLDLtFactor)
    _check_open(F)
    n, k = F.n, F.k
    Is, Js, Vs = Int[], Int[], Float64[]
    d = ones(n)
    for c in 1:k
        jc = F.index[c]
        d[jc] = F.d[c]
        for r in (c + 1):k
            v = F.L[r, c]
            v == 0.0 && continue
            push!(Is, F.index[r])
            push!(Js, jc)
            push!(Vs, v)
        end
    end
    append!(Is, 1:n)
    append!(Js, 1:n)
    append!(Vs, ones(n))
    return sparse(Is, Js, Vs, n, n), d
end

function Base.close(F::DenseLDLtFactor)
    F.closed && return nothing
    F.closed = true
    F.L = zeros(0, 0)
    F.d = Float64[]
    F.rhs = Float64[]
    F.sol = Float64[]
    F.w = Float64[]
    F.z = Float64[]
    F.k = 0
    F.cap = 0
    return nothing
end

@inline _check_open(F::DenseLDLtFactor) =
    F.closed && throw(ArgumentError("DenseLDLtFactor has been closed"))

Base.show(io::IO, F::DenseLDLtFactor) =
    print(io, "DenseLDLtFactor(", F.n, "×", F.n, ", ", F.k, " active",
          F.closed ? ", closed)" : ")")

end # module DenseLDLt
