# Column generation ordering [S, W, N] (thesis Def. 4.4.4, Extended ordering, and Thm. 4.4.5,
# Reordering theorem; the slacks' split W_s / N_s is kept in `SurplusSet`).
#
#   S : list of representatives, one column per block (S[b] = rep[b])
#   W : working set, columns fixed at zero
#   N : nullspace set, the remaining free columns; one nullspace vector zⱼ = eⱼ - e_rep(j) each
#
# Every column registered with the QPAS is in exactly one of the three sets; `set_of[j]`
# records which, so membership tests are O(1).

const NOT_REGISTERED = 0x00
const IN_S = 0x01
const IN_W = 0x02
const IN_N = 0x03

"""
    ColumnOrdering(n_blocks, capacity)

The partition `[S, W, N]` of the registered columns with O(1) updates. `rep[b]` is the
representative column of block `b` (the list `S`).
"""
struct ColumnOrdering
    rep::Vector{Int}
    W::IndexSet
    N::IndexSet
    set_of::Vector{UInt8}
end

ColumnOrdering(n_blocks::Integer, capacity::Integer) =
    ColumnOrdering(zeros(Int, n_blocks), IndexSet(capacity), IndexSet(capacity),
                   fill(NOT_REGISTERED, capacity))

"Representative `rep(j) = S[ψ(j)]` of column `j` (thesis Def. 4.4.3, Representatives)."
@inline representative(ord::ColumnOrdering, block::Vector{Int}, j::Integer) =
    @inbounds ord.rep[block[j]]

@inline in_S(ord::ColumnOrdering, j::Integer) = @inbounds ord.set_of[j] == IN_S
@inline in_W(ord::ColumnOrdering, j::Integer) = @inbounds ord.set_of[j] == IN_W
@inline in_N(ord::ColumnOrdering, j::Integer) = @inbounds ord.set_of[j] == IN_N

"Size of the column nullspace set, `|N| = n - p - |W|` (thesis `|N_x|`, the dimension of `H_eff`)."
nullspace_dimension(ord::ColumnOrdering) = length(ord.N)

function move_to_W!(ord::ColumnOrdering, j::Integer)
    s = ord.set_of[j]
    s == IN_N && delete!(ord.N, j)
    push!(ord.W, j)
    ord.set_of[j] = IN_W
    return ord
end

function move_to_N!(ord::ColumnOrdering, j::Integer)
    s = ord.set_of[j]
    s == IN_W && delete!(ord.W, j)
    push!(ord.N, j)
    ord.set_of[j] = IN_N
    return ord
end

function make_representative!(ord::ColumnOrdering, block::Vector{Int}, j::Integer)
    s = ord.set_of[j]
    s == IN_N && delete!(ord.N, j)
    s == IN_W && delete!(ord.W, j)
    ord.rep[block[j]] = j
    ord.set_of[j] = IN_S
    return ord
end

# ── surpluses of inequality rows (thesis §2.4, Rem. 2.4.4; Def. 4.4.4) ─────────────────────
#
# Every inequality coupling row i gets one slack variable 0 ≤ sσ ≤ Uσ so that it becomes an
# equality
#
#     aᵢᵀx + κσ sσ = bᵢ,     κσ = -1 for a "≥" row (surplus),  κσ = +1 for a "≤" row (slack).
#
# A "≤" row is the "≥" row -aᵢᵀx ≥ -bᵢ; flipping the sign of the slack instead of the row keeps
# the column pool untouched, and since κσ² = 1 all factor updates are identical for both kinds.
# One-sided rows have Uσ = ∞. A range row ℓᵢ ≤ aᵢᵀx ≤ uᵢ starts as the "≥" row (κσ = -1,
# bᵢ = ℓᵢ) with Uσ = uᵢ - ℓᵢ. When its free slack reaches Uσ, the substitution s̄σ = Uσ - sσ
# turns it into the "≤" row aᵢᵀx + s̄σ = uᵢ (`flip_bound!`): κσ and bᵢ change, s̄σ = 0 enters the
# working set, and nothing else does. The slack is again in [0, Uσ], so it can flip back.
# Slacks belong to no block, so they extend the ordering [S, W, N] by two sets:
#
#   W_s : slacks fixed at sσ = 0 (in the working set)        → their row acts like an equality
#   N_s : free slacks (nullspace indices z = e_σ)             → their row drops out of H_eff

"""
    SurplusSet(rows, signs, n_rows; upper = fill(Inf, length(rows)))

Slack variables of the inequality coupling rows and their working-set status. `row[σ]` is the
coupling row of slack `σ`, `sign[σ] = κσ ∈ {-1, +1}`, `upper[σ] = Uσ ∈ (0, ∞]` its upper bound
(finite only for range rows), `free` is the set `N_s`, and `row_free[i]` marks the coupling rows
whose slack is free (the rows masked out of `H_eff`). Initially every slack is in the working
set (`sσ = 0`, `N_s = ∅`). `sign` changes when a range-row slack flips its bound.
"""
struct SurplusSet
    row::Vector{Int}
    sign::Vector{Float64}
    upper::Vector{Float64}
    free::IndexSet
    row_free::Vector{Bool}
end

function SurplusSet(rows::Vector{Int}, signs::Vector{Float64}, n_rows::Integer;
                    upper::Vector{Float64} = fill(Inf, length(rows)))
    length(rows) == length(signs) == length(upper) || throw(DimensionMismatch("rows, signs, upper"))
    all(s -> s == 1.0 || s == -1.0, signs) || throw(ArgumentError("slack signs must be ±1"))
    all(>(0.0), upper) || throw(ArgumentError("slack upper bounds must be positive"))
    allunique(rows) || throw(ArgumentError("at most one slack per coupling row"))
    return SurplusSet(rows, copy(signs), copy(upper), IndexSet(length(rows)), fill(false, n_rows))
end

"Number of slack variables `m_I`."
n_surplus(sp::SurplusSet) = length(sp.row)
"`true` if slack `σ` is free (`σ ∈ N_s`)."
@inline is_free(sp::SurplusSet, σ::Integer) = @inbounds sp.row_free[sp.row[σ]]
"`true` if some row is currently masked out of the effective reduced Hessian."
@inline any_free(sp::SurplusSet) = !isempty(sp.free)

function free_surplus!(sp::SurplusSet, σ::Integer)
    push!(sp.free, σ)
    sp.row_free[sp.row[σ]] = true
    return sp
end

function fix_surplus!(sp::SurplusSet, σ::Integer)
    delete!(sp.free, σ)
    sp.row_free[sp.row[σ]] = false
    return sp
end

"""
    check_ordering(ord, pool, n)

Verify the invariants of the ordering for the first `n` columns (used by tests / debug mode).
"""
function check_ordering(ord::ColumnOrdering, pool::ColumnPool, n::Integer)
    for b in eachindex(ord.rep)
        r = ord.rep[b]
        (r != 0 && ord.set_of[r] == IN_S && pool.block[r] == b) ||
            error("ordering: invalid representative for block $b")
    end
    nS = count(==(IN_S), view(ord.set_of, 1:n))
    nW = count(==(IN_W), view(ord.set_of, 1:n))
    nN = count(==(IN_N), view(ord.set_of, 1:n))
    nS == length(ord.rep) || error("ordering: |S| = $nS ≠ p")
    nW == length(ord.W) || error("ordering: W list out of sync")
    nN == length(ord.N) || error("ordering: N list out of sync")
    nS + nW + nN == n || error("ordering: unregistered columns among the first $n")
    return true
end
