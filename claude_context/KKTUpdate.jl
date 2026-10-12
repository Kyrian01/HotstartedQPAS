

# Dependencies
#include("KKTFactor.jl")


using TimerOutputs
const to = TimerOutput()






# ============================================================
#                   KKT UPDATE FUNCTIONS
# ============================================================

function reduce_workingset!(
    kkt::KKTFactor,
    A::SparseMatrixCSC{Float64, Int64},
    r::Int64
)

    @assert kkt.column_to_set[r] == 2 "Column $r is not in working set"


    # Construct column u = A_r - A_rep_r
    rep_r = kkt.S[kkt.column_to_block[r]]
    A_r   = A[:, r]
    A_rep = A[:, rep_r]
    u = A_r - A_rep


    # Build new reduced Hessian column (AZᵀ u accumulation)
    @timeit to "AZt*u (reduce)" begin
        @inbounds for k in eachindex(u.nzind)
            sv = kkt.AZt_cols[u.nzind[k]]
            val = u.nzval[k]

            @inbounds for m in eachindex(sv.nzind)
                kkt.work_nc[sv.nzind[m]] += val * sv.nzval[m]
            end
        end
    end
    rhs = Vector(kkt.work_nc)
    kkt.work_nc[r] = dot(u, u)
    new_col = sparse(kkt.work_nc)   


    # Reset workspace (only touched entries)
    @inbounds for k in eachindex(u.nzind)
        sv = kkt.AZt_cols[u.nzind[k]]

        @inbounds for m in eachindex(sv.nzind)
            kkt.work_nc[sv.nzind[m]] = 0.0
        end
    end
    kkt.work_nc[r] = 0.0


    # Update CHOLMOD factorization (rank modification)
    common = SparseArrays.CHOLMOD.getcommon()

    @timeit to "rowadd (reduce)" begin
        cholmod_rowadd!(
            kkt.Factor,
            r,
            SparseArrays.CHOLMOD.Sparse(new_col),
            common
        )
    end



    # Diagnostic: check factor diagonal
    @timeit to "diag(LD) (reduce)" begin

        @timeit to "check diag" begin

            cholmod_ldlt_diag!(kkt.work_diag, kkt.Factor)

            diag_ok = true
            @inbounds for d in kkt.work_diag
                if !isfinite(d) || abs(d) < 1e-12
                    diag_ok = false
                    break
                end
            end
        end

        @timeit to "similarity check" begin
            if !diag_ok
                println("FACTOR ISSUES: Reducing working set failed due to linear dependency")

                # Undo factor update
                cholmod_rowdel!(kkt.Factor, r, common)

                # Determine column (in N) in AZ that is the most similar to the new column A*Z_r
                y = sparse(kkt.Factor \ rhs)

                n = length(kkt.column_to_set)   # current amount of columns used

                k = argmax(i ->
                    (kkt.column_to_set[i] == 3) ? abs(y[i]) : -Inf,
                    1:n
                )

                return k
            end
        end
    end


    # Debug update (optional full Hessian mirror)
    if kkt.debug && kkt.debug_H !== nothing
        @inbounds for (i, v) in zip(new_col.nzind, new_col.nzval)
            kkt.debug_H[i, r] = v
            kkt.debug_H[r, i] = v
        end
    end


    # Update nullspace basis Z and AZ
    @timeit to "Z/AZ update (reduce)" begin

        kkt.Z[r, r] = 1.0
        kkt.Z[rep_r, r] = -1.0

        kkt.AZ[:, r] .= u

        # update AZt column structure
        @inbounds for k in eachindex(u.nzind)
            i = u.nzind[k]
            val = u.nzval[k]

            sv = kkt.AZt_cols[i]
            pos = searchsortedfirst(sv.nzind, r)

            insert!(sv.nzind, pos, r)
            insert!(sv.nzval, pos, val)
        end
    end



    # CG ordering update
    idx = findfirst(==(r), kkt.working_set)
    idx === nothing && error("Index not found in working set")

    deleteat!(kkt.working_set, idx)

    push!(kkt.N, r)
    kkt.column_to_set[r] = 3


    return 0
end



function expand_workingset!(
    kkt::KKTFactor,
    q::Int64
)   

    col_max = size(kkt.AZ, 2)

    # Identify removal + affected columns
    removal_col, affected_cols =
        find_affected(q, kkt.N, kkt.column_to_block, kkt.column_to_set)



    
    # Precompute H[:, removal_col] from AZ structure
    rc_AZ = kkt.AZ[:, removal_col]

    Hr_col = zeros(Float64, col_max)

    @inbounds for m in eachindex(rc_AZ.nzind)
        sv  = kkt.AZt_cols[rc_AZ.nzind[m]]
        val = rc_AZ.nzval[m]

        @inbounds for l in eachindex(sv.nzind)
            Hr_col[sv.nzind[l]] += val * sv.nzval[l]
        end
    end

    Hrr = Hr_col[removal_col]



    # Empty the removal_col index from CHOLMOD factor
    common = SparseArrays.CHOLMOD.getcommon()

    @timeit to "rowdel (expand)" begin
        cholmod_rowdel!(kkt.Factor, removal_col, common)
    end



    # Update affected columns in factorization
    for k in affected_cols

        @timeit to "rowdel (expand)" begin
            cholmod_rowdel!(kkt.Factor, k, common)
        end

        # Compute updated H[:, k]
        new_AZ_k = kkt.AZ[:, k] - kkt.AZ[:, removal_col]

        new_Hk = zeros(Float64, col_max)

        @inbounds for m in eachindex(new_AZ_k.nzind)
            sv  = kkt.AZt_cols[new_AZ_k.nzind[m]]
            val = new_AZ_k.nzval[m]

            @inbounds for l in eachindex(sv.nzind)
                new_Hk[sv.nzind[l]] += val * sv.nzval[l]
            end
        end


        # Correction for modified rows
        corr = -Hr_col[k] + Hrr

        new_Hk[removal_col] += corr

        @inbounds for j in affected_cols
            new_Hk[j] += corr
        end


        # Insert updated column into CHOLMOD
        @timeit to "rowadd (expand)" begin
            cholmod_rowadd!(
                kkt.Factor,
                k,
                SparseArrays.CHOLMOD.Sparse(sparse(new_Hk)),
                common
            )
        end
    end


    # Update CG ordering
    update_CGOrdering!(kkt, q, removal_col)


    # Debug: Hessian update
    if kkt.debug && kkt.debug_H !== nothing

        Hr_dbg = kkt.debug_H[:, removal_col]
        Hrr_dbg = kkt.debug_H[removal_col, removal_col]

        for k in affected_cols
            kkt.debug_H[:, k] .-= Hr_dbg
            kkt.debug_H[vcat(removal_col, affected_cols), k] .+= -Hr_dbg[k] + Hrr_dbg
        end

        kkt.debug_H[affected_cols, :] = kkt.debug_H[:, affected_cols]'
        kkt.debug_H[:, removal_col] .= 0.0
        kkt.debug_H[removal_col, :] .= 0.0
        kkt.debug_H[removal_col, removal_col] = 1.0
    end



    # Update AZ and AZt structure
    @timeit to "AZ update (expand)" begin

        @inbounds for j in affected_cols

            kkt.AZ[:, j] .-= kkt.AZ[:, removal_col]

            @inbounds for m in eachindex(rc_AZ.nzind)

                i = rc_AZ.nzind[m]
                val = rc_AZ.nzval[m]

                sv = kkt.AZt_cols[i]
                pos = searchsortedfirst(sv.nzind, j)

                if pos <= length(sv.nzind) && sv.nzind[pos] == j
                    sv.nzval[pos] -= val
                else
                    insert!(sv.nzind, pos, j)
                    insert!(sv.nzval, pos, -val)
                end
            end
        end

        kkt.AZ[:, removal_col] .= 0.0

        # remove column from AZt structure
        @inbounds for m in eachindex(rc_AZ.nzind)

            sv = kkt.AZt_cols[rc_AZ.nzind[m]]
            pos = searchsortedfirst(sv.nzind, removal_col)

            if pos <= length(sv.nzind) && sv.nzind[pos] == removal_col
                deleteat!(sv.nzind, pos)
                deleteat!(sv.nzval, pos)
            end
        end
    end



    # Update Z basis
    @timeit to "Z update (expand)" begin
        @inbounds for j in affected_cols
            kkt.Z[:, j] .-= kkt.Z[:, removal_col]
        end
        kkt.Z[:, removal_col] .= 0.0
    end



    # 8. Diagnostic check for linear dependency
    @timeit to "diag(LD) (expand)" begin

        cholmod_ldlt_diag!(kkt.work_diag, kkt.Factor)

        ok = true
        @inbounds for d in kkt.work_diag
            if !isfinite(d) || abs(d) < 1e-12
                ok = false
                break
            end
        end

        if !ok
            println("FACTOR ISSUES: Expanding working set failed due to linear dependency")
        end
    end

end





# ============================================================
#                   CG ORDERING MANAGEMENT 
# ============================================================

function find_affected(
    q::Int64,
    N::Vector{Int64},
    column_to_block::Vector{Int64},
    column_to_set::Vector{UInt8}
)

    # Case 1: q is in N (no structural change needed)
    if column_to_set[q] == 3
        return q, Int64[]
    end

    
    # Case 2: q is in S
    if column_to_set[q] == 1

        r = column_to_block[q]

        removal = 0
        affected = Vector{Int64}(undef, length(N))
        count = 0

        @inbounds for j in N
            if column_to_block[j] == r

                if removal == 0
                    removal = j
                else
                    count += 1
                    affected[count] = j
                end
            end
        end

        removal == 0 && error("No replacement found for block $r in N")

        return removal, view(affected, 1:count)
    end


    # Invalid state
    error("q must be in S or N")
end


function update_CGOrdering!(
    kkt::KKTFactor,
    q::Int64,
    removal_col::Int64
)

    set_q = kkt.column_to_set[q]

    # Case 1: q in N  (move to working set)
    if set_q == 3

        idx = findfirst(==(q), kkt.N)
        idx === nothing && error("Index not found in N")

        deleteat!(kkt.N, idx)

        push!(kkt.working_set, q)
        kkt.column_to_set[q] = 2

        return
    end


    # Case 2: q in S  (swap representative)
    if set_q == 1

        r = kkt.column_to_block[q]

        push!(kkt.working_set, q)
        kkt.column_to_set[q] = 2

        idx = findfirst(==(removal_col), kkt.N)
        idx === nothing && error("removal_col not found in N")

        deleteat!(kkt.N, idx)

        kkt.S[r] = removal_col
        kkt.column_to_set[removal_col] = 1

        return
    end

    # Invalid state
    error("q must be in S or N")
end






# ============================================================
#                   CHOLMOD FACTOR UPDATES 
# ============================================================

# CHOLMOD row-delete wrappers
function cholmod_rowdel!(
    factor::SparseArrays.CHOLMOD.Factor,
    idx::Int,
    common::Base.RefValue{SparseArrays.LibSuiteSparse.cholmod_common_struct}
)
    ret = ccall(
        (:cholmod_l_rowdel, cholmod_lib),
        Cint,
        (Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        idx - 1,
        C_NULL,
        factor.ptr,
        common
    )

    ret == 1 || error("cholmod_l_rowdel failed (idx=$idx, code=$ret)")
end



# CHOLMOD row-add wrappers
function cholmod_rowadd!(
    factor::SparseArrays.CHOLMOD.Factor,
    idx::Int,
    row::SparseArrays.CHOLMOD.Sparse,
    common::Base.RefValue{SparseArrays.LibSuiteSparse.cholmod_common_struct}
)
    ret = ccall(
        (:cholmod_l_rowadd, cholmod_lib),
        Cint,
        (Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        idx - 1,
        row.ptr,
        factor.ptr,
        common
    )

    ret == 1 || error("cholmod_l_rowadd failed (idx=$idx, code=$ret)")
end





# ============================================================
#                   CHOLMOD FACTOR CHECK 
# ============================================================

# Fill D[j] = diag(LDL')[j] for all j in O(n).
# D must be pre-allocated to length >= size(F,1).
function cholmod_ldlt_diag!(D::Vector{Float64}, F::SparseArrays.CHOLMOD.Factor{Float64})
    f      = unsafe_load(F.ptr)
    n      = Int(f.n)
    nzmax  = Int(f.nzmax)
    colptr = unsafe_wrap(Array, Ptr{Int64}(f.p),    n + 1; own = false)
    xvals  = unsafe_wrap(Array, Ptr{Float64}(f.x), nzmax; own = false)
    @inbounds for j in 1:n
        D[j] = xvals[colptr[j] + 1]
    end
    return D
end



