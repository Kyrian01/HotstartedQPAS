

using LinearAlgebra

using SparseArrays
using SparseArrays.CHOLMOD
using SuiteSparse
using SuiteSparse_jll
cholmod_lib = SuiteSparse_jll.libcholmod_path






# ============================================================
#                       KKT FACTOR STATE 
# ============================================================


mutable struct KKTFactor


    # CG-ordering
    S::Vector{Int64}                 # Represenant list S
    working_set::Vector{Int64}       # Working set W
    N::Vector{Int64}                 # Nullspace index set N
    column_to_block::Vector{Int64}   # Column → representative block
    column_to_set::Vector{UInt8}     # 1 = S, 2 = W, 3 = N


    # Nullspace representation / Reduced Hessian
    Z::SparseMatrixCSC{Float64, Int64}
    AZ::SparseMatrixCSC{Float64, Int64}
    AZt_cols::Vector{SparseVector{Float64, Int64}}
    Factor::Union{Nothing, SparseArrays.CHOLMOD.Factor}


    # Scratch buffers (preallocated to avoid allocations in hot paths)
    work_rhs::Vector{Float64}      # - Z' * grad
    work_y::Vector{Float64}        # Factor \ rhs
    work_p::Vector{Float64}        # Z * y

    work_Ap::Vector{Float64}       # A * p
    work_AtAp::Vector{Float64}     # (1/δ) * A' * A * p

    work_diag::Vector{Float64}     # LDLᵀ diagonal
    work_nc::Vector{Float64}       # AZᵀ * u accumulation
    work_h::Vector{Float64}        # h = AtAp + grad

    work_mu::Vector{Float64}       # Convexity constraint Lagrange multipliers
    work_lam::Vector{Float64}      # Working set Lagrange multipliers


    # Persistent CHOLMOD solve2 handles
    solve2_B::Ptr{Cvoid}           # RHS dense buffer
    solve2_X::Ref{Ptr{Cvoid}}      # Solution
    solve2_Y::Ref{Ptr{Cvoid}}      # Workspace
    solve2_E::Ref{Ptr{Cvoid}}      # Workspace


    # Debug support
    debug::Bool
    debug_H::Union{Nothing, SparseMatrixCSC{Float64, Int64}}



    # Empty constructor
    function KKTFactor()
        new(
            Int64[], Int64[], Int64[],
            Int64[], UInt8[],

            spzeros(0, 0),
            spzeros(0, 0),
            SparseVector{Float64, Int64}[],
            nothing,

            Float64[], Float64[], Float64[],
            Float64[], Float64[],
            Float64[], Float64[], Float64[],
            Float64[], Float64[],

            C_NULL,
            Ref{Ptr{Cvoid}}(C_NULL),
            Ref{Ptr{Cvoid}}(C_NULL),
            Ref{Ptr{Cvoid}}(C_NULL),

            false,
            nothing
        )
    end



    # Main constructor
    function KKTFactor(
        convexity_matrix::SparseMatrixCSC{Float64, Int64},
        A::SparseMatrixCSC{Float64, Int64},
        working_set::Vector{Int64},
        n::Int64,
        col_max::Int64;
        debug::Bool = false
    )

        p = size(convexity_matrix, 1)
        m = size(A, 1)


        # Determine CG ordering: [S, W, N]
        S, N, column_to_block, column_to_set = CGOrdering(convexity_matrix, working_set, p, n)


        # Build nullspace basis
        Z, AZ, AZt_cols = build_nullspace(A, N, S, column_to_block, col_max)


        # Factorize reduced Hessian
        H = sparse(AZ' * AZ)
        ldltFactor = factorize_reducedHessian(H, N, col_max)
        debug_H = debug ? H : nothing


        # Allocate persistent CHOLMOD dense RHS buffer
        common = SparseArrays.CHOLMOD.getcommon()

        s2_B = ccall(
            (:cholmod_l_allocate_dense, cholmod_lib),
            Ptr{Cvoid},
            (Csize_t, Csize_t, Csize_t, Cint, Ptr{Cvoid}),
            col_max, 1, col_max, Cint(1), common
        )

        s2_B == C_NULL &&
            error("cholmod_l_allocate_dense failed")


        # Construct struct
        return new(

            # CG ordering
            S, working_set, N,
            column_to_block, column_to_set,

            # Reduced Hessian
            Z, AZ, AZt_cols,
            ldltFactor,

            # Scratch buffers
            zeros(col_max), zeros(col_max), zeros(col_max),
            zeros(m), zeros(col_max),
            zeros(col_max), zeros(col_max), zeros(col_max),
            zeros(p), zeros(col_max),

            # CHOLMOD handles
            s2_B, Ref{Ptr{Cvoid}}(C_NULL), Ref{Ptr{Cvoid}}(C_NULL), Ref{Ptr{Cvoid}}(C_NULL),

            # Debug
            debug, debug_H
        )
    end

end







# ============================================================
#           FACTOR CONSTRUCTION / INITIALIZATION
# ============================================================


function CGOrdering(
    convexity_matrix::SparseMatrixCSC{Float64, Int64}, 
    working_set::Vector{Int64}, 
    p::Int64, n::Int64
)

    # Representative mapping (row/block index for each column)
    column_to_block = convexity_matrix[:, 1:n].rowval[convexity_matrix[:, 1:n].colptr[1:end-1]]

    # Preallocate outputs 
    S = zeros(Int64, p)
    N = zeros(Int64, n - p - length(working_set))
    column_to_set = zeros(UInt8, n)                 # 1=S, 2=W, 3=N

    # Mark working set 
    @inbounds for idx in 1:length(working_set)
        j = working_set[idx]
        column_to_set[j] = 2        
    end

    # Build S and mark column_to_set
    @inbounds for j in 1:n
        r = column_to_block[j]
        if S[r] == 0 && column_to_set[j] != 2
            S[r] = j
            column_to_set[j] = 1         
        end
    end

    # Check feasibility
    @inbounds for r in 1:p
        S[r] == 0 && error("No available representative for row $r")
    end

    # Build N and mark column_to_set
    idx = 1
    @inbounds for j in 1:n
        if column_to_set[j] == 0
            N[idx] = j
            column_to_set[j] = 3
            idx += 1
        end
    end

    return S, N, column_to_block, column_to_set

end



function build_nullspace(
    A::SparseMatrixCSC{Float64, Int64}, 
    N::Vector{Int64}, S::Vector{Int64}, 
    column_to_block::Vector{Int64}, 
    col_max::Int64
)

    m = size(A, 1)
    Z  = spzeros(Float64, col_max, col_max)
    AZ = spzeros(Float64, m, col_max)

    @inbounds for j in N

		# representative column index
		rep_j = S[column_to_block[j]]

		# Nullvector = e_j - e_rep_j
		Z[j, j] = 1.0
		Z[rep_j, j] = -1.0

        # Fill column j of AZ
        AZ[:, j] .= A[:, j] .- A[:, rep_j]

	end

    # Build AZt_cols: AZt_cols[i] = column i of AZt = row i of AZ (sparse, length col_max)
    # Memory: O(nnz(AZ)) total, scales with nnz not with col_max × m
    AZt_full = sparse(AZ')
    AZt_cols = [AZt_full[:, i] for i in 1:m]

    return Z, AZ, AZt_cols

end



function factorize_reducedHessian(
    H::SparseMatrixCSC{Float64, Int64}, 
    N::Vector{Int64}, 
    max_col::Int64
)

    # Set diagonal 1 for columns not in set N (for positive symmetry purposes needed in ldlt)
    is_in_N = falses(max_col)
    @inbounds for j in N
        is_in_N[j] = true
    end
    @inbounds for j in 1:max_col
        if !is_in_N[j]
            H[j, j] = 1.0
        end
    end

    # Sparse LDLT factorization using CHOLMOD
    ldltFactor = SparseArrays.ldlt(SparseArrays.CHOLMOD.Sparse(H); perm = collect(1:max_col))

    return ldltFactor

end






# ============================================================
#                   DIAGNOSTICS / VALIDATION
# ============================================================


function check_factorization!(kkt::KKTFactor)
    LD = sparse(kkt.Factor.LD)
    D = diag(LD)
    L = tril(LD)
    L = L - Diagonal(D) + I  # remove D from L, restore unit diagonal
    H_recon = L * Diagonal(D) * L'

    # Use debug_H if available; otherwise recompute H = AZ'*AZ from scratch
    if kkt.debug_H !== nothing
        H_ref = kkt.debug_H
    else
        col_max = size(kkt.AZ, 2)
        H_ref = sparse(kkt.AZ' * kkt.AZ)
        is_in_N = falses(col_max)
        @inbounds for j in kkt.N; is_in_N[j] = true; end
        @inbounds for j in 1:col_max
            !is_in_N[j] && (H_ref[j, j] = 1.0)
        end
    end

    return norm(H_recon - H_ref)
end