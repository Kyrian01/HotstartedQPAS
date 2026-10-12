

# Dependencies
#include("KKTFactor.jl")


using TimerOutputs
const to = TimerOutput()





# ============================================================
#                       KKT SYSTEM SOLVER 
# ============================================================


function solve_KKTsystem(
    kkt::KKTFactor,
    grad::Vector{Float64},
    A::SparseMatrixCSC{Float64, Int64},
    delta::Float64
)

    # Compute - Z' * grad 
    @timeit to "- Z'*grad" begin
        fill!(kkt.work_rhs, 0.0)

        @inbounds for j in kkt.N
            rep_j = kkt.S[kkt.column_to_block[j]]
            kkt.work_rhs[j] = grad[rep_j] - grad[j]
        end
    end


    # Solve reduced system
    @timeit to "Factor\\rhs" begin
        cholmod_solve2!(kkt)
        lmul!(delta, kkt.work_y)
    end


    # Compute p = Z * y
    @timeit to "Z*y" begin
        fill!(kkt.work_p, 0.0)

        @inbounds for j in kkt.N
            rep = kkt.S[kkt.column_to_block[j]]
            yj = kkt.work_y[j]

            kkt.work_p[j] += yj
            kkt.work_p[rep] -= yj
        end
    end
    p = kkt.work_p


    # Compute h = (1/δ) A' A p + grad
    if norm(p) < 1e-8
        h = grad
    else
        @timeit to "A*p" mul!(kkt.work_Ap, A, p)

        @timeit to "A'*Ap" begin
            mul!(kkt.work_AtAp, A', kkt.work_Ap)
            lmul!(1 / delta, kkt.work_AtAp)
        end

        copyto!(kkt.work_h, grad)
        axpy!(1.0, kkt.work_AtAp, kkt.work_h)

        h = kkt.work_h
    end


    # Convexity constraint Lagrange multipliers mu
    @inbounds for i in eachindex(kkt.S)
        kkt.work_mu[i] = h[kkt.S[i]]
    end
    mu = view(kkt.work_mu, 1:length(kkt.S))


    # Working set Lagrange multipliers lambda
    ws = kkt.working_set
    n_ws = length(ws)

    @inbounds for i in 1:n_ws
        j = ws[i]
        kkt.work_lam[i] = h[j] - kkt.work_mu[kkt.column_to_block[j]]
    end
    lam = view(kkt.work_lam, 1:n_ws)


    # Return results
    return (
        p = p,
        mu = mu,
        lam = lam,
        AtAp = kkt.work_AtAp
    )
end







# ============================================================
#                   CHOLMOD SOLVER INTERFACE 
# ============================================================




# CHOLMOD Dense wrapper (raw C layout)
struct CHOLMODDense
    nrow::Csize_t
    ncol::Csize_t
    nzmax::Csize_t
    d::Csize_t
    x::Ptr{Cvoid}      # real data
    z::Ptr{Cvoid}      # imaginary (unused)
    xtype::Cint        # 1 = real
    dtype::Cint        # 0 = double
end




# Solve using CHOLMOD solve2 (in-place, allocation-free)
function cholmod_solve2!(kkt::KKTFactor)

    common = SparseArrays.CHOLMOD.getcommon()
    n = length(kkt.work_rhs)

    # Copy RHS into CHOLMOD dense buffer (no allocation)
    B_hdr = unsafe_load(Ptr{CHOLMODDense}(kkt.solve2_B))
    unsafe_copyto!(Ptr{Float64}(B_hdr.x), pointer(kkt.work_rhs), n)


    # Solve system using CHOLMOD solve2
    ret = ccall(
        (:cholmod_l_solve2, cholmod_lib),
        Cint,
        (Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
         Ptr{Ptr{Cvoid}}, Ptr{Cvoid},
         Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}},
         Ptr{Cvoid}),
        0,                      # sys = A x = b
        kkt.Factor.ptr,
        kkt.solve2_B,
        C_NULL,
        kkt.solve2_X,
        C_NULL,
        kkt.solve2_Y,
        kkt.solve2_E,
        common
    )

    ret == 1 || error("cholmod_l_solve2 failed (code=$ret)")

    # Copy solution back to Julia buffer
    X_hdr = unsafe_load(Ptr{CHOLMODDense}(kkt.solve2_X[]))
    unsafe_copyto!(pointer(kkt.work_y), Ptr{Float64}(X_hdr.x), n)

    return nothing
end