

using SparseArrays
using LinearAlgebra
using Random
using Printf

import HiGHS

const libhighs = HiGHS.libhighs

include("../pricing_solvers/shortest_path.jl")





# ─── LP column generation ─────────────────────────────────────────────────────

# Add one path column to a persistent HiGHS LP instance.
# ax_col: T-vector of task coverage; pilot_idx: 1-based pilot; T_: num tasks.
function lp_add_col!(h, ax_col::Vector{Float64}, pilot_idx::Int,
                      cost_val::Float64, T_::Int)
    nz   = findall(!iszero, ax_col)
    rids = Vector{Cint}(vcat(nz .- 1, T_ + pilot_idx - 1))  # 0-indexed rows
    vals = Vector{Cdouble}(vcat(ax_col[nz], 1.0))
    ccall((:Highs_addCols, libhighs), Cint,
          (Ptr{Cvoid}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
          h, Cint(1),
          [Cdouble(cost_val)], [Cdouble(0.0)], [Cdouble(Inf)],
          Cint(length(rids)), Cint[0], rids, vals)
end

"""
    cg_lp(P, N, T, cp, c_orig, F_graph, b_graph, task_matrix, n_lp, fexact)
        -> (obj, times)

LP column generation using a single persistent HiGHS instance (low-level ccall).
Master: min 10000*(s⁺+s⁻) + costs'*y
  s.t.  s⁺ - s⁻ + AX*y = ones(T),   W*y = ones(P),   s⁺,s⁻,y ≥ 0
"""
function cg_lp(P::Int, N::Int, T::Int,
               cp::Vector{Float64}, c_orig::Vector{Float64},
               F_graph::SparseMatrixCSC{Float64,Int},
               b_graph::Vector{Float64},
               task_matrix::Matrix{Float64},
               n_lp::Int, fexact::Float64)

    A1 = task_matrix[:, 1:N]   # task-edge block (same for all pilots)

    # Warm-start: one cheapest path + one reduced-cost path per pilot
    paths = zeros(N, P)
    W     = Matrix{Float64}(I, P, P)
    costs = zeros(P)
    for pilot in 1:P
        paths[:, pilot], costs[pilot] =
            sp_dag(cp[(pilot-1)*N .+ (1:N)], zeros(N), F_graph, b_graph)
    end
    for pilot in 1:P
        path, cost = sp_dag(cp[(pilot-1)*N .+ (1:N)], c_orig, F_graph, b_graph)
        paths = hcat(paths, path)
        costs = vcat(costs, cost)
        wnew = zeros(P); wnew[pilot] = 1.0
        W = hcat(W, wnew)
    end

    # Build persistent HiGHS instance
    h = ccall((:Highs_create, libhighs), Ptr{Cvoid}, ())
    ccall((:Highs_setBoolOptionValue, libhighs), Cint,
          (Ptr{Cvoid}, Cstring, Cint), h, "output_flag", 0)

    # T coverage rows + P pilot rows (equality: lb = ub = 1)
    for _ in 1:(T + P)
        ccall((:Highs_addRow, libhighs), Cint,
              (Ptr{Cvoid}, Cdouble, Cdouble, Cint, Ptr{Cint}, Ptr{Cdouble}),
              h, 1.0, 1.0, Cint(0), C_NULL, C_NULL)
    end

    # s⁺: T columns, cost=10000, +1 in coverage row i
    ccall((:Highs_addCols, libhighs), Cint,
          (Ptr{Cvoid}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
          h, Cint(T),
          fill(Cdouble(10_000.0), T), fill(Cdouble(0.0), T), fill(Cdouble(Inf), T),
          Cint(T), Vector{Cint}(0:T-1), Vector{Cint}(0:T-1), fill(Cdouble(1.0), T))

    # s⁻: T columns, cost=10000, -1 in coverage row i
    ccall((:Highs_addCols, libhighs), Cint,
          (Ptr{Cvoid}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
          h, Cint(T),
          fill(Cdouble(10_000.0), T), fill(Cdouble(0.0), T), fill(Cdouble(Inf), T),
          Cint(T), Vector{Cint}(0:T-1), Vector{Cint}(0:T-1), fill(Cdouble(-1.0), T))

    # Initial path columns
    nc = length(costs)
    AX_init = A1 * paths
    for k in 1:nc
        pilot_k = findfirst(!iszero, W[:, k])
        lp_add_col!(h, Vector{Float64}(AX_init[:, k]), pilot_k, costs[k], T)
    end

    row_dual_buf = Vector{Cdouble}(undef, T + P)
    row_val_buf  = Vector{Cdouble}(undef, T + P)
    obj   = Float64[]
    times = Float64[]
    lb    = Float64[]
    compute_LB = false
    first_feasible_iter = nothing
    rc_counter = 0

    active_columns = []
    active_values = []
    active_iters = []

    iter_times = Float64[]


    t_start = time_ns()
    for iter in 1:n_lp
        it_time = time_ns()
        col_val_buf  = Vector{Cdouble}(undef, 2 * T + nc)
        col_dual_buf = Vector{Cdouble}(undef, 2 * T + nc)

        t0 = time_ns()
        ccall((:Highs_run, libhighs), Cint, (Ptr{Cvoid},), h)
        push!(times, (time_ns() - t0) * 1e-9)

        ccall((:Highs_getSolution, libhighs), Cint,
              (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
              h, col_val_buf, col_dual_buf, row_val_buf, row_dual_buf)

        lam_task  = Vector{Float64}(row_dual_buf[1:T])
        lam_pilot = Vector{Float64}(row_dual_buf[T+1:T+P])
        yw_vals   = Vector{Float64}(col_val_buf[2*T+1:end])
        push!(obj, dot(costs, yw_vals))
        

        # for k in 1:length(yw_vals)
        #     if yw_vals[k] > 1e-3
        #         push!(active_iters, iter)
        #         push!(active_columns, k)
        #         push!(active_values, log(yw_vals[k]))
        #     end
        # end


        s_vals = col_val_buf[1:2*T]
        feasible = all(abs.(s_vals) .< 1e-8)
        @printf("LP %d  %.6f  %.6f  (%.4fs)  feasible: %s\n",
        iter, obj[end], fexact, times[end],
        feasible ? "true" : "false")

        if feasible && first_feasible_iter === nothing
            first_feasible_iter = iter
        end


        pilot = rand(1:P)
        path, cost = sp_dag(cp[(pilot-1)*N .+ (1:N)], A1' * lam_task, F_graph, b_graph)
        rc = dot(cp[(pilot-1)*N .+ (1:N)] .- A1' * lam_task, path) - lam_pilot[pilot]
        @printf("  iter:%d pilot:%d reduced_cost:%.6f\n", iter, pilot, rc)
        if rc > -0.001
            rc_counter += 1
        else
            rc_counter = 0
        end
        rc_counter > 4 && break


        if compute_LB
            rc_sum = 0.0

            for pilot in 1:P
                lb_path, _ = sp_dag(
                    cp[(pilot-1)*N .+ (1:N)],
                    A1' * lam_task,
                    F_graph,
                    b_graph
                )

                rc_pilot = dot(cp[(pilot-1)*N .+ (1:N)] .- A1' * lam_task, lb_path) - lam_pilot[pilot]

                rc_sum += min(rc_pilot, 0.0)
            end

            push!(lb, obj[end] + rc_sum)
        end

        wnew = zeros(P); wnew[pilot] = 1.0
        W     = hcat(W, wnew)
        costs = vcat(costs, cost)
        paths = hcat(paths, path)
        nc   += 1
        lp_add_col!(h, Vector{Float64}(A1 * path), pilot, cost, T)

        push!(iter_times, (time_ns() - it_time) * 1e-9)

        
    end
    t_end = (time_ns() - t_start) * 1e-9
    println("time = $(t_end) seconds\n")


    # STATISTICS
    n_solved = length(times)

    if n_solved > 0
        avg_ms = 1e3 * sum(times) / n_solved

        println("\n── QPAS solve statistics ──────────────────────────────────────────")
        @printf("  CG iterations solved : %d\n", n_solved)
        @printf("  avg time / CG iter   : %.3f ms\n", avg_ms)
    end


    ccall((:Highs_destroy, libhighs), Cvoid, (Ptr{Cvoid},), h)

    if compute_LB
        return obj, times, lb, first_feasible_iter, active_iters, active_columns, active_values
    end

    return obj, times, iter_times
end














struct Column
    pilot::Int
    cost::Float64
    ax_eq::Vector{Float64}
    ax_ineq::Vector{Float64}
end


function add_column!(h, col::Column, M_eq::Int, M_ineq::Int)

    rows = Int[]
    vals = Float64[]

    # equality rows: 1..M_eq
    nz_eq = findall(!iszero, col.ax_eq)
    append!(rows, nz_eq)
    append!(vals, col.ax_eq[nz_eq])

    # inequality rows: M_eq+1..M_eq+M_ineq
    nz_ineq = findall(!iszero, col.ax_ineq)
    append!(rows, M_eq .+ nz_ineq)
    append!(vals, col.ax_ineq[nz_ineq])

    # pilot row: M_eq+M_ineq+pilot
    push!(rows, M_eq + M_ineq + col.pilot)
    push!(vals, 1.0)

    ccall((:Highs_addCols, libhighs), Cint,
        (Ptr{Cvoid}, Cint,
         Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
         Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
        h,
        Cint(1),
        [Cdouble(col.cost)],
        [Cdouble(0.0)],
        [Cdouble(Inf)],
        Cint(length(rows)),
        Cint[0],
        Cint.(rows .- 1),   # HiGHS is 0-indexed
        Cdouble.(vals)
    )
end


function cg_lp_ineq(
    P::Int, N::Int,
    c_orig::Vector{Float64}, cp::Vector{Float64},
    F_graph::SparseMatrixCSC{Float64,Int},
    b_graph::Vector{Float64},
    A_eq::Matrix{Float64},
    A_ineq::Matrix{Float64},
    n_iter::Int
)

    # --------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------
    M_eq   = size(A_eq, 1)
    M_ineq = size(A_ineq, 1)
    T = M_eq + M_ineq

    verbose = false

    if verbose
        println("\n[CG] Start")
        println("  M_eq   = $M_eq")
        println("  M_ineq = $M_ineq")
        println("  P      = $P")
    end


    # --------------------------------------------------
    # HIgHS SETUP
    # --------------------------------------------------
    h = ccall((:Highs_create, libhighs), Ptr{Cvoid}, ())
    ccall((:Highs_setBoolOptionValue, libhighs), Cint,
          (Ptr{Cvoid}, Cstring, Cint),
          h, "output_flag", 0)


    # --------------------------------------------------
    # ROWS (MUST MATCH INDEXING)
    # --------------------------------------------------

    for _ in 1:M_eq
        ccall((:Highs_addRow, libhighs), Cint,
            (Ptr{Cvoid}, Cdouble, Cdouble, Cint, Ptr{Cint}, Ptr{Cdouble}),
            h, 1.0, 1.0, Cint(0), C_NULL, C_NULL)
    end

    for _ in 1:M_ineq
        ccall((:Highs_addRow, libhighs), Cint,
            (Ptr{Cvoid}, Cdouble, Cdouble, Cint, Ptr{Cint}, Ptr{Cdouble}),
            h, 1.0, Inf, Cint(0), C_NULL, C_NULL)
    end

    for _ in 1:P
        ccall((:Highs_addRow, libhighs), Cint,
            (Ptr{Cvoid}, Cdouble, Cdouble, Cint, Ptr{Cint}, Ptr{Cdouble}),
            h, 1.0, 1.0, Cint(0), C_NULL, C_NULL)
    end


    if verbose
        println("[CG] HiGHS rows initialized")
    end


    # --------------------------------------------------
    # SLACK VARIABLES (equality only)
    # --------------------------------------------------
    # s⁺: T columns, cost=10000, +1 in coverage row i
    ccall((:Highs_addCols, libhighs), Cint,
          (Ptr{Cvoid}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
          h, Cint(T),
          fill(Cdouble(10_000.0), T), fill(Cdouble(0.0), T), fill(Cdouble(Inf), T),
          Cint(T), Vector{Cint}(0:T-1), Vector{Cint}(0:T-1), fill(Cdouble(1.0), T))

    # s⁻: T columns, cost=10000, -1 in coverage row i
    ccall((:Highs_addCols, libhighs), Cint,
          (Ptr{Cvoid}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
          h, Cint(T),
          fill(Cdouble(10_000.0), T), fill(Cdouble(0.0), T), fill(Cdouble(Inf), T),
          Cint(T), Vector{Cint}(0:T-1), Vector{Cint}(0:T-1), fill(Cdouble(-1.0), T))


    if verbose
        println("[CG] Slack variables added")
    end



    # --------------------------------------------------
    # WARM START
    # --------------------------------------------------
    paths = zeros(N, P)
    costs = zeros(P)
    W     = Matrix{Float64}(I, P, P)

    # feasible paths
    for pilot in 1:P

        path, cost = sp_dag(
            cp[(pilot-1)*N .+ (1:N)], zeros(N),
            F_graph, b_graph
        )

        paths[:, pilot] = path
        costs[pilot] = cost
    end

    # reduced-cost seed paths
    for pilot in 1:P

        path, cost = sp_dag(
            cp[(pilot-1)*N .+ (1:N)], c_orig,
            F_graph, b_graph
        )

        paths = hcat(paths, path)
        costs = vcat(costs, cost)

        w = zeros(P)
        w[pilot] = 1.0
        W = hcat(W, w)
    end


    # push warm start to HiGHS
    AX_eq   = A_eq * paths
    AX_ineq = A_ineq * paths

    nc = length(costs)
    for k in 1:nc

        pilot_k = findfirst(!iszero, W[:, k])

        col = Column(
            pilot_k,
            costs[k],
            AX_eq[:, k],
            AX_ineq[:, k]
        )

        add_column!(h, col, M_eq, M_ineq)
    end


    if verbose
        println("[CG] Warm start completed")
        println("[CG] Entering main loop...\n")
    end


    # --------------------------------------------------
    # CG LOOP
    # --------------------------------------------------


    row_dual_buf = Vector{Cdouble}(undef, T + P)
    row_val_buf  = Vector{Cdouble}(undef, T + P)

    obj = 0
    iters = 0

    for it in 1:n_iter

        if verbose
            println("────────────────────────────────────")
            println("[ITER $it]")
        end

        ccall((:Highs_run, libhighs), Cint, (Ptr{Cvoid},), h)

        # Extract Solution
        col_val_buf  = Vector{Cdouble}(undef, 2 * T + nc)
        col_dual_buf = Vector{Cdouble}(undef, 2 * T + nc)

        ccall((:Highs_getSolution, libhighs), Cint,
              (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
              h, col_val_buf, col_dual_buf, row_val_buf, row_dual_buf)


        lam_eq  = Vector{Float64}(row_dual_buf[1 :  M_eq])
        lam_ineq = Vector{Float64}(row_dual_buf[M_eq+1 : T])
        lam_pilot = Vector{Float64}(row_dual_buf[T+1 : T+P])

        s_vals = Vector{Float64}(col_val_buf[1 : 2*T])
        yw_vals   = Vector{Float64}(col_val_buf[2*T+1 : end])
        obj = dot(costs, yw_vals)
        iters += 1

        if verbose
            feasible = all(abs.(s_vals) .< 1e-8)
            println("feasible = ",  feasible ? "true" : "false")
            println("LP objective = ", obj)
        end

        improved = false

        for pilot in randperm(P)

            path, cost = sp_dag(
                cp[(pilot-1)*N .+ (1:N)],
                A_eq' * lam_eq .+ A_ineq' * lam_ineq,
                F_graph, b_graph
            )

            rc = dot(cp[(pilot-1)*N .+ (1:N)] .- A_eq' * lam_eq .- A_ineq' * lam_ineq, path) - lam_pilot[pilot]

            if verbose
                println("  pilot $pilot → rc = $rc")
            end

            if rc < -1e-6
                if verbose
                    println("    → add column")
                end

                col = Column(
                    pilot,
                    cost,
                    A_eq * path,
                    A_ineq * path
                )

                costs = vcat(costs, cost)
                nc  += 1
                add_column!(h, col, M_eq, M_ineq)
                improved = true
                break
            end
        end

        if !improved
            if verbose
                println("[CG] STOP: no improvement")
            end
            break
        end
    end


    # --------------------------------------------------
    # RETURN SOLUTION
    # --------------------------------------------------
    ccall((:Highs_destroy, libhighs), Cvoid, (Ptr{Cvoid},), h)

    if verbose
        println("\n[CG DONE]")
        println("objective = ", obj)
    end

    return obj, iters
end