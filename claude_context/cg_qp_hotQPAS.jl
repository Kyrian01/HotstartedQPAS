

include("../pricing_solvers/shortest_path.jl")
include("../qpas_solvers/qpas.jl")



function cg_qp_qpas(
    P::Int, N::Int, T::Int,
    cp::Vector{Float64},
    task_matrix::Matrix{Float64},
    F_graph::SparseMatrixCSC{Float64,Int},
    b_graph::Vector{Float64},
    fexact::Float64;
    n_cg::Int64 = 5_000,
    delta::Float64 = 1.0
)

    # Intial column generation (pilot paths)
    A1 = task_matrix[:, 1:N]
    col_max = 2 * P + n_cg

    paths = zeros(N, P)
    costs_init = zeros(P)

    for pilot in 1:P
        paths[:, pilot], costs_init[pilot] =
            sp_dag(cp[(pilot-1)*N .+ (1:N)], zeros(N), F_graph, b_graph)
    end


    # Master problem initialization
    constr_mat = spzeros(Float64, T, col_max)
    convex_mat = spzeros(Float64, P, col_max)

    for pilot in 1:P
        constr_mat[:, pilot] .= A1 * paths[:, pilot]
        convex_mat[pilot, pilot] = 1.0
    end

    cost_vec = zeros(Float64, col_max)
    cost_vec[1:P] .= costs_init

    col_wts = zeros(Float64, col_max)
    col_wts[1:P] .= 1.0

    num_cols = P



    # Initialize KKT and MasterProblemData structures
    kkt = KKTFactor(
        convex_mat,     # convexity_matrix
        constr_mat,     # constraint_matrix
        Int64[],        # working_set
        num_cols,       # num_cols
        col_max         # col_max
    )

    mpd = MasterProblemData(
        col_max,                                        # col_max
        N,                                              # n 
        T,                                              # m
        P,                                              # p
        0,                                              # iteration
        num_cols,                                       # num_cols
        spzeros(Float64, N, col_max),                   # columns
        vcat(collect(1:P), zeros(Int64, col_max - P)),  # column_ids
        cost_vec,                                       # cost
        constr_mat,                                     # constraint_matrix 
        ones(T),                                        # constraint_rhs
        convex_mat,                                     # convexity_matrix
        0,                                              # last_QPAS_iters
        zeros(Float64, T),                              # stability_center
        delta,                                          # delta
        kkt,                                            # kkt
        col_wts,                                        # column_weights
        zeros(Float64, N * P),                          # primal_solution
        Inf,                                            # objective_value
        zeros(Float64, T),                              # dual_constraint
        zeros(Float64, P),                              # dual_convexity
        -Inf,                                           # reduced_cost
        false                                           # converged
    )



    # Timing Setup
    reset_timer!(to)

    obj        = Float64[]
    times      = Float64[]
    qpas_iters = Int[]
    rc_counter = 0

    iter_times = Float64[]


    # COLUMN GENERATION LOOP
    t_start = time_ns()
    for iter in 1:n_cg

        it_time = time_ns()

        mpd.iteration = iter
        mpd.num_cols  = num_cols


        # Solve QP master problem using hotQPAS
        t0 = time_ns()

        solve_hotQPAS!(mpd)

        push!(times, (time_ns() - t0) * 1e-9)
        push!(qpas_iters, mpd.last_qpas_iters)


        # Objective tracking
        push!(obj, dot(mpd.cost[1:num_cols], mpd.column_weights[1:num_cols]))

        @printf("QP_QPAS %d  %.6f  %.6f  (%.4fs, %d qpas-iters)\n",
                iter, obj[end], fexact, times[end], qpas_iters[end])



        # Pricing step (using DAG shortest path)
        pilot     = rand(1:P)
        lam_cov   = mpd.dual_constraint
        lam_pilot = mpd.dual_convexity

        path = cost_new = rc = nothing
        @timeit to "pricing" begin
            rc_vec     = A1' * lam_cov
            cp_pilot   = cp[(pilot-1)*N .+ (1:N)]
            path, cost_new = sp_dag(cp_pilot, rc_vec, F_graph, b_graph)
            rc = dot(cp_pilot .- rc_vec, path) - lam_pilot[pilot]
        end
        @printf("  iter:%d pilot:%d reduced_cost:%.6f\n", iter, pilot, rc)


        # Early stop criteria
        if rc > -0.001
            rc_counter += 1
        else
            rc_counter = 0
        end

        if rc_counter > 4
            break
        end


        # Add new column
        num_cols += 1

        mpd.cost[num_cols] = cost_new
        mpd.constraint_matrix[:, num_cols] .= A1 * path
        mpd.convexity_matrix[pilot, num_cols] = 1.0

        push!(mpd.kkt.working_set, num_cols)
        push!(mpd.kkt.column_to_block, pilot)
        push!(mpd.kkt.column_to_set, UInt8(2))

        push!(iter_times, (time_ns() - it_time) * 1e-9)

    end
    t_end = (time_ns() - t_start) * 1e-9
    println("time = $(t_end) seconds\n")


    # STATISTICS
    n_solved = length(times)

    if n_solved > 0
        avg_ms = 1e3 * sum(times) / n_solved
        avg_qi = sum(qpas_iters) / n_solved
        avg_us_qi = avg_qi > 0 ? avg_ms * 1e3 / avg_qi : NaN

        println("\n── QPAS solve statistics ──────────────────────────────────────────")
        @printf("  CG iterations solved : %d\n", n_solved)
        @printf("  avg time / CG iter   : %.3f ms\n", avg_ms)
        @printf("  avg QPAS iters / CG  : %.1f\n", avg_qi)
        @printf("  avg time / QPAS iter : %.2f μs\n", avg_us_qi)
    end


    # PROFILING OUTPUT
    println("\n── QPAS profiler (TimerOutputs) ──────────────────────────────────\n")
    show(to; sortby = :time)
    println()


    return obj, times, qpas_iters, iter_times
end