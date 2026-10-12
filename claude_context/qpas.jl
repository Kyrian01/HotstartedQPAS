
include("KKTFactor.jl")

include("KKTSolve.jl")
include("KKTUpdate.jl")

include("masterProblemData.jl")



function solve_hotQPAS!(
    MPData::MasterProblemData;
    max_qpas_iters::Int64 = 2_000
)

    # Unpack Data
    A = MPData.constraint_matrix
    b = MPData.constraint_rhs
    delta = MPData.delta
    s = MPData.stability_center
    kkt = MPData.kkt
    num_cols = MPData.num_cols


    # Modified cost
    c_mod = MPData.cost .- A' * (b / delta + s)


    # Warm-start initial iterate
    x = copy(MPData.column_weights)
    grad = (1 / delta) * (A' * (A * x)) + c_mod


    # QPAS Lagrange multipliers
    mu = Float64[]
    lam = Float64[]


    # QPAS state
    final_iter = 0



    # MAIN QPAS LOOP
    for iter in 1:max_qpas_iters

        final_iter = iter


        # Solve KKT System
        sol = @timeit to "KKT solve" solve_KKTsystem(kkt, grad, A, delta)
        p, mu, lam, AtAp = sol.p, sol.mu, sol.lam, sol.AtAp



        # CHECK OPTIMALITY (NO SEARCH DIRECTION)
        if norm(p) < 1e-8

            if isempty(lam)
                break
            end

            # find most negative multiplier
            min_val, min_idx = findmin(lam)

            if min_val > -1e-8
                break
            end

            r = kkt.working_set[min_idx]

            @timeit to "reduce WS" k = reduce_workingset!(kkt, A, r)

            # rollback if needed
            if k != 0
                expand_workingset!(kkt, k)

                if reduce_workingset!(kkt, A, r) != 0
                    error("ROLLBACK FAILED")
                end
            end



        # TAKE PRIMAL STEP
        else

            alpha = 1.0
            q = 0

            @inbounds for i in 1:num_cols
                p_i = p[i]

                if p_i < -1e-8 && kkt.column_to_set[i] != UInt8(2)

                    alpha_i = -x[i] / p_i

                    if alpha_i < alpha
                        alpha = alpha_i
                        q = i
                    end
                end
            end

            # iterate and gradient update (in-place)
            axpy!(alpha, p, x)
            axpy!(alpha, AtAp, grad)

            # expand working set with blocking constraint
            if alpha < 1.0
                @timeit to "expand WS" expand_workingset!(kkt, q)
            end
        end

    end


    # Final QPAS iteration count
    MPData.last_qpas_iters = final_iter


    # Objective value
    Ax = A * x

    MPData.objective_value =
        dot(c_mod, x) +
        (0.5 / delta) * dot(Ax, Ax) +
        (0.5 / delta) * dot(b + delta * s, b + delta * s) -
        (delta / 2) * dot(s, s)


    
    # Dual Variables 
    dual = s + (1 / delta) * (b - A * x)

    MPData.column_weights .= x
    MPData.dual_constraint .= dual
    MPData.dual_convexity .= mu

    MPData.stability_center = dual

end