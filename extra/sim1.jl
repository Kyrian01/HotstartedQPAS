using SparseArrays
using LinearAlgebra
using Random
using Printf
using JuMP
import HiGHS



# ─── Problem generation ───────────────────────────────────────────────────────

function edges_overlap(e1::Int, e2::Int, I::Vector{Int}, J::Vector{Int})
    return I[e1] < J[e2] && I[e2] < J[e1]
end

function can_share_task(ek::Int, t0::Int,
                        task_to_edges::Vector{Vector{Int}},
                        I::Vector{Int}, J::Vector{Int})
    existing = task_to_edges[t0 + 1]
    isempty(existing) && return true
    return all(e -> edges_overlap(ek, e, I, J), existing)
end





"""
    single_pilot_problem2(M; kans, n_tasks, max_tasks_per_edge) -> (c, F, A, b)

Random graph on M nodes.  Returns:
  c  – edge costs (N,)
  F  – flow matrix (M×N, entries ±1)
  A  – task-edge incidence (n_tasks×N, entries 0/1)
  b  – flow rhs: b[1]=1, b[end]=-1, rest 0
"""
function single_pilot_problem2(M::Int;
                               kans::Float64           = 0.05,
                               n_tasks::Int            = 10,
                               max_tasks_per_edge::Int = 10)
    I_v = Int[]; J_v = Int[]; act = Int[]
    for i in 0:(M - 1)
        if i < M - 1
            push!(I_v, i); push!(J_v, i + 1)
        end
        for j in (i + 2):(M - 1)
            if rand() < kans
                push!(act, length(I_v) + 1)
                push!(I_v, i); push!(J_v, j)
            end
        end
    end

    N = length(I_v)
    Fr = Int[]; Fc = Int[]; Fv = Float64[]; c = zeros(N)
    for k in 1:N
        i = I_v[k] + 1; j = J_v[k] + 1
        push!(Fr, i); push!(Fc, k); push!(Fv,  1.0)
        push!(Fr, j); push!(Fc, k); push!(Fv, -1.0)
        c[k] = (J_v[k] - I_v[k]) + rand()
    end
    F = sparse(Fr, Fc, Fv, M, N)
    b = zeros(M); b[1] = 1.0; b[end] = -1.0

    Ar = Int[]; Ac = Int[]; Av = Float64[]
    task_to_edges = [Int[] for _ in 1:n_tasks]
    for idx in eachindex(act)
        ek   = act[idx]
        prog = (I_v[ek] + J_v[ek]) / 2.0 / M
        ct   = min(floor(Int, prog * n_tasks), n_tasks - 1)
        wr   = max(1, floor(Int, n_tasks / 10))
        valid = [t0 for t0 in max(0, ct - wr):min(n_tasks - 1, ct + wr)
                     if can_share_task(ek, t0, task_to_edges, I_v, J_v)]
        if !isempty(valid)
            np = rand(1:min(length(valid), max_tasks_per_edge))
            for t0 in valid[randperm(length(valid))[1:np]]
                t = t0 + 1
                push!(Ar, t); push!(Ac, ek); push!(Av, 1.0)
                push!(task_to_edges[t], ek)
            end
        end
    end
    A = isempty(Ar) ? spzeros(Float64, n_tasks, N) :
                      sparse(Ar, Ac, Av, n_tasks, N)
    return c, F, A, b
end




"""
    coupled_problem(P, c, F, A, b, fraction) -> (task_matrix, rhs)

Return the task-coverage matrix and rhs for a random subset of coverable tasks.

A task is coverable if and only if it has at least one associated arc (non-zero
row in A). Tasks with no arcs would require 0 = 1 in the set-partitioning LP
and are never feasible. Any task with arcs can always be covered by routing one
pilot through one of its arcs while the remaining pilots use the sequential
backbone edges as a bypass — so no LP is needed to check feasibility.
"""
function coupled_problem1(P::Int, c::Vector{Float64},
                         F::SparseMatrixCSC{Float64,Int},
                         A::SparseMatrixCSC{Float64,Int},
                         b::Vector{Float64}, fraction::Float64)
    T = size(A, 1)

    # A task is coverable iff its row in A is non-zero (has at least one arc).
    # Tasks with zero rows would require 0 = 1 in set-partitioning and are infeasible.
    row_nnz = Vector(sum(A .!= 0; dims = 2)[:, 1])
    to_cov  = [i for i in 1:T if row_nnz[i] > 0 && rand() < fraction]

    # Return only the per-pilot arc matrix (T_sel × N). The N*P replication is
    # never needed here — each pilot uses the same A; the block structure is
    # built inside solve_exact_lp and cg_qp_qpas directly.
    return Matrix(A[to_cov, :]), ones(length(to_cov))
end





"""
    coupled_problem(P, c, F, A, b, fraction) -> (task_matrix, rhs)

Build the multi-pilot coupled task-coverage LP, solve it to find coverability,
then return the rows (tasks) that are covered and randomly selected.
"""
function coupled_problem2(P::Int, c::Vector{Float64},
                         F::SparseMatrixCSC{Float64,Int},
                         A::SparseMatrixCSC{Float64,Int},
                         b::Vector{Float64}, fraction::Float64)
    M, N = size(F);  T = size(A, 1)

    ctm = spzeros(Float64, T, N * P)
    for p in 1:P
        ctm[:, (p - 1)*N .+ (1:N)] = A
    end

    Aeq = spzeros(Float64, M * P, N * P);  beq = zeros(M * P)
    for p in 1:P
        Aeq[(p-1)*M .+ (1:M), (p-1)*N .+ (1:N)] = F
        beq[(p-1)*M .+ (1:M)] = b
    end

    model = Model(HiGHS.Optimizer)
    set_attribute(model, "output_flag", false)
    @variable(model, 0.0 <= x[1:N*P] <= 1.0)
    @constraint(model, ctm * x .<= ones(T))
    @constraint(model, Aeq * x .== beq)
    @objective(model, Max, sum(ctm * x))
    optimize!(model)
    st = termination_status(model)
    st ∈ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ||
        error("coupled_problem: LP solver returned $st")

    xv      = value.(x)
    covered = Vector(ctm * xv)
    to_cov  = [i for i in 1:T if covered[i] > 0.99 && rand() < fraction]
    return Matrix(ctm[to_cov, :]), covered[to_cov]
end



function split_task_matrix(task_matrix::AbstractMatrix,
                           frac_ineq::Float64)

    T = size(task_matrix, 1)

    n_ineq = round(Int, frac_ineq * T)

    perm = randperm(T)

    ineq_rows = perm[1:n_ineq]
    eq_rows   = perm[(n_ineq+1):end]

    A_ineq = task_matrix[ineq_rows, :]
    A_eq   = task_matrix[eq_rows, :]

    return A_eq, A_ineq, eq_rows, ineq_rows
end