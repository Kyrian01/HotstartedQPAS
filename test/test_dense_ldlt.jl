# Tests of the dense LDLᵀ backend: the same operations as the CHOLMOD wrapper, checked against
# dense reference matrices and against CHOLMOD itself on identical update sequences.

const DenseLDLt = HQ.DenseLDLt
const LDLt = HQ.LDLt

function _reconstruct_any(F)
    L, d = LDLt.factor_matrices(F)
    return Matrix(L * Diagonal(d) * L')
end

# Random SPD matrix on 1:n.
function _random_spd(rng, n; rows = 3n)
    B = randn(rng, rows, n)
    return B' * B
end

# Apply one random operation (activate / deactivate / ± rank one) to every factor in `Fs`
# and to the mirror `H` of the factorized matrix (identity on inactive indices). Activations
# and rank-one changes are first tried on the mirror; one that would make the active block
# worse conditioned than 1e6 is replaced by a deactivation (which never worsens it, by
# eigenvalue interlacing). So the comparisons test the update rules, not round-off.
# Returns the operation name.
_well_conditioned(H, act) = isempty(act) || cond(H[act, act]) < 1e6

function _random_op!(rng, Fs, H, active, n; p_add = 0.4, p_del = 0.3)
    r = rand(rng)
    if isempty(active) || (length(active) < n && r < p_add)
        k = rand(rng, setdiff(1:n, active))
        # new row h = H c (small c) and Schur complement s ∈ [1, 2]: the extended matrix is
        # [I 0; cᵀ 1] diag(H, s) [I c; 0 1]
        c = 0.5 .* randn(rng, length(active)) ./ sqrt(length(active) + 1)
        Hk = H[active, active] * c
        hkk = dot(c, Hk) + 1 + rand(rng)
        H2 = copy(H)
        H2[active, k] .= Hk
        H2[k, active] .= Hk
        H2[k, k] = hkk
        if _well_conditioned(H2, vcat(active, k))
            idx = vcat(active, k)
            vals = vcat(Hk, hkk)
            perm = shuffle(rng, 1:length(idx))               # unsorted on purpose
            for F in Fs
                @test LDLt.rowadd!(F, k, idx[perm], vals[perm], length(idx))
            end
            H .= H2
            push!(active, k)
            return :add
        end
    elseif r >= p_add + p_del
        idx = shuffle(rng, active)[1:rand(rng, 1:length(active))]
        v = randn(rng, length(idx)) ./ sqrt(length(idx))
        up = rand(rng) < 0.5
        w = zeros(n)
        w[idx] .= v
        if !up
            # keep the downdate safely positive definite: vᵀH⁻¹v ≤ 1/2
            q = dot(w[active], H[active, active] \ w[active])
            v ./= sqrt(2q)
            w[idx] .= v
        end
        H2 = H .+ (up ? 1 : -1) .* (w * w')
        if _well_conditioned(H2, active)
            for F in Fs
                @test LDLt.updown!(F, up, idx, v, length(idx))
            end
            H .= H2
            return up ? :up : :down
        end
    end
    # deactivation (also the fallback of a rejected candidate)
    k = rand(rng, active)
    for F in Fs
        LDLt.rowdel!(F, k)
    end
    filter!(!=(k), active)
    H[k, :] .= 0
    H[:, k] .= 0
    H[k, k] = 1
    return :del
end

@testset "dense LDLᵀ backend" begin
    @testset "interface" begin
        F = DenseLDLt.DenseLDLtFactor(9)
        @test F isa HQ.UpdatableLDLt
        @test LDLt.backend_name(F) == "dense"
        @test !LDLt.needs_periodic_refactor(F)
        @test LDLt.needs_periodic_refactor(HQ.Cholmod.LDLtFactor(3))
        @test LDLt.dimension(F) == 9
        @test _reconstruct_any(F) ≈ Matrix(I, 9, 9)
        rhs = LDLt.rhs_buffer(F)
        rhs .= 1:9
        @test LDLt.solve!(F) ≈ collect(1.0:9)                 # identity
        @test all(LDLt.pivot(F, k) == 1.0 for k in 1:9)
        close(F)
        @test_throws ArgumentError LDLt.solve!(F)
        @test_throws ArgumentError DenseLDLt.DenseLDLtFactor(0)
    end

    @testset "append rule: pivot is the Schur complement" begin
        F = DenseLDLt.DenseLDLtFactor(6; capacity = 2)
        H = [4.0 1.0 2.0; 1.0 3.0 0.5; 2.0 0.5 5.0]
        LDLt.rowadd!(F, 5, [5], [4.0], 1)
        LDLt.rowadd!(F, 2, [5, 2], [1.0, 3.0], 2)
        @test LDLt.rowadd!(F, 4, [2, 4, 5], [0.5, 5.0, 2.0], 3)   # grows capacity 2 → 4
        @test F.cap >= 3 && F.grows >= 1
        idx = [5, 2, 4]
        M = _reconstruct_any(F)
        @test M[idx, idx] ≈ H
        # pivot of the last activated index = Schur complement h₄₄ - h₁₂ᵀ H₁₁⁻¹ h₁₂
        S = H[3, 3] - dot(H[1:2, 3], H[1:2, 1:2] \ H[1:2, 3])
        @test LDLt.pivot(F, 4) ≈ S
        # a dependent column gives a non-positive Schur complement and rowadd! returns false
        h = H[:, 1] + H[:, 2]                                  # = column of u₅ + u₂
        G = DenseLDLt.DenseLDLtFactor(6)
        LDLt.rowadd!(G, 5, [5], [4.0], 1)
        LDLt.rowadd!(G, 2, [5, 2], [1.0, 3.0], 2)
        hjj = dot([1.0, 1.0], H[1:2, 1:2] * [1.0, 1.0])        # ‖u₅ + u₂‖²
        @test !LDLt.rowadd!(G, 1, [5, 2, 1], [h[1], h[2], hjj - 1e-9], 3)
    end

    @testset "invalid operations" begin
        F = DenseLDLt.DenseLDLtFactor(5)
        LDLt.rowadd!(F, 1, [1], [2.0], 1)
        @test_throws ArgumentError LDLt.rowadd!(F, 1, [1], [2.0], 1)          # already active
        @test_throws ArgumentError LDLt.rowadd!(F, 2, [3, 2], [1.0, 2.0], 2)  # couples to inactive 3
        @test_throws ArgumentError LDLt.updown!(F, true, [4], [1.0], 1)       # inactive
        @test_throws BoundsError LDLt.rowadd!(F, 6, [6], [1.0], 1)
        LDLt.rowdel!(F, 3)                                                     # inactive: no-op
        @test F.k == 1
        @test LDLt.updown!(F, true, Int[], Float64[], 0)                      # v = 0
    end

    @testset "deleting the first, a middle and the last slot" for pos in (:first, :middle, :last)
        rng = Xoshiro(5)
        n = 12
        Hfull = _random_spd(rng, n)
        F = DenseLDLt.DenseLDLtFactor(n)
        order = shuffle(rng, 1:8)
        for (t, k) in enumerate(order)
            idx = order[1:t]
            LDLt.rowadd!(F, k, idx, Hfull[idx, k], t)
        end
        victim = pos === :first ? order[1] : pos === :last ? order[end] : order[4]
        LDLt.rowdel!(F, victim)
        rest = filter(!=(victim), order)
        M = _reconstruct_any(F)
        @test M[rest, rest] ≈ Hfull[rest, rest] rtol = 1e-12
        @test M[victim, victim] == 1.0 && all(M[victim, rest] .== 0)
        @test F.k == 7 && F.slot[victim] == 0
        @test sort(F.index[1:F.k]) == sort(rest)
        @test all(F.slot[F.index[s]] == s for s in 1:F.k)
        # trailing fraction recorded for the statistics
        expected = pos === :first ? 7 / 8 : pos === :last ? 0.0 : 4 / 8
        @test LDLt.backend_stats(F).mean_trailing_fraction ≈ expected
    end

    @testset "random add/delete/updown sequence (seed $seed)" for seed in 1:4
        rng = Xoshiro(1000 + seed)
        n = 30
        F = DenseLDLt.DenseLDLtFactor(n; capacity = 3)        # forces several reallocations
        H = Matrix{Float64}(I, n, n)
        active = Int[]
        ops = Dict{Symbol,Int}()
        for step in 1:400
            op = _random_op!(rng, (F,), H, active, n)
            ops[op] = get(ops, op, 0) + 1
            if step % 20 == 0
                @test norm(_reconstruct_any(F) - H) <= 1e-9 * norm(H)
                b = randn(rng, n)
                LDLt.rhs_buffer(F) .= b
                @test LDLt.solve!(F) ≈ H \ b rtol = 1e-8
                # pattern solve: right-hand side supported on the active set
                if !isempty(active)
                    rhs = LDLt.rhs_buffer(F)
                    fill!(rhs, 0.0)
                    rhs[active] .= b[active]
                    x = LDLt.solve!(F, active, length(active))
                    @test x[active] ≈ H[active, active] \ b[active] rtol = 1e-8
                    @test LDLt.last_reach(F) == length(active)
                    fill!(rhs, 0.0)
                end
            end
            @test cond(H) < 1e9
        end
        @test all(get(ops, op, 0) > 10 for op in (:add, :del, :up, :down))
        @test F.grows >= 1
        LDLt.reset!(F)
        @test _reconstruct_any(F) ≈ Matrix(I, n, n)
        @test F.k == 0 && all(==(0), F.slot)
    end

    @testset "identical sequences on CHOLMOD and dense (seed $seed)" for seed in 1:3
        rng = Xoshiro(2000 + seed)
        n = 40
        C = HQ.Cholmod.LDLtFactor(n)
        D = DenseLDLt.DenseLDLtFactor(n; capacity = 8)
        H = Matrix{Float64}(I, n, n)
        active = Int[]
        for step in 1:300
            _random_op!(rng, (C, D), H, active, n)
            if step % 25 == 0 && !isempty(active)
                MC, MD = _reconstruct_any(C), _reconstruct_any(D)
                @test norm(MC - MD) <= 1e-9 * norm(H)
                b = zeros(n)
                b[active] .= randn(rng, length(active))
                for F in (C, D)
                    LDLt.rhs_buffer(F) .= b
                end
                xC = copy(LDLt.solve!(C, active, length(active)))
                xD = copy(LDLt.solve!(D, active, length(active)))
                @test xD[active] ≈ xC[active] rtol = 1e-8
                # the minimum scaled pivot is invariant information (positive definiteness)
                scale = ones(n)
                @test LDLt.min_scaled_pivot(D, active, scale) > 0
                @test LDLt.min_scaled_pivot(C, active, scale) > 0
                for F in (C, D)
                    fill!(LDLt.rhs_buffer(F), 0.0)
                end
            end
        end
    end

    @testset "singular downdate" begin
        F = DenseLDLt.DenseLDLtFactor(4)
        LDLt.rowadd!(F, 1, [1], [2.0], 1)
        LDLt.rowadd!(F, 3, [1, 3], [1.0, 3.0], 2)
        H = [2.0 1.0; 1.0 3.0]
        # vᵀH⁻¹v = 1 makes H - vvᵀ singular: a tiny or non-positive pivot appears
        v = H * [1.0, 1.0]
        v ./= sqrt(dot(v, H \ v))
        ok = LDLt.updown!(F, false, [1, 3], v, 2)
        @test !ok || min(LDLt.pivot(F, 1), LDLt.pivot(F, 3)) <= 1e-12
        # vᵀH⁻¹v > 1: indefinite, reported as failure
        G = DenseLDLt.DenseLDLtFactor(4)
        LDLt.rowadd!(G, 1, [1], [2.0], 1)
        LDLt.rowadd!(G, 3, [1, 3], [1.0, 3.0], 2)
        @test !LDLt.updown!(G, false, [1, 3], 1.1 .* v, 2)
    end

    @testset "statistics" begin
        F = DenseLDLt.DenseLDLtFactor(50; capacity = 4)
        LDLt.rowadd!(F, 2, [2], [4.0], 1)
        LDLt.rowadd!(F, 4, [2, 4], [1.0, 3.0], 2)
        LDLt.rowadd!(F, 7, [2, 7], [1.0, 5.0], 2)              # h(7,4) = 0 …
        fs = LDLt.factor_stats(F, [2, 4, 7])
        @test fs.stored == 3                                   # k(k-1)/2 with k = 3
        L, _ = LDLt.factor_matrices(F)
        @test fs.effective == count(!iszero, L - I) == 3      # … but ℓ(7,4) fills in via 2
        @test fs.active_effective == fs.effective
        @test fs.allocated == F.cap^2
        @test fs.memory > 0 && fs.memory_peak >= fs.memory
        bs = LDLt.backend_stats(F)
        @test bs.capacity == 4 && bs.peak_active == 3 && bs.deletions == 0
        LDLt.rowdel!(F, 2)
        @test LDLt.factor_stats(F, [4, 7]).stored == 1        # compacted: no garbage left
        @test LDLt.backend_stats(F).deletions == 1
    end
end
