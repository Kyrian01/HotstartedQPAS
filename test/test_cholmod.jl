# Tests of the CHOLMOD wrapper: identity initialisation, row add/delete in arbitrary order,
# solves, pivots and reset, all against dense reference computations.

const Cholmod = HQ.Cholmod

function _reconstruct(F)
    L, d = Cholmod.factor_matrices(F)
    return Matrix(L * Diagonal(d) * L')
end

@testset "Cholmod wrapper" begin
    @testset "struct layouts" begin
        @test Cholmod.check_struct_layouts() === nothing
        @test Cholmod.cholmod_version() >= v"5.0.0"
    end

    @testset "identity factor" begin
        F = Cholmod.LDLtFactor(7)
        @test _reconstruct(F) ≈ Matrix(I, 7, 7)
        rhs = Cholmod.rhs_buffer(F)
        rhs .= 1:7
        @test Cholmod.solve!(F) ≈ collect(1.0:7)
        @test all(Cholmod.pivot(F, k) == 1.0 for k in 1:7)
        close(F)
        @test_throws ArgumentError Cholmod.solve!(F)
    end

    @testset "random add/delete sequence" begin
        rng = Xoshiro(42)
        n = 30
        B = randn(rng, 60, n)
        Hfull = B' * B                       # SPD
        F = Cholmod.LDLtFactor(n)
        H = Matrix{Float64}(I, n, n)
        active = Int[]
        for step in 1:200
            if isempty(active) || (length(active) < n && rand(rng) < 0.6)
                k = rand(rng, setdiff(1:n, active))
                push!(active, k)
                idx = shuffle(rng, copy(active))          # unsorted on purpose
                @test Cholmod.rowadd!(F, k, idx, Hfull[idx, k], length(idx))
                H[active, active] .= Hfull[active, active]
            else
                k = rand(rng, active)
                Cholmod.rowdel!(F, k)
                filter!(!=(k), active)
                H[k, :] .= 0
                H[:, k] .= 0
                H[k, k] = 1
            end
            if step % 20 == 0
                @test norm(_reconstruct(F) - H) <= 1e-9 * norm(H)
                b = randn(rng, n)
                Cholmod.rhs_buffer(F) .= b
                @test Cholmod.solve!(F) ≈ H \ b rtol = 1e-9
                k = rand(rng, 1:n)
                L, d = Cholmod.factor_matrices(F)
                @test Cholmod.pivot(F, k) ≈ d[k]
            end
        end
        Cholmod.reset!(F)
        @test _reconstruct(F) ≈ Matrix(I, n, n)
    end

    # rank-one updates/downdates (inequality rows) interleaved with row additions/deletions
    @testset "random updown sequence" begin
        rng = Xoshiro(17)
        n = 25
        B = randn(rng, 80, n)
        Hfull = B' * B
        F = Cholmod.LDLtFactor(n)
        H = Matrix{Float64}(I, n, n)
        active = Int[]
        for step in 1:300
            r = rand(rng)
            if isempty(active) || (length(active) < n && r < 0.35)
                k = rand(rng, setdiff(1:n, active))
                # the new row/column must be consistent with the current (modified) active block
                Hk = Hfull[active, k]
                # diagonal chosen so that the new pivot (Schur complement) is at least 1
                hkk = (isempty(active) ? 0.0 : dot(Hk, H[active, active] \ Hk)) + 1 + rand(rng)
                push!(active, k)
                @test Cholmod.rowadd!(F, k, vcat(active[1:(end - 1)], k),
                                      vcat(Hk, hkk), length(active))
                H[active[1:(end - 1)], k] .= Hk
                H[k, active[1:(end - 1)]] .= Hk
                H[k, k] = hkk
            elseif r < 0.5
                k = rand(rng, active)
                Cholmod.rowdel!(F, k)
                filter!(!=(k), active)
                H[k, :] .= 0
                H[:, k] .= 0
                H[k, k] = 1
            else
                # v supported on a random subset of the active indices, unsorted on purpose
                idx = shuffle(rng, active)[1:rand(rng, 1:length(active))]
                v = randn(rng, length(idx))
                up = rand(rng) < 0.5
                # keep the downdate safely positive definite: scale v so that vᵀH⁻¹v ≤ 1/2
                if !up
                    w = zeros(n)
                    w[idx] .= v
                    q = dot(w[active], H[active, active] \ w[active])
                    v ./= sqrt(2q)
                end
                @test Cholmod.updown!(F, up, idx, v, length(idx))
                w = zeros(n)
                w[idx] .= v
                H .+= (up ? 1 : -1) .* (w * w')
            end
            if step % 15 == 0
                @test norm(_reconstruct(F) - H) <= 1e-8 * norm(H)
                b = randn(rng, n)
                Cholmod.rhs_buffer(F) .= b
                @test Cholmod.solve!(F) ≈ H \ b rtol = 1e-7
            end
        end
        @test Cholmod.updown!(F, true, Int[], Float64[], 0)       # v = 0 is a no-op
    end

    @testset "singular downdate is detected through the pivots" begin
        F = Cholmod.LDLtFactor(4)
        Cholmod.rowadd!(F, 1, [1], [2.0], 1)
        Cholmod.rowadd!(F, 3, [1, 3], [1.0, 3.0], 2)
        # H = [2 1; 1 3] on {1, 3}; v with vᵀH⁻¹v = 1 makes H - vvᵀ singular
        H = [2.0 1.0; 1.0 3.0]
        v = H * [1.0, 1.0]
        v ./= sqrt(dot(v, H \ v))
        Cholmod.updown!(F, false, [1, 3], v, 2)
        @test min(Cholmod.pivot(F, 1), Cholmod.pivot(F, 3)) <= 1e-12
    end

    @testset "rhs buffer is not modified by solve" begin
        F = Cholmod.LDLtFactor(5)
        Cholmod.rowadd!(F, 2, [2, 4], [4.0, 1.0], 2)
        Cholmod.rowadd!(F, 4, [2, 4], [1.0, 3.0], 2)
        rhs = Cholmod.rhs_buffer(F)
        rhs .= [0.0, 1.0, 0.0, 2.0, 0.0]
        x = copy(Cholmod.solve!(F))
        @test rhs == [0.0, 1.0, 0.0, 2.0, 0.0]
        @test x ≈ [0.0, 1 / 11, 0.0, 7 / 11, 0.0]
    end
end
