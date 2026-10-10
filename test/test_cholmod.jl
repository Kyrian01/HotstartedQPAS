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
