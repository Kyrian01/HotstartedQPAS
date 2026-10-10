using Test
using HotstartedQPAS
using LinearAlgebra
using SparseArrays
using Random

const HQ = HotstartedQPAS

# Instance sizes used by the tests; keyword arguments override any InstanceConfig field.
const TEST_SIZES = Dict(
    :tiny => (n_nodes = 20, edge_probability = 0.2, n_tasks = 12, n_pilots = 3),
    :small => (n_nodes = 50, edge_probability = 0.098, n_tasks = 40, n_pilots = 4),
)
test_instance(size::Symbol; kwargs...) =
    generate_instance(InstanceConfig(; merge(TEST_SIZES[size], NamedTuple(kwargs))...))

@testset "HotstartedQPAS" begin
    include("test_cholmod.jl")
    include("test_problem.jl")
    include("test_qpas.jl")
    include("test_solvers.jl")
    include("test_experiments.jl")
end
