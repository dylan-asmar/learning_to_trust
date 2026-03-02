using Test
using POMDPs
using POMDPTools
using MOMDPs
using TagMOMDPProblemAT
using MetaGraphs
using Graphs

@testset verbose=true "All Tests" begin
    include("constructor_tests.jl")
    include("action_space_tests.jl")
    include("state_space_tests.jl")
    include("reward_tests.jl")
    include("observation_tests.jl")
    include("transition_tests.jl")
    include("rendering_tests.jl")
end
