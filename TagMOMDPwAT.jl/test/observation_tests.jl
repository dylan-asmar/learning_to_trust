@testset "observations" begin
    map_str = """
    xxoo
    xoox
    oooo
    """
    momdp = TagMOMDPAT(; map_str=map_str)

    @inferred observations(momdp)
    
    obs = observations(momdp)
    @test length(obs) == 2
    
    @test obsindex(momdp, 1) == 1
    @test obsindex(momdp, 2) == 2
    @test_throws ErrorException obsindex(momdp, 7)
    @test_throws ErrorException obsindex(momdp, 0)

    for ri in 1:get_prop(momdp.mg, :num_grid_pos)
        sp = (TagMOMDPXState(ri, 0), TagMOMDPYState(1, 1.0))
        for a in actions(momdp)
            od = observation(momdp, a, sp)
            @test isa(od, SparseCat{Vector{Int}, Vector{Float64}})
            if ri == 1 && a != TagMOMDPProblemAT.ACTIONS_DICT[:ask]
                @test pdf(od, 1) == 1.0
            elseif a == TagMOMDPProblemAT.ACTIONS_DICT[:ask]
                @test pdf(od, 1) == 0.0
                @test pdf(od, 2) == 0.0
                @test pdf(od, 3) != 0.0
                @test pdf(od, 4) != 0.0
            end
        end
    end

    @test has_consistent_observation_distributions(momdp)
    
    momdp = TagMOMDPAT(; map_str=map_str, num_asks=-1)
    @inferred observations(momdp)
    
    obs = observations(momdp)
    @test length(obs) == 7
    @test obsindex(momdp, 1) == 1
    @test obsindex(momdp, 2) == 2
    @test obsindex(momdp, 7) == 7
    @test_throws ErrorException obsindex(momdp, 8)
    @test_throws ErrorException obsindex(momdp, 0)

    for ri in 1:get_prop(momdp.mg, :num_grid_pos)
        sp = (TagMOMDPXState(ri, 1), TagMOMDPYState(1, 1.0))
        for a in actions(momdp)
            od = observation(momdp, a, sp)
            @test isa(od, SparseCat{Vector{Int}, Vector{Float64}})
            if ri == 1 && a != TagMOMDPProblemAT.ACTIONS_DICT[:ask]
                @test pdf(od, 1) == 1.0
            elseif a == TagMOMDPProblemAT.ACTIONS_DICT[:ask]
                @test pdf(od, 1) == 0.0
                @test pdf(od, 2) == 0.0
                @test pdf(od, 3) != 0.0
                @test pdf(od, 4) != 0.0
            end
        end
    end

    @test has_consistent_observation_distributions(momdp)
    
    momdp = TagMOMDPAT(; num_asks=-1, types=[1.0, 2.0])
    sp = (TagMOMDPXState(1, 1), TagMOMDPYState(2, 2.0))
    od = observation(momdp, 6, sp)
    @test isa(od, SparseCat{Vector{Int}, Vector{Float64}})
    @test pdf(od, 1) == 0.0
    @test pdf(od, 2) == 0.0
    @test pdf(od, 3) != 0.0
    @test pdf(od, 4) != 0.0
end
