@testset "action space" begin
    momdp = TagMOMDPAT(; num_asks=0)
    
    TagMOMDPProblemAT.list_actions(momdp)
    
    @test length(actions(momdp)) == 5
    for (ii, ai) in enumerate(ordered_actions(momdp))
        @test ii == ai
    end
    @test actionindex(momdp, :north) == 1
    @test actionindex(momdp, :tag) == 5
    @test_throws AssertionError actionindex(momdp, 6)
    @test_throws AssertionError actionindex(momdp, :invalid)
    @test_throws AssertionError actionindex(momdp, :ask)
    
    momdp = TagMOMDPAT(; num_asks=-1)
    @test length(actions(momdp)) == 6
    for (ii, ai) in enumerate(ordered_actions(momdp))
        @test ii == ai
    end
    @test actionindex(momdp, :ask) == 6
    @test_throws AssertionError actionindex(momdp, 7)

    momdp = TagMOMDPAT(; num_asks=5)
    @test length(actions(momdp)) == 6
end
