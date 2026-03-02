@testset "reward" begin
    momdp = TagMOMDPAT()
    s = (TagMOMDPXState(1, 0), TagMOMDPYState(10, 1.0))
    for ai in 1:4
        @test reward(momdp, s, ai) == momdp.step_penalty
    end
    @test reward(momdp, s, 5) == momdp.tag_penalty

    s = (TagMOMDPXState(10, 0), TagMOMDPYState(10, 1.0))
    for ai in 1:4
        @test reward(momdp, s, ai) == momdp.step_penalty
    end
    @test reward(momdp, s, 5) == momdp.tag_reward

    s = (TagMOMDPXState(0, 0), TagMOMDPYState(10, 1.0))
    for ai in 1:5
        @test reward(momdp, s, ai) == 0.0
    end
    
    s = (TagMOMDPXState(1, 0), TagMOMDPYState(10, 1.0))
    @test_throws AssertionError reward(momdp, s, 6)
    
    momdp = TagMOMDPAT(; num_asks=1)
    s = (TagMOMDPXState(1, 1), TagMOMDPYState(10, 1.0))
    for ai in 1:4
        @test reward(momdp, s, ai) == momdp.step_penalty
    end
    @test reward(momdp, s, 5) == momdp.tag_penalty
    
    @test reward(momdp, s, 6) == momdp.ask_penalty
    
    s = (TagMOMDPXState(1, 0), TagMOMDPYState(10, 1.0))
    @test reward(momdp, s, 6) == momdp.tag_penalty
end
