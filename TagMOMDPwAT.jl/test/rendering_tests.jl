@testset "rendering" begin
    momdp = TagMOMDPAT(; num_asks=2, types=[1.0, 2.0])
    render(momdp)
    s0 = (TagMOMDPXState(2, 0), TagMOMDPYState(4, 1.0))
    render(momdp, (s=s0,))
    
    render(momdp, (s=s0, a=2))
    
    b0 = initialstate(momdp)
    render(momdp, (s=s0, a=3, b=b0))
    s0 = (TagMOMDPXState(3, 0), TagMOMDPYState(3, 1.0))
    render(momdp, (s=s0, a=4))
    render(momdp, (s=s0, a=5); pre_act_text="Pre-Action Text ")

    s1 = (TagMOMDPXState(1, 0), TagMOMDPYState(2, 1.0))
    s2 = (TagMOMDPXState(2, 0), TagMOMDPYState(3, 2.0))
    s3 = (TagMOMDPXState(1, 0), TagMOMDPYState(5, 1.0))
    s4 = (TagMOMDPXState(1, 0), TagMOMDPYState(7, 1.0))
    render(momdp, (b=SparseCat([s1, s2, s3, s4], [0.1, 0.2, 0.3, 0.4]), a=4))

    # Test belief vector rendering
    b = zeros(length(states(momdp)))
    b[2] = 0.1
    b[10] = 0.3
    b[end-10] = 0.4
    b[end-1] = 0.2
    render(momdp, (b=b,))

    # Test rendering with DiscreteBelief
    b_db = DiscreteBelief(momdp, ordered_states(momdp), b)
    render(momdp, (b=b_db,))

    # Test error handling with invalid belief vector
    b = zeros(length(states(momdp)) - 2)
    @test_throws AssertionError render(momdp, (b=b,))
end
