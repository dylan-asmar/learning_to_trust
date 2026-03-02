@testset "state space" begin
    momdp = TagMOMDPAT(; num_asks=0)
    num_grid_pos = get_prop(momdp.mg, :num_grid_pos)
    
    @test !isterminal(momdp, (TagMOMDPXState(0, 0), TagMOMDPYState(1, 1)))
    @test !isterminal(momdp, (TagMOMDPXState(1, 1), TagMOMDPYState(1, 2)))
    
    states_vec_x = ordered_states_x(momdp)
    states_vec_y = ordered_states_y(momdp)
    
    @test length(states_vec_x) == num_grid_pos + 1
    @test length(states_vec_y) == num_grid_pos
    
    @test stateindex_x(momdp, TagMOMDPXState(0, 0)) == 1
    @test stateindex_x(momdp, TagMOMDPXState(5, 0)) == 6
    @test_throws AssertionError stateindex_x(momdp, TagMOMDPXState(5, 1))

    @test stateindex_y(momdp, TagMOMDPYState(1, 1)) == 1
    @test stateindex_y(momdp, TagMOMDPYState(6, 1)) == 6
    @test_throws AssertionError stateindex_y(momdp, TagMOMDPYState(6, 2))
    
    xid = initialstate_x(momdp)
    yid = initialstate_y(momdp, TagMOMDPXState(1, 0))
    @test xid isa SparseCat
    @test yid isa SparseCat
    @test length(xid.probs) == num_grid_pos
    @test length(yid.probs) == num_grid_pos
    @test isapprox(xid.probs[1], 1/num_grid_pos; atol=1e-6)
    @test isapprox(yid.probs[1], 1/num_grid_pos; atol=1e-6)
    @test isapprox(sum(xid.probs), 1.0; atol=1e-6)
    @test isapprox(sum(yid.probs), 1.0; atol=1e-6)
    @test has_consistent_initial_distribution(momdp)
    
    
    momdp = TagMOMDPAT(; num_asks=0, types=[1.0, 2.0, 3.0], init_type_dist=[0.1, 0.2, 0.7])
    yid = initialstate_y(momdp, TagMOMDPXState(1, 0))
    @test length(yid.probs) == num_grid_pos * 3
    @test isapprox(sum(yid.probs), 1.0; atol=1e-6)
    # Check marginalized type distributions
    p_type = zeros(length(momdp.types))
    for (yi, pyi) in zip(yid.vals, yid.probs)
        type_i = findfirst(isequal(yi.sugg_type), momdp.types)
        p_type[type_i] += pyi
    end
    p_type = p_type ./ sum(p_type)
    @test isapprox(p_type, [0.1, 0.2, 0.7]; atol=1e-6)
    
    @test has_consistent_initial_distribution(momdp)
    
    momdp = TagMOMDPAT(; num_asks=-1, types=[1.5])
    
    states_vec_x = ordered_states_x(momdp)
    states_vec_y = ordered_states_y(momdp)
    @test length(states_vec_x) == num_grid_pos + 1
    @test length(states_vec_y) == num_grid_pos

    @test stateindex_x(momdp, TagMOMDPXState(0, 1)) == 1
    @test stateindex_x(momdp, TagMOMDPXState(5, 1)) == 6
    @test_throws AssertionError stateindex_x(momdp, TagMOMDPXState(5, 0))

    @test stateindex_y(momdp, TagMOMDPYState(1, 1.5)) == 1
    @test stateindex_y(momdp, TagMOMDPYState(6, 1.5)) == 6
    @test_throws AssertionError stateindex_y(momdp, TagMOMDPYState(6, 2.5))
    
    xid = initialstate_x(momdp)
    yid = initialstate_y(momdp, TagMOMDPXState(1, 0))
    @test xid isa SparseCat
    @test yid isa SparseCat
    @test length(xid.probs) == num_grid_pos
    @test length(yid.probs) == num_grid_pos
    @test isapprox(xid.probs[1], 1/num_grid_pos; atol=1e-6)
    @test isapprox(yid.probs[1], 1/num_grid_pos; atol=1e-6)
    @test isapprox(sum(xid.probs), 1.0; atol=1e-6)
    @test isapprox(sum(yid.probs), 1.0; atol=1e-6)
    @test has_consistent_initial_distribution(momdp)
    
    momdp = TagMOMDPAT(; num_asks=3, types=[1.0, 5.0])
    
    states_vec_x = ordered_states_x(momdp)
    states_vec_y = ordered_states_y(momdp)
    @test length(states_vec_x) == (num_grid_pos + 1) * 4
    @test length(states_vec_y) == num_grid_pos * 2
    
    @test stateindex_x(momdp, TagMOMDPXState(0, 0)) == 1
    @test stateindex_x(momdp, TagMOMDPXState(5, 1)) == num_grid_pos + 1 + 6
    @test_throws AssertionError stateindex_x(momdp, TagMOMDPXState(5, -1))

    @test stateindex_y(momdp, TagMOMDPYState(1, 1.0)) == 1
    @test stateindex_y(momdp, TagMOMDPYState(6, 1.0)) == 6
    @test stateindex_y(momdp, TagMOMDPYState(6, 5.0)) == num_grid_pos + 6
    @test_throws AssertionError stateindex_y(momdp, TagMOMDPYState(6, 2.5))
    
    xid = initialstate_x(momdp)
    yid = initialstate_y(momdp, TagMOMDPXState(1, 0))
    @test xid isa SparseCat
    @test yid isa SparseCat
    @test length(xid.probs) == num_grid_pos
    @test length(yid.probs) == num_grid_pos * length(momdp.types)
    @test isapprox(xid.probs[1], 1/num_grid_pos; atol=1e-6)
    @test isapprox(yid.probs[1], 1/(num_grid_pos * length(momdp.types)); atol=1e-6)
    @test isapprox(sum(xid.probs), 1.0; atol=1e-6)
    @test isapprox(sum(yid.probs), 1.0; atol=1e-6)
    @test has_consistent_initial_distribution(momdp)
    
    
    state_vec = ordered_states(momdp)
    @test length(states(momdp)) == (num_grid_pos + 1) * 4 * num_grid_pos * 2
    @test stateindex(momdp, state_vec[11]) == 11
    @test stateindex(momdp, state_vec[end]) == length(states(momdp))
    
    
    map_str = """
    xxxxxxxxxx
    xoooooooox
    xoxoxxxxox
    xoxoxxxxox
    xoxooooxox
    xoxoxxoxox
    xoxoxxoxox
    xoxoxxoxox
    xoooooooox
    xxxxxxxxxx
    """
    momdp = TagMOMDPAT(; map_str=map_str, num_asks=0)
    @test get_prop(momdp.mg, 1, 2, :action) == :east
    @test get_prop(momdp.mg, 1, 9, :action) == :south
    @test get_prop(momdp.mg, 17, 16, :action) == :west
    @test get_prop(momdp.mg, 20, 14, :action) == :north
end
