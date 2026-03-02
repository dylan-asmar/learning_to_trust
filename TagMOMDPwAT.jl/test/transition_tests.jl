@testset "transition" begin
    momdp = TagMOMDPAT()
    @inferred transition_x(momdp, (TagMOMDPXState(1, 0), TagMOMDPYState(2, 1.0)), 1)
    @inferred transition_y(momdp, (TagMOMDPXState(1, 0), TagMOMDPYState(2, 1.0)), 1, TagMOMDPXState(1, 0))
    @inferred transition(momdp, (TagMOMDPXState(1, 0), TagMOMDPYState(2, 1.0)), 1)
    @test_throws AssertionError transition(momdp, (TagMOMDPXState(1, 0), TagMOMDPYState(2, 1.0)), 6)
    @test has_consistent_transition_distributions(momdp)
    
    map_str = """
    ooo
    ooo
    ooo
    """
    
    momdp = TagMOMDPAT(; map_str=map_str, num_asks=1, types=[1.0, 2.0])
    @inferred transition_x(momdp, (TagMOMDPXState(1, 1), TagMOMDPYState(2, 1.0)), 6)
    @inferred transition_y(momdp, (TagMOMDPXState(1, 1), TagMOMDPYState(2, 2.0)), 6, TagMOMDPXState(1, 0))
    @inferred transition(momdp, (TagMOMDPXState(1, 0), TagMOMDPYState(2, 1.0)), 1)
    @test has_consistent_transition_distributions(momdp)
    
    momdp = TagMOMDPAT(; map_str=map_str, transition_option=:invalid)
    @test_throws ErrorException transition(momdp, (TagMOMDPXState(1, 0), TagMOMDPYState(2, 1.0)), 1)

    # Test transition function with the default transition option (modified)
    momdp = TagMOMDPAT(; map_str=map_str, transition_option=:modified)

    # In in terminal state, should return initial state
    td = transition(momdp, (TagMOMDPXState(0, 0), TagMOMDPYState(1, 1.0)), 1)
    @test isa(td, SparseCat)
    @test !isterminal(momdp, td.vals[1])
    @test length(td.vals) == get_prop(momdp.mg, :num_grid_pos) * get_prop(momdp.mg, :num_grid_pos)
    @test isapprox(sum(td.probs), 1.0)
    @test isapprox(td.probs[1], 1.0 / (get_prop(momdp.mg, :num_grid_pos) * get_prop(momdp.mg, :num_grid_pos)))
    @test isapprox(td.probs[end], 1.0 / (get_prop(momdp.mg, :num_grid_pos) * get_prop(momdp.mg, :num_grid_pos)))

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 1)
    @test all([s[1].r_pos == 2 for s in td.vals])

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 2)
    @test all([s[1].r_pos == 6 for s in td.vals])

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 3)
    @test all([s[1].r_pos == 8 for s in td.vals])

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 4)
    @test all([s[1].r_pos == 4 for s in td.vals])

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 5)
    @test isa(td, SparseCat)
    @test td.vals[1][1].r_pos == 0
    
    @test_throws AssertionError transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 6)

    td = transition(momdp, (TagMOMDPXState(2, 0), TagMOMDPYState(5, 1.0)), 2)
    @test length(td.vals) == 4
    @test sum(td.probs) == 1.0
    @test all([s[1].r_pos == 3 for s in td.vals])
    for (ii, s) in enumerate(td.vals)
        if s[2].t_pos != 5
            @test isapprox(td.probs[ii], momdp.move_away_probability/3; atol=1e-6)
        else
            @test isapprox(td.probs[ii], 1.0 - momdp.move_away_probability; atol=1e-6)
        end
    end

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(6, 1.0)), 5)
    @test all([s[1].r_pos == 5 for s in td.vals])
    @test length(td.vals) == 3
    @test sum(td.probs) == 1.0
    for (ii, s) in enumerate(td.vals)
        if s[2].t_pos != 6
            @test isapprox(td.probs[ii], momdp.move_away_probability/2; atol=1e-6)
        else
            @test isapprox(td.probs[ii], 1.0 - momdp.move_away_probability; atol=1e-6)
        end
    end

    td = transition(momdp, (TagMOMDPXState(7, 0), TagMOMDPYState(3, 1.0)), 1)
    @test isa(td, SparseCat)
    @test td.vals[1][1].r_pos == 4
    @test td.vals[1][2].t_pos == 3

    # Test original paper transition function
    momdp = TagMOMDPAT(; map_str=map_str, transition_option=:orig)

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 1)
    @test all([s[1].r_pos == 2 for s in td.vals])

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 2)
    @test all([s[1].r_pos == 6 for s in td.vals])

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 3)
    @test all([s[1].r_pos == 8 for s in td.vals])

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 4)
    @test all([s[1].r_pos == 4 for s in td.vals])

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(5, 1.0)), 5)
    @test isa(td, SparseCat)
    @test td.vals[1][1].r_pos == 0

    td = transition(momdp, (TagMOMDPXState(2, 0), TagMOMDPYState(5, 1.0)), 2)
    @test length(td.vals) == 4
    @test sum(td.probs) == 1.0
    @test all([s[1].r_pos == 3 for s in td.vals])
    for (ii, s) in enumerate(td.vals)
        if s[2].t_pos != 5
            if s[2].t_pos == 8
                @test isapprox(td.probs[ii], momdp.move_away_probability/2; atol=1e-6)
            else
                @test isapprox(td.probs[ii], momdp.move_away_probability/2/2; atol=1e-6)
            end
        else
            @test isapprox(td.probs[ii], 1.0 - momdp.move_away_probability; atol=1e-6)
        end
    end

    td = transition(momdp, (TagMOMDPXState(5, 0), TagMOMDPYState(6, 1.0)), 5)
    @test all([s[1].r_pos == 5 for s in td.vals])
    @test length(td.vals) == 3
    @test sum(td.probs) == 1.0
    for (ii, s) in enumerate(td.vals)
        if s[2].t_pos != 6
            @test isapprox(td.probs[ii], momdp.move_away_probability/2/2; atol=1e-6)
        else
            @test isapprox(td.probs[ii], 1.0 - momdp.move_away_probability/2; atol=1e-6)
        end
    end

    map_str = """
    ooo
    ooo
    ooo
    """
    momdp = TagMOMDPAT(; map_str=map_str, num_asks=2, types=[1.0, 2.0])

    xd = transition_x(momdp, (TagMOMDPXState(1, 2), TagMOMDPYState(2, 2.0)), 4)
    @test length(xd.vals) == 1
    @test all([x.r_pos == 1 for x in xd.vals])
    @test all([x.num_ask_remain == 2 for x in xd.vals])
    
    xd = transition_x(momdp, (TagMOMDPXState(1, 2), TagMOMDPYState(2, 2.0)), 6)
    @test length(xd.vals) == 1
    @test all([x.r_pos == 1 for x in xd.vals])
    @test all([x.num_ask_remain == 1 for x in xd.vals])
    
    xd = transition_x(momdp, (TagMOMDPXState(1, 0), TagMOMDPYState(2, 2.0)), 6)
    @test length(xd.vals) == 1
    @test all([x.r_pos == 1 for x in xd.vals])
    @test all([x.num_ask_remain == 0 for x in xd.vals])
    
    xd = transition_x(momdp, (TagMOMDPXState(5, 2), TagMOMDPYState(5, 2.0)), 5)
    @test length(xd.vals) == 1
    @test all([x.r_pos == 0 for x in xd.vals])
    @test all([x.num_ask_remain == 2 for x in xd.vals])

    xd = transition_x(momdp, (TagMOMDPXState(0, 1), TagMOMDPYState(2, 2.0)), 1)
    @test length(xd.vals) == get_prop(momdp.mg, :num_grid_pos)
    @test isapprox(xd.probs[1], 1.0 / get_prop(momdp.mg, :num_grid_pos))
    @test all([x.num_ask_remain == 2 for x in xd.vals])
    
    yd = transition_y(momdp, (TagMOMDPXState(1, 2), TagMOMDPYState(2, 2.0)), 4, TagMOMDPXState(1, 1))
    @test all([y.sugg_type == 2.0 for y in yd.vals])
    
    yd = transition_y(momdp, (TagMOMDPXState(2, 2), TagMOMDPYState(2, 2.0)), 5, TagMOMDPXState(0, 2))
    @test length(yd.vals) == 1
    @test yd.vals[1].t_pos == 2
    @test yd.vals[1].sugg_type == 2.0
    
    yd = transition_y(momdp, (TagMOMDPXState(0, 2), TagMOMDPYState(2, 2.0)), 3, TagMOMDPXState(3, 2))
    @test length(yd.vals) == get_prop(momdp.mg, :num_grid_pos)
    @test isapprox(yd.probs[1], 1.0 / get_prop(momdp.mg, :num_grid_pos))
    @test all([y.sugg_type == 2.0 for y in yd.vals])
    
    momdp = TagMOMDPAT(; num_asks=-1, types=[1.0, 2.0])
    xd = transition_x(momdp, (TagMOMDPXState(1, 1), TagMOMDPYState(2, 2.0)), 6)
    @test length(xd.vals) == 1
    @test all([x.r_pos == 1 for x in xd.vals])
    @test all([x.num_ask_remain == 1 for x in xd.vals])
    
    xd = transition_x(momdp, (TagMOMDPXState(0, 1), TagMOMDPYState(2, 2.0)), 1)
    @test length(xd.vals) == get_prop(momdp.mg, :num_grid_pos)
    @test isapprox(xd.probs[1], 1.0 / get_prop(momdp.mg, :num_grid_pos))
    @test all([x.num_ask_remain == 1 for x in xd.vals])
    
    momdp = TagMOMDPAT(; num_asks=0, types=[1.0, 2.0, 3.0], type_trans=0.2)
    yd = transition_y(momdp, (TagMOMDPXState(1, 0), TagMOMDPYState(2, 2.0)), 4, TagMOMDPXState(1, 0))
    p_sugg_type = zeros(length(momdp.types))
    for (yi, pyi) in zip(yd.vals, yd.probs)
        type_i = findfirst(isequal(yi.sugg_type), momdp.types)
        p_sugg_type[type_i] += pyi
    end
    p_sugg_type = p_sugg_type ./ sum(p_sugg_type)
    @test isapprox(p_sugg_type[1], 0.1, atol=1e-6)
    @test isapprox(p_sugg_type[2], 0.8, atol=1e-6)
    @test isapprox(p_sugg_type[3], 0.1, atol=1e-6)
    
    @test has_consistent_transition_distributions(momdp)
end
