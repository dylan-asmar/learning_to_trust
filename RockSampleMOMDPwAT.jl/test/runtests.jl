using POMDPs
using POMDPTools
using MOMDPs
using Random
using Test
using RockSampleMOMDPProblemAT
using Compose

@testset "RockSampleMOMDPProblemAT" begin    
    @testset "constructor" begin
        @test RockSampleMOMDPAT() isa RockSampleMOMDPAT
        @test RockSampleMOMDPAT(rocks_positions=[(1,1),(2,2)]) isa RockSampleMOMDPAT{2}
        @test RockSampleMOMDPAT(; map_size=(11,5), rocks_positions=[(1,2), (2,4), (11,5)]) isa RockSampleMOMDPAT{3}
        
        @test !is_y_prime_dependent_on_x_prime(RockSampleMOMDPAT())
        @test !is_x_prime_dependent_on_y(RockSampleMOMDPAT())
        @test is_initial_distribution_independent(RockSampleMOMDPAT())
    end
    @testset "state space" begin 
        momdp = RockSampleMOMDPAT(; map_size=(3, 3))
        statesx = states_x(momdp) 
        sox = ordered_states_x(momdp)
        @test length(sox) == 10
        @test length(statesx) == 10
        
        sy = states_y(momdp)
        soy = ordered_states_y(momdp)
        @test length(soy) == 2^3
        @test length(sy) == 2^3
        
        momdp = RockSampleMOMDPAT(
            map_size=(7, 10), 
            rocks_positions=[(1,1), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8)]
        )
        statesx = states_x(momdp)
        sox = ordered_states_x(momdp)
        @test length(sox) == 71
        @test length(statesx) == 71
        
        sy = states_y(momdp)
        soy = ordered_states_y(momdp)
        @test length(soy) == 2^7
        @test length(sy) == 2^7
        
        momdp = RockSampleMOMDPAT(; init_pos=(1,1), rocks_positions=[(1,1), (3,3), (4,4)])
        init_state_d = initialstate(momdp)
        @test isa(init_state_d, SparseCat)
        @test length(init_state_d.vals) == (length(states_x(momdp)) - 1) * length(states_y(momdp))
        @test all(init_state_d.probs .== 1 / length(init_state_d.vals))
        
        @test has_consistent_initial_distribution(momdp)
        
        momdp = RockSampleMOMDPAT(; map_size=(3,3), num_asks=-1, types=[1.0, 2.0], init_type_dist=[0.1, 0.9])
        @test length(states_x(momdp)) == 10
        @test length(states_y(momdp)) == 2^3 * 2
        @test has_consistent_initial_distribution(momdp)
        
        momdp = RockSampleMOMDPAT(; map_size=(3,3), num_asks=2, types=[1.0, 2.0], init_type_dist=[0.1, 0.9])
        @test length(states_x(momdp)) == 10*3
        @test length(states_y(momdp)) == 2^3 * 2
        @test has_consistent_initial_distribution(momdp)
        
        @test !isterminal(momdp, (RockSampleMOMDPXState((3,3), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0)))
        @test isterminal(momdp, (RockSampleMOMDPXState((-1, -1), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0)))
    end

    @testset "action space" begin 
        momdp = RockSampleMOMDPAT(
            map_size=(5, 5), 
            rocks_positions=[(1,1), (3,3), (4,4)]
        )
        acts = actions(momdp)
        @test length(acts) == 5 + 3
        
        s = (RockSampleMOMDPXState((3,1), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
        @test length(actions(momdp, s)) == length(actions(momdp)) - 1
        s2 = (RockSampleMOMDPXState((3,3), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
        @test actions(momdp, s2) == actions(momdp)
        @test actionindex(momdp, 1) == 1
        @test actionindex(momdp, length(actions(momdp))) == length(actions(momdp))
        
        
        momdp = RockSampleMOMDPAT(
            map_size=(5, 5), 
            rocks_positions=[(1,1), (3,3), (4,4)],
            num_asks=-1
        )
        acts = actions(momdp)
        @test length(acts) == 5 + 3 + 1
        
        s = (RockSampleMOMDPXState((3,1), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
        @test length(actions(momdp, s)) == length(actions(momdp)) - 1
        s2 = (RockSampleMOMDPXState((3,3), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
        @test actions(momdp, s2) == actions(momdp)
        @test actionindex(momdp, 1) == 1
        @test actionindex(momdp, length(actions(momdp))) == length(actions(momdp))
        
        momdp = RockSampleMOMDPAT(
            map_size=(5, 5), 
            rocks_positions=[(1,1), (3,3), (4,4)],
            num_asks=3
        )
        acts = actions(momdp)
        @test length(acts) == 5 + 3 + 1
        
        s = (RockSampleMOMDPXState((3,1), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
        @test length(actions(momdp, s)) == length(actions(momdp)) - 1
        s2 = (RockSampleMOMDPXState((3,3), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
        @test actions(momdp, s2) == actions(momdp)
        @test actionindex(momdp, 1) == 1
        @test actionindex(momdp, length(actions(momdp))) == length(actions(momdp))
    end

    @testset "transition" begin        
        @testset "transition_x" begin        
            momdp = RockSampleMOMDPAT(rocks_positions=[(1,1), (3,3), (4,4)])
        
            s0 = (RockSampleMOMDPXState((1, 1), 0), RockSampleMOMDPYState{3}((true, false, true), 1.0))
            s1 = (RockSampleMOMDPXState((2, 2), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            s2 = (RockSampleMOMDPXState((5, 4), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            
            @inferred transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample])
            @inferred transition_x(momdp, s2, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east])
            
            # North
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 2), 0)
            
            # East
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 1), 0)
            
            d = transition_x(momdp, s2, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((-1, -1), 0)            
            
            # South
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:south])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 0)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:south])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 1), 0)
            
            # West
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:west])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 0)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:west])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 2), 0)
            
            # sample
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 0)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 2), 0)
            
            # sense
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.N_BASIC_ACTIONS + 1)
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 0)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.N_BASIC_ACTIONS + 2)
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 2), 0)
            
            
            momdp = RockSampleMOMDPAT(rocks_positions=[(1,1), (3,3), (4,4)], num_asks=-1)
            s0 = (RockSampleMOMDPXState((1, 1), 1), RockSampleMOMDPYState{3}((true, false, true), 1.0))
            s1 = (RockSampleMOMDPXState((2, 2), 1), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            s2 = (RockSampleMOMDPXState((5, 4), 1), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            
            @inferred transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample])
            @inferred transition_x(momdp, s2, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east])
            
            # North
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 2), 1)
            
            # East
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 1), 1)
            
            d = transition_x(momdp, s2, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((-1, -1), 1)            
            
            # South
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:south])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 1)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:south])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 1), 1)
            
            # West
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:west])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 1)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:west])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 2), 1)
            
            # sample
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 1)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 2), 1)
            
            # sense
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.N_BASIC_ACTIONS + 1)
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 1)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.N_BASIC_ACTIONS + 2)
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 2), 1)
            
            # ask
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:ask])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 1)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:ask])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 2), 1)
            
            @inferred transition_x(momdp, s0, 6)
            
            momdp = RockSampleMOMDPAT(rocks_positions=[(1,1), (3,3), (4,4)], num_asks=2)
            s3 = (RockSampleMOMDPXState((3, 3), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            
            # North
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 2), 1)
            
            # East
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 1), 1)
            
            d = transition_x(momdp, s2, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((-1, -1), 1)            
            
            # South
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:south])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 1)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:south])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 1), 1)
            
            # West
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:west])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 1)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:west])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 2), 1)
            
            # sample
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 1)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 2), 1)
            
            # sense
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.N_BASIC_ACTIONS + 1)
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 1)
            
            d = transition_x(momdp, s1, RockSampleMOMDPProblemAT.N_BASIC_ACTIONS + 2)
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((2, 2), 1)
            
            # ask
            d = transition_x(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:ask])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((1, 1), 0)
            
            d = transition_x(momdp, s3, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:ask])
            @test length(d.vals) == 1
            @test d.vals[1] == RockSampleMOMDPXState((3, 3), 0)
            
            @inferred transition_x(momdp, s0, 6)
            @inferred transition_x(momdp, s3, 6)
            
            s5 = (RockSampleMOMDPXState((-1, -1), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            d = transition_x(momdp, s5, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north])
            # @test length(d.vals) == momdp.map_size[1] * momdp.map_size[2]
            # @test isapprox(d.probs[1], 1 / length(d.vals))
            # @test d.vals[1].num_ask_remain == 0
            @test length(d.vals) == 1
            @test d.vals[1] == s5[1]            
            
            s5 = (RockSampleMOMDPXState((-1, -1), 1), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            d = transition_x(momdp, s5, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:west])
            # @test length(d.vals) == momdp.map_size[1] * momdp.map_size[2]
            # @test isapprox(d.probs[1], 1 / length(d.vals))
            # @test d.vals[1].num_ask_remain == 1
            @test length(d.vals) == 1
            @test d.vals[1] == s5[1]
            
            @test has_consistent_transition_distributions(momdp)
        end
        
        @testset "transition_y" begin
            momdp = RockSampleMOMDPAT(rocks_positions=[(1,1), (3,3), (4,4)], num_asks=2)
            s0 = (RockSampleMOMDPXState((1, 1), 1), RockSampleMOMDPYState{3}((true, false, true), 1.0))
            s1 = (RockSampleMOMDPXState((2, 2), 1), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            s2 = (RockSampleMOMDPXState((5, 4), 1), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            s3 = (RockSampleMOMDPXState((3, 3), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            
            @inferred transition_y(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north], RockSampleMOMDPXState((1, 2), 0))
            
            # Moving or sensing, rocks wouldn't change
            d = transition_y(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north], RockSampleMOMDPXState((1, 2), 0))
            @test d.vals[1] == s0[2]
            
            d = transition_y(momdp, s0, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:south], RockSampleMOMDPXState((1, 1), 0))
            @test d.vals[1] == s0[2]
            
            d = transition_y(momdp, s0, 6, s0[1])
            @test d.vals[1] == s0[2]
            
            # Sampling
            d = transition_y(momdp, s2, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample], s2[1])
            @test d.vals[1] == s2[2]
            
            d = transition_y(momdp, s3, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample], s3[1])
            rock_ind = findfirst(isequal(s3[1].pos), momdp.rocks_positions)
            for i in 1:length(d.vals[1].rocks)
                if i == rock_ind
                    @test d.vals[1].rocks[i] == false
                else
                    @test d.vals[1].rocks[i] == s3[2].rocks[i]
                end
            end
            
            s4 = (RockSampleMOMDPXState((4, 4), 1), RockSampleMOMDPYState{3}((false, false, false), 1.0))
            d = transition_y(momdp, s4, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample], s4[1])
            @test d.vals[1] == s4[2]
            
            momdp = RockSampleMOMDPAT(rocks_positions=[(1,1), (3,3), (4,4)], num_asks=2, type_trans=0.1, types=[1.0, 2.0])
            d = transition_y(momdp, s4, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample], s4[1])
            @test length(d.vals) == 2
            @test pdf(d, RockSampleMOMDPYState{3}((false, false, false), 1.0)) == 0.9
            @test pdf(d, RockSampleMOMDPYState{3}((false, false, false), 2.0)) == 0.1
            
            @inferred transition_y(momdp, s4, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north], RockSampleMOMDPXState((1, 2), 0))
            @inferred transition_y(momdp, s4, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample], RockSampleMOMDPXState((1, 1), 0))     
            
            s5 = (RockSampleMOMDPXState((-1, -1), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
            d = transition_y(momdp, s5, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north], RockSampleMOMDPXState((1, 2), 0))
            # @test length(d.vals) == 2^3
            # @test isapprox(d.probs[1], 1 / length(d.vals))
            # @test d.vals[1].sugg_type == 1.0
            # @test d.vals[2].sugg_type == 1.0
            @test length(d.vals) == 1
            @test d.vals[1] == s5[2]
            
            @test has_consistent_transition_distributions(momdp)
        end 
    end

    @testset "observation" begin 
        momdp = RockSampleMOMDPAT(rocks_positions=[(1,1), (3,3), (4,4)])
        @test has_consistent_observation_distributions(momdp)
        
        s0 = (RockSampleMOMDPXState((2, 2), 0), RockSampleMOMDPYState{3}((true, true, false), 1.0))
        
        @inferred observation(momdp, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north], s0)
        @inferred observation(momdp, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample], s0)
        
        momdp = RockSampleMOMDPAT()
        obs = observations(momdp)
        @test length(obs) == 3
        
        od = observation(momdp, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample], s0)
        @test od isa SparseCat
        @test od.vals[1] == 3
        @test od.probs[1] == 1.0
        
        od = observation(momdp, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north], s0)
        @test od isa SparseCat
        @test od.vals[1] == 3
        @test od.probs[1] == 1.0
        
        od = observation(momdp, RockSampleMOMDPProblemAT.N_BASIC_ACTIONS - 1 + 1, s0)
        probs_1 = pdf(od, 1)
        probs_2 = pdf(od, 2)
        @test probs_1 > probs_2
        @test length(od.vals) == 2
        
        od = observation(momdp, RockSampleMOMDPProblemAT.N_BASIC_ACTIONS - 1 + 2, s0)
        probs_1 = pdf(od, 1)
        probs_2 = pdf(od, 2)
        @test probs_1 > probs_2
        @test length(od.vals) == 2
        
        momdp = RockSampleMOMDPAT(; num_asks=2)
        
        obs = observations(momdp)
        @test length(obs) == 3 + RockSampleMOMDPProblemAT.N_BASIC_ACTIONS + 3 - 1
        
        od = observation(momdp, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample], s0)
        @test od isa SparseCat
        @test od.vals[1] == 3
        @test od.probs[1] == 1.0
        
        od = observation(momdp, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north], s0)
        @test od isa SparseCat
        @test od.vals[1] == 3
        @test od.probs[1] == 1.0
        
        od = observation(momdp, RockSampleMOMDPProblemAT.N_BASIC_ACTIONS + 1, s0)
        probs_1 = pdf(od, 1)
        probs_2 = pdf(od, 2)
        @test probs_1 > probs_2
        @test length(od.vals) == 2
        
        od = observation(momdp, RockSampleMOMDPProblemAT.N_BASIC_ACTIONS + 2, s0)
        probs_1 = pdf(od, 1)
        probs_2 = pdf(od, 2)
        @test probs_1 > probs_2
        @test length(od.vals) == 2
        
        
        od = observation(momdp, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:ask], s0)
        @test length(od.vals) == length(actions(momdp)) - 1
        
        @test has_consistent_observation_distributions(momdp)
    end

    @testset "reward" begin
        momdp = RockSampleMOMDPAT(rocks_positions=[(1,1), (3,3), (4,4)])
        s = (RockSampleMOMDPXState((1, 1), 0), RockSampleMOMDPYState{3}((false, true, false), 1.0))
        
        @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample]) == momdp.bad_rock_penalty
        @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample], s) == momdp.bad_rock_penalty
        @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north]) == momdp.step_penalty
        # @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:ask]) == momdp.ask_cost
        
        s = (RockSampleMOMDPXState((3, 3), 0), s[2])
        @test reward(momdp, s, 1) == momdp.good_rock_reward
        @test reward(momdp, s, 3) == momdp.step_penalty
        @test reward(momdp, s, 6) == momdp.sensor_use_penalty
        @test reward(momdp, s, 6) == 0.
        @test reward(momdp, s, 2) == 0.
        
        s = (RockSampleMOMDPXState((5, 4), 0), s[2])
        
        xp = transition_x(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east]).vals[1]
        yp = transition_y(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east], xp).vals[1]
        sp = (xp, yp)
        @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east]) == momdp.exit_reward
        @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:east], sp) == momdp.exit_reward
        
        momdp = RockSampleMOMDPAT(rocks_positions=[(1,1), (3,3), (4,4)], step_penalty=-1., sensor_use_penalty=-5.)
        s = (RockSampleMOMDPXState((1, 1), 0), RockSampleMOMDPYState{3}((false, true, false), 1.0))
        @test reward(momdp, s, 2) == momdp.step_penalty
        @test reward(momdp, s, 6) == momdp.sensor_use_penalty + momdp.step_penalty
        @test reward(momdp, s, 1) == momdp.bad_rock_penalty + momdp.step_penalty

        s = (RockSampleMOMDPXState((3, 3), 0), s[2])
        @test reward(momdp, s, 1) == momdp.good_rock_reward + momdp.step_penalty
        
        
        momdp = RockSampleMOMDPAT(rocks_positions=[(1,1), (3,3), (4,4)], num_asks=2, type_trans=0.1, types=[1.0, 2.0])
        s = (RockSampleMOMDPXState((1, 1), 1), RockSampleMOMDPYState{3}((false, true, false), 1.0))
        
        @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample]) == momdp.bad_rock_penalty
        @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:sample], s) == momdp.bad_rock_penalty
        @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:north]) == momdp.step_penalty
        @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:ask]) == momdp.ask_cost
        
        s = (RockSampleMOMDPXState((1, 1), 0), RockSampleMOMDPYState{3}((false, true, false), 1.0))
        @test reward(momdp, s, RockSampleMOMDPProblemAT.BASIC_ACTIONS_DICT[:ask]) == momdp.bad_rock_penalty
        
        s = (RockSampleMOMDPXState((3, 3), 0), s[2])
        @test reward(momdp, s, 1) == momdp.good_rock_reward
        @test reward(momdp, s, 3) == momdp.step_penalty
        @test reward(momdp, s, 7) == momdp.sensor_use_penalty
        @test reward(momdp, s, 7) == 0.
        @test reward(momdp, s, 2) == 0.
    end

    @testset "rendering" begin 
        momdp = RockSampleMOMDPAT(rocks_positions=[(1,1), (3,3), (4,4)])
        s0 = (RockSampleMOMDPXState((1, 1), 0), RockSampleMOMDPYState{3}((true, false, true), 1.0))
        render(momdp, (s=s0, a=3))
        b0 = initialstate(momdp)
        render(momdp, (s=s0, a=3, b=b0))
        render(momdp, (s=s0, a=3, b=b0), pre_act_text="t=4, ")
        render(momdp, (s=s0, a=3, b=b0), pre_act_text="t=4, ", viz_belief=false)
        render(momdp, (s=s0, a=3, b=b0), pre_act_text="t=4, ", viz_types=false)
        render(momdp, (s=s0, a=3, b=b0), pre_act_text="t=4, ", viz_rock_state=false)
        plt = render(momdp, (s=s0, a=7, b=b0))
        plt |> SVG("rocksample.svg")
        @test isfile("rocksample.svg")
        isfile("rocksample.svg") && rm("rocksample.svg")
    end
end
