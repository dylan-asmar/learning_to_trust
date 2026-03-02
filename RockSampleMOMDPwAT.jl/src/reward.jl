function POMDPs.reward(problem::RockSampleMOMDPAT{K}, s::RockSampleMOMDPState{K}, a::Int) where {K}
    r = problem.step_penalty
    
    n_basic_actions_mod = problem.num_asks == 0 ? N_BASIC_ACTIONS - 1 : N_BASIC_ACTIONS
    
    if problem.num_asks != 0 && a == BASIC_ACTIONS_DICT[:ask]
        if s[1].num_ask_remain > 0
            r += problem.ask_cost
        else
            r += problem.bad_rock_penalty
        end
        return r
    end
    
    if next_position(s[1].pos, a)[1] > problem.map_size[1]
        r += problem.exit_reward
        return r
    end

    if a == BASIC_ACTIONS_DICT[:sample] && in(s[1].pos, problem.rocks_positions) # sample 
        rock_ind = findfirst(isequal(s[1].pos), problem.rocks_positions)
        r += s[2].rocks[rock_ind] ? problem.good_rock_reward : problem.bad_rock_penalty
    elseif a > n_basic_actions_mod # using sensor
        r += problem.sensor_use_penalty
    end
    return r
end
