function MOMDPs.transition_x(problem::RockSampleMOMDPAT{K}, s::RockSampleMOMDPState{K}, a::Int) where {K}
    x_state = s[1]
    if x_state.pos == (-1, -1)
        # map_states = vec([(i, j) for i in 1:problem.map_size[1], j in 1:problem.map_size[2]])
        # statesX = [RockSampleMOMDPXState(pos, x_state.num_ask_remain) for pos in map_states]
        # probs = normalize(ones(length(statesX)), 1)
        # return SparseCat(statesX, probs)
        return SparseCat([x_state], [1.0])
    end
    
    new_pos = next_position(x_state.pos, a)
    
    if new_pos[1] > problem.map_size[1]
        new_pos = (-1, -1)
        new_asks = x_state.num_ask_remain
    else
        new_pos = (clamp(new_pos[1], 1, problem.map_size[1]),
                         clamp(new_pos[2], 1, problem.map_size[2]))
    
        ask_avail = problem.num_asks != 0
        if ask_avail && a == BASIC_ACTIONS_DICT[:ask]
            if problem.num_asks == -1
                new_asks = 1
            else
                new_asks = max(0, x_state.num_ask_remain - 1)
            end
        else
            new_asks = x_state.num_ask_remain
        end
    end
    
    new_x_state = RockSampleMOMDPXState(new_pos, new_asks)
    return SparseCat([new_x_state], [1.0])
end

function MOMDPs.transition_y(problem::RockSampleMOMDPAT{K}, s::RockSampleMOMDPState{K}, a::Int, x_prime::RockSampleMOMDPXState) where {K}
    x_state = s[1]
    y_state = s[2]
    if x_state.pos == (-1, -1)
        # rock_states = Vector{RSRocksAT{K}}(undef, 2^K)
        # for (i,rocks) in enumerate(Iterators.product(ntuple(x->[false, true], K)...))
        #     rock_states[i] = Tuple(rocks)
        # end
        
        # states_y_vec = [RockSampleMOMDPYState{K}(rock_state, s[2].sugg_type) for rock_state in rock_states]
        # probs = normalize(ones(2^K), 1)
        # return SparseCat(states_y_vec, probs)
        return SparseCat([y_state], [1.0])
    end

    if x_prime.pos == (-1, -1) # The robot exited, so just keep the rocks the same
        return SparseCat([y_state], [1.0])
    end
    
    if a == BASIC_ACTIONS_DICT[:sample] && in(x_state.pos, problem.rocks_positions)
        # Sample action: rock becomes false
        rock_ind = findfirst(isequal(x_state.pos), problem.rocks_positions)
        new_rocks = Tuple([r == rock_ind ? false : y_state.rocks[r] for r in 1:K])
    else
        # No sampling, so rock states remain unchanged
        new_rocks = y_state.rocks
    end
    
    num_new_states = length(problem.types)
    new_probs = zeros(num_new_states)
    new_states = Vector{RockSampleMOMDPYState{K}}(undef, num_new_states)
    
    for (ii, type_i) in enumerate(problem.types)
        if type_i == y_state.sugg_type
            p = 1 - problem.type_trans
        else
            p = problem.type_trans / (length(problem.types) - 1)
        end
        new_probs[ii] = p
        new_states[ii] = RockSampleMOMDPYState{K}(new_rocks, type_i)
    end
    normalize!(new_probs, 1)
    nz_probs = findall(new_probs .> 0)
    return SparseCat(new_states[nz_probs], new_probs[nz_probs])
    
end

# New AT version helper function
function next_position(x::RSPosAT, a::Int)
    if a >= N_BASIC_ACTIONS # We are either sensing or asking
        return x
    end
    return x .+ ACTION_DIRS_AT[a]
end
