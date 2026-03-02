
function MOMDPs.states_x(problem::RockSampleMOMDPAT{K}) where {K}
    map_states = vec([(i, j) for i in 1:problem.map_size[1], j in 1:problem.map_size[2]])
    push!(map_states, (-1, -1)) # Terminal state
    
    if problem.num_asks == -1
        num_asks_vec = [1]
    else
        num_asks_vec = 0:problem.num_asks
    end
    return vec([RockSampleMOMDPXState(pos, asks) for pos in map_states, asks in num_asks_vec])
end

function MOMDPs.states_y(problem::RockSampleMOMDPAT{K}) where {K}
    bool_options = [[true, false] for _ in 1:K]
    vec_bool_options = vec(collect(Iterators.product(bool_options...)))
    ntup_vec_bool_options = [RSRocksAT{K}(bool_vec) for bool_vec in vec_bool_options]
    return vec([RockSampleMOMDPYState{K}(rocks, sugg_type) for rocks in ntup_vec_bool_options, sugg_type in problem.types])
end

function MOMDPs.stateindex_x(problem::RockSampleMOMDPAT{K}, x::RockSampleMOMDPXState) where {K}
    temp_rocks = Tuple(trues(K))
    st = (x, RockSampleMOMDPYState{K}(temp_rocks, problem.types[1]))
    return MOMDPs.stateindex_x(problem, st)
end

function MOMDPs.stateindex_x(problem::RockSampleMOMDPAT{K}, s::RockSampleMOMDPState{K}) where {K}
    num_map_states = problem.map_size[1] * problem.map_size[2] + 1  # +1 for terminal
    if s[1].pos == (-1, -1)
        pos_idx = num_map_states
    else
        pos_idx = LinearIndices(problem.map_size)[s[1].pos[1], s[1].pos[2]]
    end
    
    if problem.num_asks == 0
        @assert s[1].num_ask_remain == 0 "Invalid number of asks for x state $(s[1])"
        return pos_idx
    elseif problem.num_asks == -1
        @assert s[1].num_ask_remain == 1 "Invalid number of asks for x state $(s[1])"
        return pos_idx
    else
        @assert s[1].num_ask_remain >= 0 && s[1].num_ask_remain <= problem.num_asks "Invalid number of asks for x state $(s[1])"
        ask_idx = s[1].num_ask_remain + 1
        return LinearIndices((num_map_states, problem.num_asks + 1))[pos_idx, ask_idx]
    end
end

function MOMDPs.stateindex_y(problem::RockSampleMOMDPAT{K}, y::RockSampleMOMDPYState{K}) where {K}
    temp_pos = (1, 1)
    temp_asks = problem.num_asks == -1 ? 1 : problem.num_asks
    st = (RockSampleMOMDPXState(temp_pos, temp_asks), y)
    return MOMDPs.stateindex_y(problem, st)
end

function MOMDPs.stateindex_y(problem::RockSampleMOMDPAT{K}, s::RockSampleMOMDPState{K}) where {K}
    rocks = s[2].rocks
    rocks_re_idx = Vector{Int}(undef, K)
    for (i, rock) in enumerate(rocks)
        rocks_re_idx[i] = rock ? 1 : 2
    end
    rock_array = Tuple([2 for _ in 1:K])
    rock_idx = LinearIndices(rock_array)[rocks_re_idx...]
    sugg_idx = findfirst(isequal(s[2].sugg_type), problem.types)
    @assert !isnothing(sugg_idx) "Invalid suggestion type $(s[2].sugg_type)"
    return LinearIndices((2^K, length(problem.types)))[rock_idx, sugg_idx]
end

function MOMDPs.initialstate_x(problem::RockSampleMOMDPAT{K}) where {K}
    map_states = vec([(i, j) for i in 1:problem.map_size[1], j in 1:problem.map_size[2]])
    num_asks = problem.num_asks == -1 ? 1 : problem.num_asks
    statesX = [RockSampleMOMDPXState(pos, num_asks) for pos in map_states]
    probs = normalize(ones(length(statesX)), 1)
    return SparseCat(statesX, probs)
end

# If non-uniform initial distirbution for x is desired, use this function
# function MOMDPs.initialstate_x(problem::RockSampleMOMDPAT)
#     num_asks = problem.num_asks == -1 ? 1 : problem.num_asks
#     return Deterministic(RockSampleMOMDPXState(problem.init_pos, num_asks))
# end

function MOMDPs.initialstate_y(problem::RockSampleMOMDPAT{K}, x::RockSampleMOMDPXState) where K
    rock_states = Vector{RSRocksAT{K}}(undef, 2^K)
    for (i,rocks) in enumerate(Iterators.product(ntuple(x->[false, true], K)...))
        rock_states[i] = Tuple(rocks)
    end
    
    rock_probs = ones(2^K) ./ 2^K
    
    num_init_states = 2^K * length(problem.types)
    states_y_vec = Vector{RockSampleMOMDPYState{K}}(undef, num_init_states)
    probs = zeros(num_init_states)
    
    ii = 0
    for (jj, rock_state) in enumerate(rock_states)
        for (kk, type_val) in enumerate(problem.types)
            ii += 1
            probs[ii] = rock_probs[jj] * problem.init_type_dist[kk]
            states_y_vec[ii] = RockSampleMOMDPYState{K}(rock_state, type_val)
        end
    end
    @assert isapprox(sum(probs), 1.0, atol=1e-6) "Invalid initial state y distribution. Probabilities sum to $(sum(probs))"
    return SparseCat(states_y_vec, probs)
end
