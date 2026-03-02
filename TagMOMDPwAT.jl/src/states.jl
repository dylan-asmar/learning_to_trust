function MOMDPs.states_x(problem::TagMOMDPAT)
    num_grid_pos = get_prop(problem.mg, :num_grid_pos)
    grid_pos = 0:num_grid_pos
    if problem.num_asks == -1
        num_asks_vec = [1]
    else
        num_asks_vec = 0:problem.num_asks
    end
    return vec([TagMOMDPXState(i, j) for i in grid_pos, j in num_asks_vec])
end

function MOMDPs.stateindex_x(problem::TagMOMDPAT, x::TagMOMDPXState)
    return MOMDPs.stateindex_x(problem, (x, TagMOMDPYState(1, 1)))
end
function MOMDPs.stateindex_x(problem::TagMOMDPAT, s::TagMOMDPState)
    num_grid_pos = get_prop(problem.mg, :num_grid_pos)
    @assert s[1].r_pos >= 0 && s[1].r_pos <= num_grid_pos "Invalid robot position"
    r_idx = s[1].r_pos + 1
    if problem.num_asks == 0
        @assert s[1].num_ask_remain == 0 "Invalid number of asks for x state $(s[1])"
        return r_idx
    elseif problem.num_asks == -1
        @assert s[1].num_ask_remain == 1 "Invalid number of asks for x state $(s[1])"
        return r_idx
    else
        @assert s[1].num_ask_remain >= 0 && s[1].num_ask_remain <= problem.num_asks "Invalid number of asks for x state $(s[1])"
        ask_idx = s[1].num_ask_remain + 1
        return LinearIndices((num_grid_pos + 1, problem.num_asks + 1))[r_idx, ask_idx]
    end
end



function MOMDPs.states_y(problem::TagMOMDPAT)
    num_grid_pos = get_prop(problem.mg, :num_grid_pos)
    grid_pos = 1:num_grid_pos
    return vec([TagMOMDPYState(i, j) for i in grid_pos, j in problem.types])
end

function MOMDPs.stateindex_y(problem::TagMOMDPAT, y::TagMOMDPYState)
    return MOMDPs.stateindex_y(problem, (TagMOMDPXState(1, 0), y))
end
function MOMDPs.stateindex_y(problem::TagMOMDPAT, s::TagMOMDPState)
    num_grid_pos = get_prop(problem.mg, :num_grid_pos)
    @assert s[2].t_pos >= 1 && s[2].t_pos <= num_grid_pos "Invalid target position $s.t_pos"
    t_idx = s[2].t_pos
    sugg_idx = findfirst(isequal(s[2].sugg_type), problem.types)
    @assert !isnothing(sugg_idx) "Invalid suggestion type $(s[2].sugg_type)"
    return LinearIndices((num_grid_pos, length(problem.types)))[t_idx, sugg_idx]
end

function MOMDPs.initialstate_x(problem::TagMOMDPAT)
    num_grid_pos = get_prop(problem.mg, :num_grid_pos)
    if problem.num_asks == -1
        num_asks = 1
    else
        num_asks = problem.num_asks
    end
    states_x_vec = [TagMOMDPXState(i, num_asks) for i in 1:num_grid_pos]
    probs = normalize(ones(length(states_x_vec)), 1)
    return SparseCat(states_x_vec, probs)
end

function MOMDPs.initialstate_y(problem::TagMOMDPAT, x::TagMOMDPXState)
    num_grid_pos = get_prop(problem.mg, :num_grid_pos)
    num_init_states = num_grid_pos * length(problem.types)
    
    grid_probs = normalize(ones(num_grid_pos), 1)
    states_y_vec = Vector{TagMOMDPYState}(undef, num_init_states)
    probs = zeros(num_init_states)
    
    ii = 0
    for (jj, t_pos) in enumerate(1:num_grid_pos)
        for (kk, type_i) in enumerate(problem.types)
            ii += 1
            probs[ii] = grid_probs[jj] * problem.init_type_dist[kk]
            states_y_vec[ii] = TagMOMDPYState(t_pos, type_i)
        end
    end
    
    return SparseCat(states_y_vec, probs)
end
