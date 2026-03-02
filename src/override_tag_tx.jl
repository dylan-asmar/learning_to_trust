const TagMOMDPState = Tuple{TagMOMDPXState, TagMOMDPYState}

const ACTIONS_DICT = Dict(:north => 1, :east => 2, :south => 3, :west => 4, :tag => 5, :ask => 6)
const ACTION_NAMES = Dict(1 => "North", 2 => "East", 3 => "South", 4 => "West", 5 => "Tag", 6 => "Ask")

function MOMDPs.transition_y(problem::TagMOMDPAT, s::TagMOMDPState, a::Int, xp::TagMOMDPXState)::SparseCat{Vector{TagMOMDPYState}, Vector{Float64}}
    if s[1].r_pos == 0
        # Suggester type stays the same, but start problem over again (uniform over target positions)
        states_y_vec = [TagMOMDPYState(i, s[2].sugg_type) for i in 1:get_prop(problem.mg, :num_grid_pos)]
        probs = normalize(ones(length(states_y_vec)), 1)
        return SparseCat(states_y_vec, probs)
    end
    if problem.transition_option == :orig
        return orig_transition_y_mod(problem, s, a)
    elseif problem.transition_option == :modified
        return modified_transition_y_mod(problem, s, a)
    else
        return error("Invalid transition option. $(problem.transition_option) is not implemented.")
    end
end

function orig_transition_y_mod(problem::TagMOMDPAT, s::TagMOMDPState, a::Int)::SparseCat{Vector{TagMOMDPYState}, Vector{Float64}}
    # Check if tagged first. If so, move to the terminal state
    if a == ACTIONS_DICT[:tag] &&s[1].r_pos == s[2].t_pos
        return SparseCat([s[2]], [1.0])
    end

    # If action is ask, no changes happen
    if a == ACTIONS_DICT[:ask]
        return SparseCat([s[2]], [1.0])
    end
    
    # Find nodes that are within one step of the target from the graph
    target_neighbors = neighbors(problem.mg, s[2].t_pos)

    # Distance from the target neighbors to the robot position on the graph
    target_neigh_to_robot_dist = problem.dist_matrix[s[1].r_pos, target_neighbors]
    current_dist = problem.dist_matrix[s[1].r_pos, s[2].t_pos]

    max_dist = max(maximum(target_neigh_to_robot_dist), current_dist)
    t_move_pos_options = target_neighbors[target_neigh_to_robot_dist .>= max_dist]

    # If there are no valid moves, stay in place
    if length(t_move_pos_options) == 0
        return SparseCat([s[2]], [1.0])
    end

    # Move directions for the move options
    t_move_dirs = [get_prop(problem.mg, s[2].t_pos, t_pos′, :action) for t_pos′ in t_move_pos_options]

    # Isolate ns and ew moves
    ns_moves = t_move_pos_options[(t_move_dirs .== :north) .| (t_move_dirs .== :south)]
    ew_moves = t_move_pos_options[(t_move_dirs .== :east) .| (t_move_dirs .== :west)]

    robot_map_coord = get_prop(problem.mg, :node_pos_mapping)[s[1].r_pos]
    target_map_coord = get_prop(problem.mg, :node_pos_mapping)[s[2].t_pos]

    # Check if moving in each direction would result in moving away from the target
    # ignoring walls and only considering map coordinates
    north_move_away = robot_map_coord[1] >= target_map_coord[1]
    south_move_away = robot_map_coord[1] <= target_map_coord[1]
    east_move_away = robot_map_coord[2] <= target_map_coord[2]
    west_move_away = robot_map_coord[2] >= target_map_coord[2]

    ns_probs = 0.0
    ew_probs = 0.0

    ns_away = north_move_away + south_move_away
    ew_away = east_move_away + west_move_away

    if ns_away > 0
        ns_probs = problem.move_away_probability / 2 / ns_away
    end

    if ew_away > 0
        ew_probs = problem.move_away_probability / 2 / ew_away
    end

    # Create the transition probability array
    t_probs = zeros(length(t_move_pos_options) + 1)
    if length(ns_moves) > 0
        t_probs[1:length(ns_moves)] .= ns_probs
    end
    if length(ew_moves) > 0
        t_probs[length(ns_moves)+1:(length(ns_moves) + length(ew_moves))] .= ew_probs
    end

    # Add the stay in place probability
    push!(t_move_pos_options, s[2].t_pos)
    t_probs[end] = 1.0 - sum(t_probs[1:end-1])

    num_new_states = length(t_move_pos_options) * length(problem.types)
    new_probs = zeros(num_new_states)
    new_states = Vector{TagMOMDPYState}(undef, num_new_states)

    ii = 0
    for (jj, t_pos′) in enumerate(ns_moves)
        for type_i in problem.types
            ii += 1
            if type_i == s[2].sugg_type
                p = 1 - problem.type_trans
            else
                p = problem.type_trans / (length(problem.types) - 1)
            end
            new_probs[ii] = t_probs[jj] * p
            new_states[ii] = TagMOMDPYState(t_pos′, type_i)
        end
    end
    for (jj, t_pos′) in enumerate(ew_moves)
        for type_i in problem.types
            ii += 1
            if type_i == s[2].sugg_type
                p = 1 - problem.type_trans
            else
                p = problem.type_trans / (length(problem.types) - 1)
            end
            new_probs[ii] = t_probs[jj + length(ns_moves)] * p
            new_states[ii] = TagMOMDPYState(t_pos′, type_i)
        end
    end
    for type_i in problem.types
        ii += 1
        if type_i == s[2].sugg_type
            p = 1 - problem.type_trans
        else
            p = problem.type_trans / (length(problem.types) - 1)
        end
        new_probs[ii] = t_probs[end] * p
        new_states[ii] = TagMOMDPYState(s[2].t_pos, type_i)
    end
    normalize!(new_probs, 1)
    nz_probs = findall(new_probs .> 0)
    return SparseCat(new_states[nz_probs], new_probs[nz_probs])
end

function modified_transition_y_mod(problem::TagMOMDPAT, s::TagMOMDPState, a::Int)::SparseCat{Vector{TagMOMDPYState}, Vector{Float64}}
    # Check if tagged first. If so, move to the terminal state
    if a == ACTIONS_DICT[:tag] && s[1].r_pos == s[2].t_pos
        return SparseCat([s[2]], [1.0])
    end

    # If action is ask, no changes happen
    if a == ACTIONS_DICT[:ask]
        return SparseCat([s[2]], [1.0])
    end
    
    # Find nodes that are within one step of the target from the graph
    target_neighbors = neighbors(problem.mg, s[2].t_pos)

    # Distance from the target neighbors to the robot position on the graph
    target_neigh_to_robot_dist = problem.dist_matrix[s[1].r_pos, target_neighbors]
    current_dist = problem.dist_matrix[s[1].r_pos, s[2].t_pos]

    max_dist = max(maximum(target_neigh_to_robot_dist), current_dist)
    t_move_pos_options = target_neighbors[target_neigh_to_robot_dist .>= max_dist]

    new_y_states = [TagMOMDPYState(t_pos′, s[2].sugg_type) for t_pos′ in t_move_pos_options]
    
    # If there are no valid moves, stay in place
    if length(t_move_pos_options) == 0
        new_probs = zeros(length(problem.types))
        new_y_states = Vector{TagMOMDPYState}(undef, length(problem.types))
        for (ii, type_i) in enumerate(problem.types)
            if type_i == s[2].sugg_type
                p = 1 - problem.type_trans
            else
                p = problem.type_trans / (length(problem.types) - 1)
            end
            new_probs[ii] = p
            new_y_states[ii] = TagMOMDPYState(s[2].t_pos, type_i)
        end
        normalize!(new_probs, 1)
        nz_probs = findall(new_probs .> 0)
        return SparseCat(new_y_states[nz_probs], new_probs[nz_probs])
    end

    # Create the transition probability array
    t_probs = ones(length(t_move_pos_options) + 1)
    t_probs[1:end-1] .= problem.move_away_probability / length(t_move_pos_options)

    # Add the stay in place probability
    push!(new_y_states, s[2])
    t_probs[end] = 1.0 - problem.move_away_probability
    
    num_new_states = length(new_y_states) * length(problem.types)
    new_probs = zeros(num_new_states)
    new_states = Vector{TagMOMDPYState}(undef, num_new_states)

    ii = 0
    for (jj, y_state) in enumerate(new_y_states)
        for type_i in problem.types
            ii += 1
            if type_i == y_state.sugg_type
                p = 1 - problem.type_trans
            else
                p = problem.type_trans / (length(problem.types) - 1)
            end
            new_probs[ii] = t_probs[jj] * p
            new_states[ii] = TagMOMDPYState(y_state.t_pos, type_i)
        end
    end
    normalize!(new_probs, 1)
    nz_probs = findall(new_probs .> 0)
    return SparseCat(new_states[nz_probs], new_probs[nz_probs])
end

function move_direction(problem::TagMOMDPAT, v::Int, a::Int)::Int
    neighs = neighbors(problem.mg, v)
    for n_i in neighs
        if ACTIONS_DICT[get_prop(problem.mg, v, n_i, :action)] == a
            return n_i
        end
    end
    return v
end
