const OBSERVATION_NAME = (:good, :bad, :none)

function num_observations(problem::RockSampleMOMDPAT{K}) where {K}
    if problem.num_asks != 0
        return 3 + N_BASIC_ACTIONS - 1 + K
    else
        return 3
    end
end

function POMDPs.observations(problem::RockSampleMOMDPAT{K}) where {K}
    num_rocks = K
    if problem.num_asks != 0
        # good, bad, none, sample, north, south, east, west, sense 1, sense 2, ..., sense K
        return 1:(3 + N_BASIC_ACTIONS - 1 + num_rocks)
    else
        return 1:3 # good, bad, none
    end
end

function POMDPs.obsindex(problem::RockSampleMOMDPAT{K}, o::Int) where {K}
    @assert o >= 1 "Invalid observation index $o"
    @assert o <= num_observations(problem) "Invalid observation index $o"
    return o
end

function POMDPs.observation(problem::RockSampleMOMDPAT{K}, a::Int, s::RockSampleMOMDPState{K})::SparseCat{Vector{Int}, Vector{Float64}} where {K}
    x_state = s[1]
    y_state = s[2]

    n_basic_actions_mod = problem.num_asks == 0 ? N_BASIC_ACTIONS - 1 : N_BASIC_ACTIONS
    
    if a <= n_basic_actions_mod && a != BASIC_ACTIONS_DICT[:ask]
        # no obs for movement or sample actions
        return SparseCat([3], [1.0])
    elseif problem.num_asks != 0 && a == BASIC_ACTIONS_DICT[:ask]
        # The agent asks for a suggestion. We get the probability of an observation based on Q values
        x_idx = stateindex_x(problem, x_state)
        num_map_states = problem.map_size[1] * problem.map_size[2] + 1  # +1 for terminal
        if x_state.pos == (-1, -1)
            x_idx = num_map_states
        else
            x_idx = LinearIndices(problem.map_size)[x_state.pos[1], x_state.pos[2]]
        end
        
        rocks = s[2].rocks
        rocks_re_idx = Vector{Int}(undef, K)
        for (i, rock) in enumerate(rocks)
            rocks_re_idx[i] = rock ? 1 : 2
        end
        rock_array = Tuple([2 for _ in 1:K])
        y_idx = LinearIndices(rock_array)[rocks_re_idx...]
        
        Qsa = problem.Q_ask_array[x_idx, y_idx, :]
        Qsa_max = maximum(Qsa)
        probs = exp.(s[2].sugg_type .* (Qsa .- Qsa_max))
        probs = probs ./ sum(probs)
        na = N_BASIC_ACTIONS + K - 1 # -1 for ask action
        vals = collect(4:(3 + na))
        @assert length(vals) == length(probs) "Invalid observation probabilities"
        return SparseCat(vals, probs)
    else
        # We are sensing a rock
        rock_ind = a - n_basic_actions_mod
        rock_pos = problem.rocks_positions[rock_ind]
        dist = norm(rock_pos .- x_state.pos)
        efficiency = 0.5 * (1.0 + exp(-dist * log(2) / problem.sensor_efficiency))
        rock_state = y_state.rocks[rock_ind]
        if rock_state
            return SparseCat([1, 2], [efficiency, 1.0 - efficiency])
        else
            return SparseCat([1, 2], [1.0 - efficiency, efficiency])
        end
    end
end
