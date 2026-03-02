POMDPs.observations(problem::TagMOMDPAT) = collect(1:num_observations(problem))
POMDPs.obsindex(problem::TagMOMDPAT, o::Int) = o in 1:num_observations(problem) ? o : error("Invalid observation: $o")

function POMDPs.observation(problem::TagMOMDPAT, a::Int, sp::TagMOMDPState)::SparseCat{Vector{Int}, Vector{Float64}}
    if a == ACTIONS_DICT[:ask]
        @assert problem.num_asks != 0 "Cannot ask when num_asks is 0"
        
        # The agent asks for a suggestion. We can get the probability of an observation
        x_surr = sp[1].r_pos + 1
        y_surr = sp[2].t_pos
        Qsa = problem.Q_ask_array[x_surr, y_surr, :]
        Qsa_max = maximum(Qsa)
        probs = exp.(sp[2].sugg_type .* (Qsa .- Qsa_max))
        probs = probs ./ sum(probs)
        vals = collect(3:num_observations(problem))
        return SparseCat(vals, probs)
    end
    if sp[1].r_pos == sp[2].t_pos
        return SparseCat([1], [1.0])
    end
    return SparseCat([2], [1.0]) # 2 is the observation for not being able to tag
end

# num_asks = -1: infinite asking
function num_observations(problem::TagMOMDPAT)
    if problem.num_asks == 0
        return 2
    end
    return 2 + length(ACTIONS_DICT) - 1 # 2 + 1 for each non-ask action
end
