

POMDPs.actions(problem::RockSampleMOMDPAT{K}) where {K} = collect(1:num_actions(problem))

function POMDPs.actionindex(problem::RockSampleMOMDPAT, a::Int)
    @assert a >= 1 "Invalid action index $a"
    @assert a <= num_actions(problem) "Invalid action index $a"
    return a
end

function POMDPs.actions(problem::RockSampleMOMDPAT{K}, s::RockSampleMOMDPState{K}) where {K}
    x = s[1].pos
    if in(x, problem.rocks_positions)
        return actions(problem)
    else
        # sample not available
        if problem.num_asks == 0
            return collect(2:(N_BASIC_ACTIONS-1+K)) # action 6 = sense rock 1
        else
            return collect(2:(N_BASIC_ACTIONS+K))  # all actions, 6 = ask, 7 = sense rock 1
        end
    end
end

function num_actions(problem::RockSampleMOMDPAT{K}) where {K}
    if problem.num_asks == 0
        return N_BASIC_ACTIONS - 1 + K  # -1 for ask action
    end
    return N_BASIC_ACTIONS + K
end

"""
    list_actions(problem::RockSampleMOMDPAT)

Prints a list of actions and their meanings.
"""
function list_actions(problem::RockSampleMOMDPAT{K}) where {K}
    println("Actions:")
    println("  1: sample")
    println("  2: north")
    println("  3: east")
    println("  4: south")
    println("  5: west")
    if problem.num_asks != 0
        println("  6: ask")
    end
    for i in 1:K
        strt_idx = num_actions(problem)
        println("  $(strt_idx+i): check rock $i")
    end
end
