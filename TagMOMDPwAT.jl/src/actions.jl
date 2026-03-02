POMDPs.actions(problem::TagMOMDPAT) = collect(1:num_actions(problem))

function POMDPs.actionindex(problem::TagMOMDPAT, a::Int)
    @assert a in 1:num_actions(problem) "Invalid action index $a"
    return a
end
function POMDPs.actionindex(problem::TagMOMDPAT, a::Symbol)
    if problem.num_asks == 0
        @assert a != :ask "Invalid action symbol $a. No asking allowed."
    end
    @assert a in keys(ACTIONS_DICT) "Invalid action symbol $a"
    return ACTIONS_DICT[a]
end

"""
    list_actions(pomdp::TagMOMDPAT)

Prints a list of actions and their symbol (name).
"""
function list_actions(problem::TagMOMDPAT)
    println("Actions:")
    for (name, a) in ACTIONS_DICT
        if problem.num_asks == 0 && name == :ask
            continue
        end
        println("  $a: $name")
    end
end

# num_asks = -1: infinite asking
function num_actions(problem::TagMOMDPAT)
    if problem.num_asks == 0
        return length(ACTIONS_DICT) - 1
    end
    return length(ACTIONS_DICT)
end
