"""
Abstract base type for all suggester implementations.
"""
abstract type AbstractSuggester end

"""
Policy-based suggester that uses action values from a policy and applies softmax sampling.

# Fields
- `policy`: The policy to use for generating action values
- `λ`: Temperature parameter for softmax sampling
"""
struct PolicySuggester <: AbstractSuggester
    policy
    λ::Float64
end

"""
Rule-based suggester that implements deterministic or stochastic rules for action selection.

# Fields
- `rule_function`: Function that takes (momdp, state, rng) and returns an action
"""
struct RuleSuggester <: AbstractSuggester
    rule_function::Function
    λ::Float64
end

"""
Random suggester that selects actions uniformly at random from the action space.
"""
struct RandomSuggester <: AbstractSuggester
    λ::Float64
end

"""
No suggester that does not suggest any action.
"""
struct NoSuggester <: AbstractSuggester
    λ::Float64
end


"""
    get_suggestion(suggester::AbstractSuggester, momdp, s, rng)

Generate a suggestion based on the suggester type and current state.

# Arguments
- `suggester`: The suggester instance to use
- `momdp`: The MOMDP problem
- `s`: The current state 
- `rng`: Random number generator

# Returns
- `o`: The suggested action observation
"""
function get_suggestion end

"""
Policy-based suggestion generation using softmax sampling over action values.
"""
function get_suggestion(suggester::PolicySuggester, momdp, s, rng)
    # Create suggestion state based on problem type
    if suggester.policy.momdp isa RockSampleMOMDPAT
        sugg_y = RockSampleMOMDPYState(s[2].rocks, 1.0)
        bₛ_y = SparseCat([sugg_y], [1.0])
        sugg_x = RockSampleMOMDPXState(s[1].pos, 0)
    elseif suggester.policy.momdp isa TagMOMDPAT
        sugg_y = TagMOMDPYState(s[2].t_pos, 1.0)
        bₛ_y = SparseCat([sugg_y], [1.0])
        sugg_x = TagMOMDPXState(s[1].r_pos, 0)
    else
        error("Unsupported problem type: $(typeof(suggester.policy.momdp))")
    end
    
    # Get action values and apply softmax
    q_as = actionvalues(suggester.policy, bₛ_y, sugg_x)
    p_as = exp.(suggester.λ * (q_as .- maximum(q_as)))
    p_as = p_as ./ sum(p_as)
    o = sample(rng, Weights(p_as))
    return o
end

"""
Rule-based suggestion generation using custom rule function.
"""
function get_suggestion(suggester::RuleSuggester, momdp, s, rng)
    # Apply the rule function to get an action
    action = suggester.rule_function(momdp, s, suggester.λ, rng)
    return action
end

"""
Random suggestion generation.
"""
function get_suggestion(suggester::RandomSuggester, momdp, s, rng)
    # Sample a random action from the action space
    na = length(actions(momdp))
    action = rand(rng, 1:na)    
    return action
end

"""
No suggester that does not suggest any action.
"""
function get_suggestion(suggester::NoSuggester, momdp, s, rng)
    return -1
end

function tag_regional_heuristic_normal_map(momdp::TagMOMDPAT, s, λ::Float64, rng)
    @assert momdp.mg.gprops[:num_grid_pos] == 29 "This heuristic only works for the normal Tag map"
    
    r_pos = s[1].r_pos
    t_pos = s[2].t_pos
    
    left_region = [10, 11, 20, 21]
    right_region = [18, 19, 28, 29]
    upper_region = [1, 2, 3, 4, 5, 6]
    
    if t_pos in left_region 
        if !(r_pos in left_region)
            return TagMOMDPProblemAT.ACTIONS_DICT[:west] 
        else
            return -1
        end
    elseif t_pos in right_region
        if !(r_pos in right_region)
            return TagMOMDPProblemAT.ACTIONS_DICT[:east]
        else
            return -1
        end
    elseif t_pos in upper_region
        if !(r_pos in upper_region)
            return TagMOMDPProblemAT.ACTIONS_DICT[:north]
        else
            return -1
        end
    end
    return -1
end
