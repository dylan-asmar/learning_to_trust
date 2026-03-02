function POMDPs.reward(problem::TagMOMDPAT, s::TagMOMDPState, a::Int)
    if s[1].r_pos == 0
        return 0.0
    end
    if a == ACTIONS_DICT[:tag]
        if s[1].r_pos == s[2].t_pos
            return problem.tag_reward
        end
        return problem.tag_penalty
    end
    if a == ACTIONS_DICT[:ask]
        @assert problem.num_asks != 0 "Cannot ask when num_asks is 0"
        if s[1].num_ask_remain > 0
            return problem.ask_penalty
        end
        # Might need to adjust/change this, but it is a large penalty for asking when you can't
        return problem.tag_penalty
    end
    return problem.step_penalty
end
