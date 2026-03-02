"""
    get_problem_and_policy(problem::Symbol)

Returns the saved pomdp, policy and a string based on naming scheme

# Arguments
- `problem::Symbol`: problem of interest
"""
function get_problem_and_policy(problem::Symbol)
    if problem == :rs78
        load_str = "rs_7-8-20-0"
    elseif problem == :rs78_inf_1
        load_str = "rs_7-8-20-0-inf-1"
    elseif problem == :rs78_inf_2
        load_str = "rs_7-8-20-0-inf-2"
    elseif problem == :rs78_inf_5
        load_str = "rs_7-8-20-0-inf-5"
    elseif problem == :rs78_inf_0_1_2_5_10
        load_str = "rs_7-8-20-0-inf-0_1_2_5_10"
    elseif problem == :rs78_inf_0_1_2_5_10_w05
        load_str = "rs_7-8-20-0-inf-0_1_2_5_10_w05"
    elseif problem == :rs84
        load_str = "rs_8-4-10-1"
    elseif problem == :rs84_inf_1
        load_str = "rs_8-4-10-1-inf-1"
    elseif problem == :rs84_inf_2
        load_str = "rs_8-4-10-1-inf-2"
    elseif problem == :rs84_inf_5
        load_str = "rs_8-4-10-1-inf-5"
    elseif problem == :rs84_inf_0_1_2_5_10
        load_str = "rs_8-4-10-1-inf-0_1_2_5_10"
    elseif problem == :rs84_inf_0_1_2_5_10_w05
        load_str = "rs_8-4-10-1-inf-0_1_2_5_10_w05"
    elseif problem == :tag
        load_str = "tag_m"
    elseif problem == :tag_inf_1
        load_str = "tag_mat_inf_1"
    elseif problem == :tag_inf_2
        load_str = "tag_mat_inf_2"
    elseif problem == :tag_inf_5
        load_str = "tag_mat_inf_5"
    elseif problem == :tag_1_5
        load_str = "tag_mat_1_5"
    elseif problem == :tag_inf_0_1_2_5_10
        load_str = "tag_mat_inf_0_1_2_5_10"
    elseif problem == :tag_inf_0_1_2_5_10_w05
        load_str = "tag_mat_inf_0_1_2_5_10_w05"
    elseif problem == :tag_1_0_1_2_5_10
        load_str = "tag_mat_1_0_1_2_5_10"
    elseif problem == :tag_1_0_1_2_5_10_w05
        load_str = "tag_mat_1_0_1_2_5_10_w05"
    else
        error("Problem not defined: $problem")
    end
    load_str = "policies/" * load_str
    @printf("Loading problem and policy...")
    @load(load_str * "_pol.jld2", pol)
    @printf("complete!\n")
    return pol.momdp, pol, load_str
end

abstract type Result_Type end

struct SimResult <: Result_Type
    agent::Symbol
    λ_sugg::Float64
    problem::Symbol
    suggester_type
    no_ask_problem::Symbol
    max_suggestions
    seed::Int
    init_pos::Union{Tuple{Int, Int}, Nothing}
    max_steps::Int
    num_trials_per_sim::Int
    num_sims::Int
    reward_per_trial_per_sim::Vector{Vector{Float64}}
    steps_per_trial_per_sim::Vector{Vector{Int}}
    asks_per_trial_per_sim::Vector{Vector{Int}}
    total_reward_per_sim::Vector{Float64}
    total_steps_per_sim::Vector{Int}
    total_asks_per_sim::Vector{Int}
    b_sugg_type_per_trial_per_sim::Vector{Vector{Vector{Float64}}}
end

struct SimResultTypeEval <: Result_Type
    agent::Symbol
    λ_sugg::Float64
    problem::Symbol
    suggester_type
    type_problem::Symbol
    seed::Int
    init_pos::Union{Tuple{Int, Int}, Nothing}
    max_steps::Int
    num_trials_per_sim::Int
    num_sims::Int
    reward_per_trial_per_sim::Vector{Vector{Float64}}
    steps_per_trial_per_sim::Vector{Vector{Int}}
    asks_per_trial_per_sim::Vector{Vector{Int}}
    total_reward_per_sim::Vector{Float64}
    total_steps_per_sim::Vector{Int}
    total_asks_per_sim::Vector{Int}
    b_sugg_type_per_trial_per_sim::Vector{Vector{Vector{Float64}}}
    ν::Float64
end

function Base.show(io::IO, sim_result::Result_Type)
    println(io, "SimResult for Agent $(sim_result.agent)")
    for name in fieldnames(typeof(sim_result))
        if typeof(getfield(sim_result, name)) <: Vector
            print(io, "\t", name, ": ")
            if typeof(getfield(sim_result, name)[1]) <: Vector
                if typeof(getfield(sim_result, name)[1][1]) <: Vector
                    field_val = getfield(sim_result, name)
                    size_field_val = size(field_val)[1]
                    size_vector = size(field_val[1])[1]
                    size_vector_2 = size(field_val[1][1])[1]
                    print(io, typeof(field_val), " $size_field_val x $size_vector x $size_vector_2\n")
                else
                    field_val = getfield(sim_result, name)
                    size_field_val = size(field_val)[1]
                    size_vector = size(field_val[1])[1]
                    print(io, typeof(field_val), " $size_field_val x $size_vector\n")
                end
            else
                field_val = getfield(sim_result, name)
                size_field_val = size(field_val)[1]
                print(io, typeof(field_val), " $size_field_val\n")
            end
        else
            println(io, "\t", name, ": ", getfield(sim_result, name))
        end
    end
end

function get_stats(vector_of_data::Vector{Vector{Float64}})
    data = vcat(vector_of_data...)
    total_num_trials = length(data)
    data_ave = mean(data)
    data_std = std(data)
    data_std_err = data_std / sqrt(total_num_trials)
    data_95ci = data_std_err * 1.96
    return data_ave, data_std, data_std_err, data_95ci
end

function print_sim_result(sim_result::SimResult)
    r_ave = mean(sim_result.total_reward_per_sim)
    r_std = std(sim_result.total_reward_per_sim)
    r_std_err = r_std / sqrt(sim_result.num_sims)
    
    r_per_trial = vcat(sim_result.reward_per_trial_per_sim...)
    total_num_trials = length(r_per_trial)
    r_per_trial_ave = mean(r_per_trial)
    r_per_trial_std = std(r_per_trial)
    r_per_trial_std_err = r_per_trial_std / sqrt(total_num_trials)
    
    steps_per_trial = vcat(sim_result.steps_per_trial_per_sim...)
    steps_per_trial_ave = mean(steps_per_trial)
    steps_per_trial_std = std(steps_per_trial)
    steps_per_trial_std_err = steps_per_trial_std / sqrt(total_num_trials)
    
    asks_per_trial = vcat(sim_result.asks_per_trial_per_sim...)
    asks_per_trial_ave = mean(asks_per_trial)
    asks_per_trial_std = std(asks_per_trial)
    asks_per_trial_std_err = asks_per_trial_std / sqrt(total_num_trials)
    
    step_ave = mean(sim_result.total_steps_per_sim)
    step_std = std(sim_result.total_steps_per_sim)
    step_std_err = step_std / sqrt(sim_result.num_sims)

    @printf("--------------------------------\n")
    @printf("Agent: %s\n", sim_result.agent)
    @printf("λ_sugg: %.2f\n", sim_result.λ_sugg)
    @printf("Problem: %s\n", sim_result.problem)
    @printf("Sugggeter Type: %s\n", sim_result.suggester_type)
    @printf("No Ask Problem: %s\n", sim_result.no_ask_problem)
    @printf("Max Suggestions: %d\n", sim_result.max_suggestions)
    @printf("Seed: %d\n", sim_result.seed)
    @printf("Initial Position: %s\n", sim_result.init_pos)
    @printf("Max Steps: %d\n", sim_result.max_steps)
    @printf("Num Trials: %d\n", sim_result.num_trials_per_sim)
    @printf("Num Simulations: %d\n", sim_result.num_sims)
    @printf("Total Num Trials: %d\n", total_num_trials)
    
    @printf("\n")
    @printf("%20s | %15s | %15s | %15s | %15s\n",
        "Metric", "Mean", "Standard Dev", "Standard Error", "+/- 95 CI")
    @printf("%20s | %15s | %15s | %15s | %15s\n",
        "---------------", "---------------", "---------------",
        "---------------", "---------------")
    @printf("%20s | %15.5f | %15.5f | %15.5f | %15.5f\n",
        "Reward", r_ave, r_std, r_std_err, 1.96 * r_std_err)
    @printf("%20s | %15.5f | %15.5f | %15.5f | %15.5f\n",
        "Total Steps", step_ave, step_std, step_std_err, 1.96 * step_std_err)
    @printf("%20s | %15.5f | %15.5f | %15.5f | %15.5f\n",
        "Reward Per Trial", r_per_trial_ave, r_per_trial_std, r_per_trial_std_err,
        1.96 * r_per_trial_std_err)
    @printf("%20s | %15.5f | %15.5f | %15.5f | %15.5f\n",
        "Steps Per Trial", steps_per_trial_ave, steps_per_trial_std, steps_per_trial_std_err,
        1.96 * steps_per_trial_std_err)
    @printf("%20s | %15.5f | %15.5f | %15.5f | %15.5f\n",
        "Asks Per Trial", asks_per_trial_ave, asks_per_trial_std, asks_per_trial_std_err,
        1.96 * asks_per_trial_std_err)
end

function print_sim_result(sim_result::SimResultTypeEval)
    r_ave = mean(sim_result.total_reward_per_sim)
    r_std = std(sim_result.total_reward_per_sim)
    r_std_err = r_std / sqrt(sim_result.num_sims)
    
    r_per_trial = vcat(sim_result.reward_per_trial_per_sim...)
    total_num_trials = length(r_per_trial)
    r_per_trial_ave = mean(r_per_trial)
    r_per_trial_std = std(r_per_trial)
    r_per_trial_std_err = r_per_trial_std / sqrt(total_num_trials)
    
    steps_per_trial = vcat(sim_result.steps_per_trial_per_sim...)
    steps_per_trial_ave = mean(steps_per_trial)
    steps_per_trial_std = std(steps_per_trial)
    steps_per_trial_std_err = steps_per_trial_std / sqrt(total_num_trials)
    
    asks_per_trial = vcat(sim_result.asks_per_trial_per_sim...)
    asks_per_trial_ave = mean(asks_per_trial)
    asks_per_trial_std = std(asks_per_trial)
    asks_per_trial_std_err = asks_per_trial_std / sqrt(total_num_trials)
    
    step_ave = mean(sim_result.total_steps_per_sim)
    step_std = std(sim_result.total_steps_per_sim)
    step_std_err = step_std / sqrt(sim_result.num_sims)

    @printf("--------------------------------\n")
    @printf("Agent: %s\n", sim_result.agent)
    @printf("λ_sugg: %.2f\n", sim_result.λ_sugg)
    @printf("Problem: %s\n", sim_result.problem)
    @printf("Sugggeter Type: %s\n", sim_result.suggester_type)
    @printf("Type Problem: %s\n", sim_result.type_problem)
    @printf("ν: %.2f\n", sim_result.ν)
    @printf("Seed: %d\n", sim_result.seed)
    @printf("Initial Position: %s\n", sim_result.init_pos)
    @printf("Max Steps: %d\n", sim_result.max_steps)
    @printf("Num Trials: %d\n", sim_result.num_trials_per_sim)
    @printf("Num Simulations: %d\n", sim_result.num_sims)
    @printf("Total Num Trials: %d\n", total_num_trials)
    
    @printf("\n")
    @printf("%20s | %15s | %15s | %15s | %15s\n",
        "Metric", "Mean", "Standard Dev", "Standard Error", "+/- 95 CI")
    @printf("%20s | %15s | %15s | %15s | %15s\n",
        "---------------", "---------------", "---------------",
        "---------------", "---------------")
    @printf("%20s | %15.5f | %15.5f | %15.5f | %15.5f\n",
        "Reward", r_ave, r_std, r_std_err, 1.96 * r_std_err)
    @printf("%20s | %15.5f | %15.5f | %15.5f | %15.5f\n",
        "Total Steps", step_ave, step_std, step_std_err, 1.96 * step_std_err)
    @printf("%20s | %15.5f | %15.5f | %15.5f | %15.5f\n",
        "Reward Per Trial", r_per_trial_ave, r_per_trial_std, r_per_trial_std_err,
        1.96 * r_per_trial_std_err)
    @printf("%20s | %15.5f | %15.5f | %15.5f | %15.5f\n",
        "Steps Per Trial", steps_per_trial_ave, steps_per_trial_std, steps_per_trial_std_err,
        1.96 * steps_per_trial_std_err)
    @printf("%20s | %15.5f | %15.5f | %15.5f | %15.5f\n",
        "Asks Per Trial", asks_per_trial_ave, asks_per_trial_std, asks_per_trial_std_err,
        1.96 * asks_per_trial_std_err)
end

function no_ask_action(momdp::MOMDP{X,Y,A,O}, state_list_y::Vector{Y}, 
    momdp_no_ask::MOMDP{X,Y,A,O}, state_list_y_no::Vector{Y}, 
    policy_no_ask::Policy, b::AbstractVector, s::Tuple{X,Y}) where {X,Y,A,O}
    
    if momdp isa TagMOMDPAT
        s_x_no = TagMOMDPXState(s[1].r_pos, 0)
    elseif momdp isa RockSampleMOMDPAT
        s_x_no = RockSampleMOMDPXState(s[1].pos, 0)
    else
        error("Unsupported problem type: $(typeof(momdp))")
    end
    
    # Now we need to marginalize the belief over the non type part of the state
    @assert length(state_list_y) == length(b)
    by = zeros(length(state_list_y_no))
    for (i, yi) in enumerate(state_list_y)
        yi_no = convert_state_to_no_ask(momdp_no_ask, yi)
        yi_no_idx = stateindex_y(momdp_no_ask, yi_no)
        by[yi_no_idx] += b[i]
    end
    @assert isapprox(sum(by), 1.0; atol=1e-6) "Marginalized belief over types does not sum to 1: $(sum(by))"
    
    a_n = action(policy_no_ask, by, s_x_no)
    a_n = action_map(momdp_no_ask, momdp, a_n)
    return a_n
end

convert_state_to_no_ask(momdp::MOMDP{X,Y,A,O}, x::X) where {X,Y,A,O} = error("Not implemented for $(typeof(momdp))")
convert_state_to_no_ask(::RockSampleMOMDPAT, x::RockSampleMOMDPXState) = RockSampleMOMDPXState(x.pos, 0)
convert_state_to_no_ask(::TagMOMDPAT, x::TagMOMDPXState) = TagMOMDPXState(x.r_pos, 0)
convert_state_to_no_ask(momdp_no_ask::RockSampleMOMDPAT, y::RockSampleMOMDPYState) = RockSampleMOMDPYState(y.rocks, momdp_no_ask.types[1])
convert_state_to_no_ask(momdp_no_ask::TagMOMDPAT, y::TagMOMDPYState) = TagMOMDPYState(y.t_pos, momdp_no_ask.types[1])

convert_state_to_inf_ask(momdp::MOMDP{X,Y,A,O}, x::X) where {X,Y,A,O} = error("Not implemented for $(typeof(momdp))")
convert_state_to_inf_ask(::RockSampleMOMDPAT, x::RockSampleMOMDPXState) = RockSampleMOMDPXState(x.pos, 1)
convert_state_to_inf_ask(::TagMOMDPAT, x::TagMOMDPXState) = TagMOMDPXState(x.r_pos, 1)
convert_state_to_inf_ask(momdp_inf_ask::RockSampleMOMDPAT, y::RockSampleMOMDPYState) = RockSampleMOMDPYState(y.rocks, momdp_inf_ask.types[1])
convert_state_to_inf_ask(momdp_inf_ask::TagMOMDPAT, y::TagMOMDPYState) = TagMOMDPYState(y.t_pos, momdp_inf_ask.types[1])

suggestion_to_observation(momdp::MOMDP{X,Y,A,O}, suggestion::A) where {X,Y,A,O} = error("Not implemented for $(typeof(momdp))")
suggestion_to_observation(::RockSampleMOMDPAT, suggestion::Int) = suggestion + 3
suggestion_to_observation(::TagMOMDPAT, suggestion::Int) = suggestion + 2


# Gets the action from an alpha vector policy if your belief is based at one state (truth)
function action_known_state(policy, state_idx::Int)
    α_idx = argmax(αᵢ[state_idx] for αᵢ in policy.alphas)
    return policy.action_map[α_idx]
end

function action_known_state(policy::MOMDPAlphaVectorPolicy, x_idx, y_idx)
    α_idx = argmax(αᵢ[y_idx] for αᵢ in policy.alphas[x_idx])
    return policy.action_map[x_idx][α_idx]
end

function POMDPTools.Policies.action(p::MOMDPAlphaVectorPolicy, b::AbstractVector, x)
    x_idx = stateindex_x(p.momdp, x)
    by_vec = b

    best_dot = -Inf
    best_dot_idx = 1
    for i in 1:length(p.alphas[x_idx])
        val = dot(p.alphas[x_idx][i], by_vec)
        if val > best_dot
            best_dot = val
            best_dot_idx = i
        end
    end
    return p.action_map[x_idx][best_dot_idx]
end

function action_map(::POMDP{S,A,O}, ::POMDP{S,A,O}, a::A) where {S,A,O}
    return a
end

function action_map(problem1::RockSampleMOMDPAT{K}, problem2::RockSampleMOMDPAT{K}, a::Int) where {K}
    if (problem1.num_asks != 0 && problem2.num_asks != 0) ||
        (problem1.num_asks == 0 && problem2.num_asks == 0)
        return a
    end
    if problem1.num_asks != 0 && problem2.num_asks == 0
        if a <= 5
            return a
        end
        if a > 6 && a <= 6 + K
            return a - 1
        end
    end
    if problem1.num_asks == 0 && problem2.num_asks != 0
        if a <= 5
            return a
        end
        if a >= 6
            return a + 1
        end
    end
    error("Invalid action conversion options: $problem1, $problem2, $a")
end

function get_problem(problem::Symbol)
    
    problem in RS_PROBS || problem in TG_PROBS || error("Invalid problem: $problem")

    if problem in RS_PROBS
        discount_factor = 0.95
        exit_reward = 10.0
        good_rock_reward = 10.0
        bad_rock_penalty = -10.0
        step_penalty = 0.0

        if problem == :rs84
            save_str = "rs_8-4-10-1_pol.jld2"
            map_size = (8, 8)
            sensor_efficiency = 10.0
            sensor_use_penalty = -1.0
            rocks_positions = [(1,1),
                            (2,7),
                            (6,2),
                            (7,8)]
            init_pos = (3,4)
            
            num_asks = 0
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                num_asks=num_asks
            )
            
        elseif problem == :rs78
            save_str = "rs_7-8-20-0_pol.jld2"
            map_size = (7, 8)
            sensor_efficiency = 20.0
            sensor_use_penalty = 0.0
            rocks_positions = [
                (1,2),
                (2,7),
                (3,1),
                (3,5),
                (4,2),
                (4,5),
                (6,6),
                (7,4) # original
                # (7,2)
            ]
            init_pos = (1,4)
            
            num_asks = 0
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                num_asks=num_asks
            )
        
        elseif problem == :rs84_inf_1
            save_str = "rs_8-4-10-1-inf-1_pol.jld2"
            map_size = (8, 8)
            sensor_efficiency = 10.0
            sensor_use_penalty = -1.0
            rocks_positions = [(1,1),
                            (2,7),
                            (6,2),
                            (7,8)]
            init_pos = (3,4)
            
            Q_str = "policies/rs_8-4-10-1_Q.jld2"
            ask_cost = -1.0
            @load(Q_str, Q)
            
            λs = [1.0]
            init_type_dist = [1.0]
            num_asks = -1
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                ask_cost=ask_cost,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                Q_ask_array=Q,
                types=λs,
                init_type_dist=init_type_dist,
                num_asks=num_asks
            )
            
        elseif problem == :rs84_inf_2
            save_str = "rs_8-4-10-1-inf-2_pol.jld2"
            map_size = (8, 8)
            sensor_efficiency = 10.0
            sensor_use_penalty = -1.0
            rocks_positions = [(1,1),
                            (2,7),
                            (6,2),
                            (7,8)]
            init_pos = (3,4)
            
            Q_str = "policies/rs_8-4-10-1_Q.jld2"
            ask_cost = -1.0
            @load(Q_str, Q)
            
            λs = [2.0]
            init_type_dist = [1.0]
            num_asks = -1
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                ask_cost=ask_cost,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                Q_ask_array=Q,
                types=λs,
                init_type_dist=init_type_dist,
                num_asks=num_asks
            )
            
            
        elseif problem == :rs84_inf_5
            save_str = "rs_8-4-10-1-inf-5_pol.jld2"
            map_size = (8, 8)
            sensor_efficiency = 10.0
            sensor_use_penalty = -1.0
            rocks_positions = [(1,1),
                            (2,7),
                            (6,2),
                            (7,8)]
            init_pos = (3,4)
            
            Q_str = "policies/rs_8-4-10-1_Q.jld2"
            ask_cost = -1.0
            @load(Q_str, Q)
            
            λs = [5.0]
            init_type_dist = [1.0]
            num_asks = -1
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                ask_cost=ask_cost,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                Q_ask_array=Q,
                types=λs,
                init_type_dist=init_type_dist,
                num_asks=num_asks
            )
        
        elseif problem == :rs84_inf_0_1_2_5_10
            save_str = "rs_8-4-10-1-inf-0_1_2_5_10_pol.jld2"
            map_size = (8, 8)
            sensor_efficiency = 10.0
            sensor_use_penalty = -1.0
            rocks_positions = [(1,1),
                            (2,7),
                            (6,2),
                            (7,8)]
            init_pos = (3,4)
            
            Q_str = "policies/rs_8-4-10-1_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = -1.0
            λs = [0.0, 1.0, 2.0, 5.0, 10.0]
            init_type_dist = [0.1, 0.2, 0.4, 0.2, 0.1]
            num_asks = -1
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                ask_cost=ask_cost,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                Q_ask_array=Q,
                types=λs,
                num_asks=num_asks,
                init_type_dist=init_type_dist
            )
        
        elseif problem == :rs84_inf_0_1_2_5_10_w05
            save_str = "rs_8-4-10-1-inf-0_1_2_5_10_w05_pol.jld2"
            map_size = (8, 8)
            sensor_efficiency = 10.0
            sensor_use_penalty = -1.0
            rocks_positions = [(1,1),
                            (2,7),
                            (6,2),
                            (7,8)]
            init_pos = (3,4)
            
            Q_str = "policies/rs_8-4-10-1_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = -1.0
            λs = [0.0, 1.0, 2.0, 5.0, 10.0]
            init_type_dist = [0.1, 0.2, 0.4, 0.2, 0.1]
            num_asks = -1
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                ask_cost=ask_cost,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                Q_ask_array=Q,
                types=λs,
                num_asks=num_asks,
                init_type_dist=init_type_dist,
                type_trans=0.05
            )
    
        elseif problem == :rs78_inf_1
            save_str = "rs_7-8-20-0-inf-1_pol.jld2"
            map_size = (7, 8)
            sensor_efficiency = 20.0
            sensor_use_penalty = 0.0
            rocks_positions = [
                (1,2),
                (2,7),
                (3,1),
                (3,5),
                (4,2),
                (4,5),
                (6,6),
                (7,4) # original
                # (7,2)
            ]
            init_pos = (1,4)
            
            Q_str = "policies/rs_7-8-20-0_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = 0.0
            λs = [1.0]
            num_asks = -1
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                ask_cost=ask_cost,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                Q_ask_array=Q,
                types=λs,
                num_asks=num_asks
            ) 
            
        elseif problem == :rs78_inf_2
            save_str = "rs_7-8-20-0-inf-2_pol.jld2"
            map_size = (7, 8)
            sensor_efficiency = 20.0
            sensor_use_penalty = 0.0
            rocks_positions = [
                (1,2),
                (2,7),
                (3,1),
                (3,5),
                (4,2),
                (4,5),
                (6,6),
                (7,4) # original
                # (7,2)
            ]
            init_pos = (1,4)
            
            Q_str = "policies/rs_7-8-20-0_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = 0.0
            λs = [2.0]
            num_asks = -1
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                ask_cost=ask_cost,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                Q_ask_array=Q,
                types=λs,
                num_asks=num_asks
            ) 
            
            
        elseif problem == :rs78_inf_5
            save_str = "rs_7-8-20-0-inf-5_pol.jld2"
            map_size = (7, 8)
            sensor_efficiency = 20.0
            sensor_use_penalty = 0.0
            rocks_positions = [
                (1,2),
                (2,7),
                (3,1),
                (3,5),
                (4,2),
                (4,5),
                (6,6),
                (7,4) # original
                # (7,2)
            ]
            init_pos = (1,4)
            
            Q_str = "policies/rs_7-8-20-0_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = 0.0
            λs = [5.0]
            num_asks = -1
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                ask_cost=ask_cost,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                Q_ask_array=Q,
                types=λs,
                num_asks=num_asks
            ) 
            
        elseif problem == :rs78_inf_0_1_2_5_10
            save_str = "rs_7-8-20-0-inf-0_1_2_5_10_pol.jld2"
            map_size = (7, 8)
            sensor_efficiency = 20.0
            sensor_use_penalty = 0.0
            rocks_positions = [
                (1,2),
                (2,7),
                (3,1),
                (3,5),
                (4,2),
                (4,5),
                (6,6),
                (7,4) # original
                # (7,2)
            ]
            init_pos = (1,4)
            
            Q_str = "policies/rs_7-8-20-0_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = 0.0
            λs = [0.0, 1.0, 2.0, 5.0, 10.0]
            num_asks = -1
            init_type_dist = [0.1, 0.2, 0.4, 0.2, 0.1]
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                ask_cost=ask_cost,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                Q_ask_array=Q,
                types=λs,
                num_asks=num_asks,
                init_type_dist=init_type_dist,
                type_trans=0.00
            )
            
        elseif problem == :rs78_inf_0_1_2_5_10_w05
            save_str = "rs_7-8-20-0-inf-0_1_2_5_10_w05_pol.jld2"
            map_size = (7, 8)
            sensor_efficiency = 20.0
            sensor_use_penalty = 0.0
            rocks_positions = [
                (1,2),
                (2,7),
                (3,1),
                (3,5),
                (4,2),
                (4,5),
                (6,6),
                (7,4) # original
                # (7,2)
            ]
            init_pos = (1,4)
            
            Q_str = "policies/rs_7-8-20-0_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = 0.0
            λs = [0.0, 1.0, 2.0, 5.0, 10.0]
            num_asks = -1
            init_type_dist = [0.1, 0.2, 0.4, 0.2, 0.1]
            
            momdp = RockSampleMOMDPAT(;
                map_size=map_size,
                rocks_positions=rocks_positions,
                init_pos=init_pos,
                sensor_efficiency=sensor_efficiency,
                ask_cost=ask_cost,
                bad_rock_penalty=bad_rock_penalty,
                good_rock_reward=good_rock_reward,
                step_penalty=step_penalty,
                sensor_use_penalty=sensor_use_penalty,
                exit_reward=exit_reward,
                discount_factor=discount_factor,
                Q_ask_array=Q,
                types=λs,
                num_asks=num_asks,
                init_type_dist=init_type_dist,
                type_trans=0.05
            )
            
        end
    elseif problem in TG_PROBS
        if problem == :tag
            save_str = "tag_m_pol.jld2"
            momdp = TagMOMDPAT(; num_asks=0, types=[1.0])
        elseif problem == :tag_inf_1
            save_str = "tag_mat_inf_1_pol.jld2"
            Q_str = "policies/tag_m_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = -1.0
            λ = 1.0
            momdp = TagMOMDPAT(;
                num_asks=-1,
                Q_ask_array=Q,
                ask_penalty=ask_cost,
                types=[λ]
            )
        elseif problem == :tag_inf_2
            save_str = "tag_mat_inf_2_pol.jld2"
            Q_str = "policies/tag_m_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = -1.0
            λ = 2.0
            momdp = TagMOMDPAT(;
                num_asks=-1,
                Q_ask_array=Q,
                ask_penalty=ask_cost,
                types=[λ]
            )
        elseif problem == :tag_inf_5
            save_str = "tag_mat_inf_5_pol.jld2"
            Q_str = "policies/tag_m_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = -1.0
            λs = [5.0]
            momdp = TagMOMDPAT(;
                num_asks=-1,
                Q_ask_array=Q,
                ask_penalty=ask_cost,
                types=λs
            )
        elseif problem == :tag_1_5
            save_str = "tag_mat_1_5_pol.jld2"
            Q_str = "policies/tag_m_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = -1.0
            λs = [5.0]
            momdp = TagMOMDPAT(;
                num_asks=1,
                Q_ask_array=Q,
                ask_penalty=ask_cost,
                types=λs
            )
        elseif problem == :tag_inf_0_1_2_5_10
            save_str = "tag_mat_inf_0_1_2_5_10_pol.jld2"
            Q_str = "policies/tag_m_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = -1.0
            λs = [0.0, 1.0, 2.0, 5.0, 10.0]
            momdp = TagMOMDPAT(;
                num_asks=-1,
                Q_ask_array=Q,
                ask_penalty=ask_cost,
                types=λs,
                init_type_dist=[0.1, 0.2, 0.4, 0.2, 0.1],
                type_trans=0.00
            )
        elseif problem == :tag_inf_0_1_2_5_10_w05
            save_str = "tag_mat_inf_0_1_2_5_10_w05_pol.jld2"
            Q_str = "policies/tag_m_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = -1.0
            λs = [0.0, 1.0, 2.0, 5.0, 10.0]
            momdp = TagMOMDPAT(;
                num_asks=-1,
                Q_ask_array=Q,
                ask_penalty=ask_cost,
                types=λs,
                init_type_dist=[0.1, 0.2, 0.4, 0.2, 0.1],
                type_trans=0.05
            )
        elseif problem == :tag_1_0_1_2_5_10
            save_str = "tag_mat_1_0_1_2_5_10_pol.jld2"
            Q_str = "policies/tag_m_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = -1.0
            λs = [0.0, 1.0, 2.0, 5.0, 10.0]
            momdp = TagMOMDPAT(;
                num_asks=1,
                Q_ask_array=Q,
                ask_penalty=ask_cost,
                types=λs,
                init_type_dist=[0.1, 0.2, 0.4, 0.2, 0.1],
                type_trans=0.00
            )
        elseif problem == :tag_1_0_1_2_5_10_w05
            save_str = "tag_mat_1_0_1_2_5_10_w05_pol.jld2"
            Q_str = "policies/tag_m_Q.jld2"
            @load(Q_str, Q)
            
            ask_cost = -1.0
            λs = [0.0, 1.0, 2.0, 5.0, 10.0]
            momdp = TagMOMDPAT(;
                num_asks=1,
                Q_ask_array=Q,
                ask_penalty=ask_cost,
                types=λs,
                init_type_dist=[0.1, 0.2, 0.4, 0.2, 0.1],
                type_trans=0.05
            )
        end
    end
    
    return momdp, save_str
end
