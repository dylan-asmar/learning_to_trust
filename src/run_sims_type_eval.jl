
using LinearAlgebra
using StatsBase
using Random
using Printf
using JLD2
using ProgressMeter

using POMDPs
using POMDPTools
using MOMDPs

using RockSampleMOMDPProblemAT
using TagMOMDPProblemAT

using Distances

# To visualize RockSample
using Cairo
using Fontconfig

using Graphs
using MetaGraphs

include("constants.jl")
include("utils.jl")
include("suggesters.jl")

include("override_tag_tx.jl")

include("plot_results.jl")

function run_sim_type_eval(
    problem::Symbol,
    type_problem::Symbol;
    num_trials::Int=1,
    max_steps::Int=50,
    num_sims::Int=1,
    verbose::Bool=false,
    visualize::Bool=false,
    agent::Symbol=:normal,
    init_rocks=nothing,
    init_pos=nothing,
    seed=42,
    suggester::AbstractSuggester=NoSuggester(),
    ν::Float64=1.0
)

    problem in RS_PROBS || problem in TG_PROBS || error("Invalid problem: $problem")
    agent in AGENTS || error("Invalid agent: $agent")

    momdp, π, load_str = get_problem_and_policy(problem);
    momdp_types, _ = get_problem(type_problem);
    
    temp_λ = momdp.types[1] # This doesn't affect the simulation, but is needed for part of the y state
    
    state_list_x = ordered_states_x(momdp)
    state_list_y = ordered_states_y(momdp)
    state_list_y_types = ordered_states_y(momdp_types)
    nx = length(state_list_x)
    ny = length(state_list_y)
    ny_types = length(state_list_y_types)
    na = length(actions(momdp))

    if problem in RS_PROBS
        num_rocks = length(momdp.rocks_positions)
    end

    # Action value array for the agent where it is Q(x_idx, y_idx, a_idx)
    Q_str = load_str * "_Q.jld2"
    @load(Q_str, Q)
    
    r_vec = Vector{Float64}(undef, num_sims)
    step_vec = Vector{Int}(undef, num_sims)
    num_asks_vec = Vector{Int}(undef, num_sims)
    r_per_trial_vec = [Vector{Float64}(undef, num_trials) for _ in 1:num_sims]
    steps_per_trial_vec = [Vector{Int}(undef, num_trials) for _ in 1:num_sims]
    asks_per_trial_vec = [Vector{Int}(undef, num_trials) for _ in 1:num_sims]
    b_sugg_type_vec = [Vector{Vector{Float64}}(undef, num_trials) for _ in 1:num_sims]
    for i in 1:num_sims
        for j in 1:num_trials
            b_sugg_type_vec[i][j] = zeros(length(momdp.types))
        end
    end
    
    my_lock = ReentrantLock()
    
    p = Progress(num_sims; desc="Running Simulations", barlen=50, showspeed=true)
    Threads.@threads for ijk = 1:num_sims
    # for ijk = 1:num_sims
        # ijk = 1
        rng = MersenneTwister(seed + ijk - 1)
        
        policy_agent = deepcopy(π)

        belief_updater_agent = MOMDPDiscreteUpdater(momdp)
        belief_updater_types = MOMDPDiscreteUpdater(momdp_types)

        sugg_thread = deepcopy(suggester)
        
        # Get iniital state
        sᵢ = rand(rng, initialstate(momdp))
        if !isnothing(init_pos)
            if problem in RS_PROBS
                xi = RockSampleMOMDPXState(init_pos, momdp.num_asks)
                yi = RockSampleMOMDPYState(sᵢ[2].rocks, temp_λ)
                sᵢ = (xi, yi)
            elseif problem in TG_PROBS
                num_asks = momdp.num_asks == -1 ? 1 : momdp.num_asks
                xi = TagMOMDPXState(init_pos[1], num_asks)
                yi = TagMOMDPYState(init_pos[2], temp_λ)
                sᵢ = (xi, yi)
            end
        else
            if problem in RS_PROBS
                xi = RockSampleMOMDPXState(momdp.init_pos, momdp.num_asks)
                yi = RockSampleMOMDPYState(sᵢ[2].rocks, temp_λ)
                sᵢ = (xi, yi)
            end
        end
        
        
        if problem in RS_PROBS && !isnothing(init_rocks)
            # If we are in a RockSample problem and we provided initial rocks
            length(init_rocks) == num_rocks || error("Invalid init_rocks: $init_rocks")
            xi = sᵢ[1]
            yi_rocks = Tuple(init_rocks)
            yi = RockSampleMOMDPYState(yi_rocks, temp_λ)
            sᵢ = (xi, yi)
        end
        
        bᵢ = beliefvec_y(momdp, ny, initialstate_y(momdp, sᵢ[1]))
        bᵢ_types = beliefvec_y(momdp_types, ny_types, initialstate_y(momdp_types, sᵢ[1]))
        
        suggestion_cnt = 0
        step_cnt = 0
        total_reward = 0.0
        trial_reward = 0.0
        trial_time = 0
        num_ask = 0
        trial_cnt = 0
        ask_trial_cnt = 0
        try
            t = 0
            while trial_cnt < num_trials
                t += 1
                trial_time += 1
                step_cnt += 1
                bₒ = bᵢ # Original belief before any updates
                bₒ_types = bᵢ_types
                
                
                a_n = action(policy_agent, bᵢ, sᵢ[1])
                
                Qsa = Q[stateindex_x(momdp, sᵢ[1]), stateindex_y(momdp, sᵢ[2]), :]
                a_p = argmax(Qsa) # Perfect knowledge action

                # a is exectued action. Select based on agent type
                if agent == :normal
                    a = a_n
                elseif agent == :perfect
                    a = a_p
                elseif agent == :random
                    a = rand(rng, actions(momdp))
                end
                
                # Get a suggestion and update the belief.
                # If the agent is naive, then we just follow the suggestion
                if !(agent in [:perfect, :random]) && !(sugg_thread isa NoSuggester)
                    suggestion = get_suggestion(sugg_thread, momdp, sᵢ, rng)
                    if agent == :naive
                        if rand(rng) <= ν
                            a = suggestion
                        else
                            a = a_n
                        end
                    elseif agent == :normal
                        os = suggestion_to_observation(momdp, suggestion)
                        
                        if visualize
                            x_state = convert_state_to_inf_ask(momdp, sᵢ[1])
                            state_list_f = [(x_state, yi) for yi in state_list_y_types]
                              
                            sugg_type_txt = sugg_thread.λ
                            if sugg_thread isa RuleSuggester
                                sugg_type_txt = "Heuristic"
                            end
                            
                            at = action_map(momdp, momdp_types, a_n)
                            step = (s=sᵢ, a=at, b=SparseCat(state_list_f, bᵢ_types), sugg_type=sugg_type_txt, num_asks_remain="")
                            if num_trials > 1
                                pre_act_text = "Pre Trial: $(trial_cnt+1), t=$t, os=$suggestion, "
                            else
                                pre_act_text = "Pre t=$t, os=$suggestion, "
                            end
                            display(render(momdp_types, step; pre_act_text=pre_act_text))
                        end
                        
                        
                        # Hardcoded ask action (assumes ask action is 6)
                        bᵢ_types = SparseCat(state_list_y_types, bᵢ_types)
                        x_state = convert_state_to_inf_ask(momdp, sᵢ[1])
                        bᵢ_types = update(belief_updater_types, bᵢ_types, 6, os, x_state, x_state)
                        
                        # Convert the type belief back to a belief with no types
                        by_vec = zeros(ny)
                        for (yi, prob_i) in zip(bᵢ_types.state_list, bᵢ_types.b)
                            yi_no = convert_state_to_no_ask(momdp, yi)
                            yi_no_idx = stateindex_y(momdp, yi_no)
                            by_vec[yi_no_idx] += prob_i
                        end
                        @assert isapprox(sum(by_vec), 1.0; atol=1e-6) "Marginalized belief over types does not sum to 1: $(sum(by_vec))"
                        bᵢ = by_vec
                        bᵢ_types = beliefvec_y(momdp_types, ny_types, bᵢ_types)
                        a = action(policy_agent, bᵢ, sᵢ[1])
                    else
                        error("Invalid agent: $agent")
                    end
                end
                
                # Simulate a step forward with action `a` from state `sᵢ`
                (sp, o, r) = @gen(:sp, :o, :r)(momdp, sᵢ, a, rng)

                if visualize
                    x_state = convert_state_to_inf_ask(momdp, sᵢ[1])
                    state_list_f = [(x_state, yi) for yi in state_list_y_types]
                      
                    sugg_type_txt = sugg_thread.λ
                    if sugg_thread isa RuleSuggester
                        sugg_type_txt = "Heuristic"
                    end
                    
                    at = action_map(momdp, momdp_types, a)
                    step = (s=sᵢ, a=at, b=SparseCat(state_list_f, bᵢ_types), sugg_type=sugg_type_txt, num_asks_remain="")
                    if num_trials > 1
                        pre_act_text = "Trial: $(trial_cnt+1), t=$t, o=$o, "
                    else
                        pre_act_text = "t=$t, o=$o, "
                    end
                    display(render(momdp_types, step; pre_act_text=pre_act_text))
                end
                
                if verbose
                    println("--------------------------------")
                    println("Time                     : $t")
                    println("State                    : $sᵢ")
                    println("Initial Action:          : $a_n")
                    println("Perfect Knowledge Action : $a_p")
                    println("Selected Action          : $a")
                    println("Next State               : $sp")
                    println("Observation              : $o")
                    println("Immediate Reward         : $r")
                    println("Discounted Reward        : $(momdp.discount_factor^(t-1)*r)")
                    println()
                    println("--- Initial Belief at t = $t ---")
                    display(SparseCat(state_list_y[bₒ .> 0], bₒ[bₒ .> 0]))
                end

                if !(bᵢ isa SparseCat)
                    bᵢ = SparseCat(state_list_y, bᵢ)
                end
                if !(bᵢ_types isa SparseCat)
                    bᵢ_types = SparseCat(state_list_y_types, bᵢ_types)
                end
                
                # Update agent's belief with observation from environment
                bᵢ′ = update(belief_updater_agent, bᵢ, a, o, sᵢ[1], sp[1])
                bᵢ = beliefvec_y(momdp, ny, bᵢ′)
                
                x_state = convert_state_to_inf_ask(momdp, sᵢ[1])
                xp_state = convert_state_to_inf_ask(momdp, sp[1])
                
                at = action_map(momdp, momdp_types, a)
                bᵢ_types′ = update(belief_updater_types, bᵢ_types, at, o, x_state, xp_state)
                bᵢ_types = beliefvec_y(momdp_types, ny_types, bᵢ_types′)
                
                sᵢ = sp # Update state to transitioned to state
                total_reward += momdp.discount_factor^(t - 1) * r
                trial_reward += momdp.discount_factor^(trial_time - 1) * r
                
                if (momdp isa TagMOMDPAT && sᵢ[1].r_pos == 0) ||
                    (momdp isa RockSampleMOMDPAT && all(sᵢ[1].pos .== (-1, -1))) ||
                    (trial_time > max_steps)
                    
                    if trial_time > max_steps
                        @warn "Simulation $ijk, Trial $trial_cnt exceeded max steps: $max_steps"
                    end
                    
                    trial_cnt += 1
                    # Get the marginal of belief types and store them
                    b_sugg_type = zeros(length(momdp_types.types))
                    for (ii, yi) in enumerate(state_list_y_types)
                        type_i = findfirst(isequal(yi.sugg_type), momdp_types.types)
                        b_sugg_type[type_i] += bᵢ_types[ii]
                    end
                    b_sugg_type = b_sugg_type ./ sum(b_sugg_type)
                    
                    @lock my_lock b_sugg_type_vec[ijk][trial_cnt] = b_sugg_type
                    @lock my_lock r_per_trial_vec[ijk][trial_cnt] = trial_reward
                    @lock my_lock steps_per_trial_vec[ijk][trial_cnt] = trial_time
                    @lock my_lock asks_per_trial_vec[ijk][trial_cnt] = ask_trial_cnt
                    
                    
                    if momdp isa TagMOMDPAT    
                        # Reset problem and counters
                        different = false
                        while !different
                            (sp, o, r) = @gen(:sp, :o, :r)(momdp, sᵢ, 1, rng)
                            different = sp[1].r_pos != sp[2].t_pos
                        end
                        
                        bᵢ = SparseCat(state_list_y[bᵢ .> 0.0], bᵢ[bᵢ .> 0.0])
                        bᵢ′ = update(belief_updater_agent, bᵢ, 1, o, sᵢ[1], sp[1])
                        bᵢ = beliefvec_y(momdp, ny, bᵢ′)
                        
                        bᵢ_types = SparseCat(state_list_y_types[bᵢ_types .> 0.0], bᵢ_types[bᵢ_types .> 0.0])
                        bᵢ_types′ = update(belief_updater_types, bᵢ_types, 1, o, sᵢ[1], sp[1])
                        bᵢ_types = beliefvec_y(momdp_types, ny_types, bᵢ_types′)
                        
                        sᵢ = sp

                    elseif momdp isa RockSampleMOMDPAT                        
                        id = initialstate(momdp)
                        sp′ = sample(rng, id.vals, Weights(id.probs))
                        x′ = RockSampleMOMDPXState(momdp.init_pos, sp′[1].num_ask_remain)
                        y′ = RockSampleMOMDPYState(sp′[2].rocks, sᵢ[2].sugg_type)
                        sp′ = (x′, y′)
                        
                        bᵢ_uniform = uniform_belief_y(momdp)
                        
                        bᵢ = zeros(ny)
                        for (ii, bi) in enumerate(bᵢ_uniform.b)
                            if bi > 0.0
                                type_idx = findfirst(isequal(bᵢ_uniform.state_list[ii].sugg_type), momdp.types)
                                bᵢ[ii] = b_sugg_type[type_idx]
                            end
                        end
                        
                        bᵢ = bᵢ ./ sum(bᵢ)
                        
                        bᵢ_types_uniform = uniform_belief_y(momdp_types)
                        
                        bᵢ_types = zeros(ny_types)
                        for (ii, bi) in enumerate(bᵢ_types_uniform.b)
                            if bi > 0.0
                                type_idx = findfirst(isequal(bᵢ_types_uniform.state_list[ii].sugg_type), momdp_types.types)
                                bᵢ_types[ii] = b_sugg_type[type_idx]
                            end
                        end
                        bᵢ_types = bᵢ_types ./ sum(bᵢ_types)
                        
                        sᵢ = sp′
                    else
                        error("Invalid momdp: $(typeof(momdp))")
                    end                   
                    trial_reward = 0.0
                    trial_time = 0
                    ask_trial_cnt = 0
                    suggestion_cnt = 0
                end    
            end
        
        catch e
            println("****** ERROR IN SIMULATION $ijk ******")
            println("Agent: $agent")
            println("Problem: $problem")
            println("Suggester: $(typeof(suggester))")
            println("Type Problem: $type_problem")
            println("Simulation Number: $ijk")
            println("Step Count: $step_cnt")
            println("State: $sᵢ")
            println("Trial Count: $trial_cnt")
            
            rethrow(e)
        end
             
        r_vec[ijk] = total_reward
        step_vec[ijk] = step_cnt
        num_asks_vec[ijk] = num_ask
        
        next!(p)
    end

    sim_result = SimResultTypeEval(
        agent,
        suggester.λ,
        problem,
        typeof(suggester),
        type_problem,
        seed,
        init_pos,
        max_steps,
        num_trials,
        num_sims,
        r_per_trial_vec,
        steps_per_trial_vec,
        asks_per_trial_vec,
        r_vec,
        step_vec,
        num_asks_vec,
        b_sugg_type_vec,
        ν
    )
    
    print_sim_result(sim_result)
    
    return sim_result
end
