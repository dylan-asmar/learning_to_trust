
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

include("constants.jl")
include("utils.jl")
include("suggesters.jl")
include("plot_results.jl")

function run_sim(
    problem::Symbol;
    no_ask_problem::Symbol=:invalid,
    num_trials::Int=1,
    max_steps::Int=50,
    num_sims::Int=1,
    verbose::Bool=false,
    visualize::Bool=false,
    agent::Symbol=:normal,
    max_suggestions_per_trial=Inf,
    init_rocks=nothing,
    init_pos=nothing,
    seed=42,
    suggester::AbstractSuggester=NoSuggester(),
)


    problem in RS_PROBS || problem in TG_PROBS || error("Invalid problem: $problem")
    agent in AGENTS || error("Invalid agent: $agent")

    momdp, π, load_str = get_problem_and_policy(problem);
    momdp_no_ask, π0, _ = get_problem_and_policy(no_ask_problem);
    
    temp_λ = momdp.types[1] # This doesn't affect the simulation, but is needed for part of the y state
    
    state_list_x = ordered_states_x(momdp)
    state_list_y = ordered_states_y(momdp)
    state_list_y_no = ordered_states_y(momdp_no_ask)
    nx = length(state_list_x)
    ny = length(state_list_y)
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
        policy_no_ask = deepcopy(π0)

        belief_updater_agent = MOMDPDiscreteUpdater(momdp)

        sugg_thread = deepcopy(suggester)
        
        # Get iniital state
        sᵢ = rand(rng, initialstate(momdp))
        if !isnothing(init_pos)
            if problem in RS_PROBS
                num_asks = momdp.num_asks == -1 ? 1 : momdp.num_asks
                xi = RockSampleMOMDPXState(init_pos, num_asks)
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
                num_asks = momdp.num_asks == -1 ? 1 : momdp.num_asks
                xi = RockSampleMOMDPXState(momdp.init_pos, num_asks)
                yi = RockSampleMOMDPYState(sᵢ[2].rocks, temp_λ)
                sᵢ = (xi, yi)
            end
        end
        
        if problem in RS_PROBS && !isnothing(init_rocks)
            # If we are in a RockSample problem and we provided initial rocks
            length(init_rocks) == num_rocks || error("Invalid init_rocks: $init_rocks")
            xi = sᵢ[1]
            yi_rocks = Tuple(init_rocks)
            yi = RockSampleMOMDPYState(yi_rocks, init_λ)
            sᵢ = (xi, yi)
        end
        
        bᵢ = beliefvec_y(momdp, ny, initialstate_y(momdp, sᵢ[1]))
        
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
                
                if suggestion_cnt >= max_suggestions_per_trial                    
                    a_n = no_ask_action(momdp, state_list_y, momdp_no_ask, state_list_y_no, policy_no_ask, bᵢ, sᵢ)
                else
                    a_n = action(policy_agent, bᵢ, sᵢ[1])
                    # av = actionvalues(policy_agent, SparseCat(state_list_y, bᵢ), sᵢ[1])
                    # a_n = argmax(av)
                    if a_n == 6 && momdp.num_asks != 0
                        suggestion_cnt += 1
                        ask_trial_cnt += 1
                    end
                end
                
                Qsa = Q[stateindex_x(momdp, sᵢ[1]), stateindex_y(momdp, sᵢ[2]), :]
                a_p = argmax(Qsa)

                # a is exectued action. Select based on agent type
                if agent == :normal
                    a = a_n
                elseif agent == :perfect
                    a = a_p
                elseif agent == :random
                    a = rand(rng, actions(pomdp))
                end
                
                o = nothing
                if a == 6 && momdp.num_asks != 0
                    num_ask += 1
                    (sp, r) = @gen(:sp, :r)(momdp, sᵢ, a, rng)
                    suggestion = get_suggestion(sugg_thread, momdp, sp, rng)
                    
                    if suggestion == -1
                        # No suggestion, so we must use the no-ask policy and get new sp, o, r
                        a = no_ask_action(momdp, state_list_y, momdp_no_ask, state_list_y_no, policy_no_ask, bᵢ, sᵢ)
                        (sp, o, r) = @gen(:sp, :o, :r)(momdp, sᵢ, a, rng)
                        
                        #? Should we penalize the reward for asking here?
                    else
                        o = suggestion_to_observation(momdp, suggestion)
                    end
                    
                else
                    # Simulate a step forward with action `a` from state `sᵢ`
                    (sp, o, r) = @gen(:sp, :o, :r)(momdp, sᵢ, a, rng)
                end

                if visualize
                    state_list_f = [(sᵢ[1], yi) for yi in state_list_y]
                    
                    if momdp.num_asks == -1
                        num_asks_state = Inf
                    else
                        num_asks_state = sᵢ[1].num_ask_remain
                    end
                    if isinf(max_suggestions_per_trial)
                        num_asks_restriction = Inf
                    else
                        num_asks_restriction = max_suggestions_per_trial - suggestion_cnt
                    end
                    if isinf(num_asks_restriction) && isinf(num_asks_state)
                        num_asks_remaining = Inf
                    else
                        num_asks_remaining = min(num_asks_state, num_asks_restriction)
                    end
                    
                    sugg_type_txt = sugg_thread.λ
                    if sugg_thread isa RuleSuggester
                        sugg_type_txt = "Heuristic"
                    end
                    
                    step = (s=sᵢ, a=a_n, b=SparseCat(state_list_f, bᵢ), num_asks_remain=num_asks_remaining, sugg_type=sugg_type_txt)
                    if num_trials > 1
                        pre_act_text = "Trial: $(trial_cnt+1), t=$t, o=$o, "
                    else
                        pre_act_text = "t=$t, o=$o, "
                    end
                    display(render(momdp, step; pre_act_text=pre_act_text))
                end
                
                if verbose
                    println("--------------------------------")
                    println("Time                     : $t")
                    println("State                    : $sᵢ")
                    println("Initial Action:          : $a_n")
                    # if agent in [:naive, :scaled, :noisy]
                        # println("Suggested Action         : $suggestion")
                    # end
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
                
                # Update agent's belief with observation from environment
                bᵢ′ = update(belief_updater_agent, bᵢ, a, o, sᵢ[1], sp[1])
                bᵢ = beliefvec_y(momdp, ny, bᵢ′)
                
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
                    b_sugg_type = zeros(length(momdp.types))
                    for (ii, yi) in enumerate(state_list_y)
                        type_i = findfirst(isequal(yi.sugg_type), momdp.types)
                        b_sugg_type[type_i] += bᵢ[ii]
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
            println("No Ask Problem: $no_ask_problem")
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

    sim_result = SimResult(
        agent,
        suggester.λ,
        problem,
        typeof(suggester),
        no_ask_problem,
        max_suggestions_per_trial,
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
        b_sugg_type_vec
    )
    
    print_sim_result(sim_result)
    
    return sim_result
end
