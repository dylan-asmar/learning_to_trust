
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


SAVE_AS_TEX = true
# plot_save_str = "tag_step_4_2"
plot_save_str = "tag_naive_4"

# agent = :normal
agent = :naive

problem = :tag
type_problem = :tag_inf_2
max_steps = 10

init_pos = (12,10)
seed = 42
ν = 1.0

update_steps = [4]

_, π_sugg, _ = get_problem_and_policy(problem);
suggester=PolicySuggester(π_sugg, 1000.0);


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



# Action value array for the agent where it is Q(x_idx, y_idx, a_idx)
Q_str = load_str * "_Q.jld2"
@load(Q_str, Q)



if SAVE_AS_TEX
    pgfplotsx()
else
    gr()
end



rng = MersenneTwister(seed)

policy_agent = deepcopy(π);

belief_updater_agent = MOMDPDiscreteUpdater(momdp);
belief_updater_types = MOMDPDiscreteUpdater(momdp_types);

sugg_thread = deepcopy(suggester);

# Get iniital state
sᵢ = rand(rng, initialstate(momdp))
if !isnothing(init_pos)
    num_asks = momdp.num_asks == -1 ? 1 : momdp.num_asks
    xi = TagMOMDPXState(init_pos[1], num_asks)
    yi = TagMOMDPYState(init_pos[2], temp_λ)
    sᵢ = (xi, yi)
end


bᵢ = beliefvec_y(momdp, ny, initialstate_y(momdp, sᵢ[1]))
bᵢ_types = beliefvec_y(momdp_types, ny_types, initialstate_y(momdp_types, sᵢ[1]))


step_cnt = 0
t = 0
for _ in 1:max_steps
    t += 1
    step_cnt += 1
    bₒ = bᵢ # Original belief before any updates
    bₒ_types = bᵢ_types
    
    
    a_n = action(policy_agent, bᵢ, sᵢ[1])
    # av = actionvalues(policy_agent, belief_sparse(bᵢ, state_list_y), sᵢ[1])
    # a_n = argmax(av)
    
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
            if rand(rng) <= ν && t in update_steps
                a = suggestion
            else
                a = a_n
            end
        elseif agent == :normal && t in update_steps
            @info """Time: $t, 
            Robot position: $(sᵢ[1])
            Target position: $(sᵢ[2])
            Suggestion: $suggestion
            """
            os = suggestion_to_observation(momdp, suggestion)
            
            
            x_state = convert_state_to_inf_ask(momdp, sᵢ[1])
            state_list_f = [(x_state, yi) for yi in state_list_y_types]
                
            sugg_type_txt = sugg_thread.λ
            if sugg_thread isa RuleSuggester
                sugg_type_txt = "Heuristic"
            end
            
            at = action_map(momdp, momdp_types, a_n)
            step = (s=sᵢ, a=at, b=SparseCat(state_list_f, bᵢ_types), sugg_type=sugg_type_txt, num_asks_remain="")
            # display(render(momdp_types, step))
            plt = render(momdp_types, step; text_below=false, viz_types=false, plot_target=false, plot_x_black=false)
            
            if SAVE_AS_TEX
                savefig(plt, "plots/$(plot_save_str)_$(step_cnt)_UPDATE.tex")
            else
                display(plt)    
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
        elseif agent == :normal
            # Do nothing, not an udpate step
        else
            error("Invalid agent: $agent")
        end
    end
    
    # Simulate a step forward with action `a` from state `sᵢ`
    (sp, o, r) = @gen(:sp, :o, :r)(momdp, sᵢ, a, rng)

    #? Visualize the step
    x_state = convert_state_to_inf_ask(momdp, sᵢ[1])
    state_list_f = [(x_state, yi) for yi in state_list_y_types]
        
    sugg_type_txt = sugg_thread.λ
    if sugg_thread isa RuleSuggester
        sugg_type_txt = "Heuristic"
    end
    
    at = action_map(momdp, momdp_types, a)
    step = (s=sᵢ, a=at, b=SparseCat(state_list_f, bᵢ_types), sugg_type=sugg_type_txt, num_asks_remain="")
    
    # display(render(momdp_types, step))
    plt = render(momdp_types, step; text_below=false, viz_types=false, plot_target=false, plot_x_black=false)
    
    
    if SAVE_AS_TEX
        savefig(plt, "plots/$(plot_save_str)_$(step_cnt).tex")
    else
        display(plt)    
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

    
    
end
