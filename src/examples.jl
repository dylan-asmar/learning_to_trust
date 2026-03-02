
# constants.jl contains the problems

# utils.jl has a bunch of helper functions. but it also has get_problem which 
# defines the MOMDPs for each problem.



# Generate policy and action value function (Q) for a problem
# Uses pol_generator.jl
prob = :tag_1_0_1_2_5_10_w05
generate_problem_and_policy(prob; timeout=1200.0)
generate_and_save_Q(prob)

# Precomputed policies and action value functions are provided in the policies folder. You 
# can retrieve them using get_problem_and_policy.


# suggesters.jl contains the different types of suggesters and is setup to allow flexibility
# to define a plethora of different suggesters.
# PolicySuggester uses a policy and a parameter λ and perfrorms a noisy softmax sampling
# over the action values.
# RandomSuggester returns a random action
# NoSuggester returns -1
# RuleSuggester allows you to define a custum rule function.
# To define a new suggester, you can extend the AbstractSuggester type and implement 
# the get_suggestion function for your new type. You can also use your own rule by passing
# a function to the RuleSuggester constructor.



# To run a simulation, use run_sims.jl and `run_sim` function.
# The first argument is the problem defined in get_problem in utils.jl
# The rest are keyword arguments.
#- no_ask_problem: The problem to use when the agent does not ask for a suggestion.
#- num_trials: The number of trials to run for each simulation.
#- max_steps: The maximum number of steps to run for each trial.
#- num_sims: The number of simulations to run.
#- verbose: Whether to print verbose output.
#- visualize: Whether to visualize the environment.
#- agent: The agent to use for the simulation. Only :normal, :perfect, and :random are supported.
#- max_suggestions_per_trial: The maximum number of suggestions to receive per trial.
#- init_rocks: The initial rocks for the RockSample problem, if desired.
#- init_pos: The initial positions for agent in either problem, if desired.
#- seed: The seed for the random number generator.
#- suggester: The suggester to use for the simulation. Default is NoSuggester.

# An example is 
_, π_sugg, _ = get_problem_and_policy(:tag);
λ_sugg = 3.0;
suggester = PolicySuggester(π_sugg, λ_sugg);

sim_result = run_sim(
    :tag_inf_5, # Infinite number of asks and we assume a single suggester type of λ=5.0 (noisy rational)
    no_ask_problem=:tag,
    suggester=suggester,
    num_sims=1,
    num_trials=10,
    max_steps=500,
    agent=:normal,
    verbose=true,
    visualize=true,
    seed=45,
);


# Another simulation function is run_sim_type_eval in run_sims_type_eval.jl
# This is the function used to generate results for Table 1 in the paper. This assumes a 
# static suggester type. If the agent is normal, we can model noisy suggesters by passing
# the appropraite "type_problem" argument. For example, for the tag problem if we want to 
# simulate an agent that assumes the suggester is noisy with λ=5.0, but the actual suggester
# is λ=3.0, we would use parameters like:


prob = :tag
prob_type = :tag_inf_5
agent = :normal
suggester_λ = 3.0
ν = 1.0 # not used for normal agent

_, π_sugg, _ = get_problem_and_policy(prob)
suggester = PolicySuggester(π_sugg, suggester_λ)

sim_result = run_sim_type_eval(
    prob, 
    prob_type; 
    num_trials=50, 
    max_steps=500, 
    num_sims=200, 
    agent=agent, 
    seed=45, 
    suggester=suggester, 
    ν=ν
)

# The ν parameter is used for the naive agent.


# For simulating a dynamic suggester, we need another simulation that allows the suggester 
# to change at different time steps. We use `run_sim_dynamic_suggester` in 
# run_sims_type_eval_dynamic.jl. The primary difference in this case in the `suggesters` 
# keyword argument. This is a dictionary of suggester types keyed by the time step. Here is
# is an example to plots similar to Figure 1 and Figure 2 in the paper.


_, π_sugg, _ = get_problem_and_policy(:tag);

suggesters = Dict(
    0 => PolicySuggester(π_sugg, 3.0),
    15 => PolicySuggester(π_sugg, 5.0),
    25 => PolicySuggester(π_sugg, 1.0),
    40 => PolicySuggester(π_sugg, 15.0),
    50 => PolicySuggester(π_sugg, 2.0),
    60 => PolicySuggester(π_sugg, 0.0),
    75 => PolicySuggester(π_sugg, 10.0)
)

# Get sorted keys of suggesters
sorted_keys = sort(collect(keys(suggesters)))

num_sims = 100
num_trials = 100

sr1 = run_sim(
    :tag,
    :tag_inf_0_1_2_5_10;
    num_trials=num_trials,
    max_steps=20000,
    num_sims=num_sims,
    agent=:normal,
    seed=45,
    suggesters=suggesters
);

sr2 = run_sim(
    :tag,
    :tag_inf_0_1_2_5_10_w05;
    num_trials=num_trials,
    max_steps=20000,
    num_sims=num_sims,
    agent=:normal,
    seed=45,
    suggesters=suggesters
);

colors = Dict(
    :normal => RGB(0.97, 0.85, 0.0942), # Yellow/Gold
    :perfect => RGB(0.8, 0.5, 0.1), # Orange
    :random => RGB(0.35, 0.35, 0.35), # Gray
    :naive_1 => RGB(0.0, 0.6658, 0.681), # Teal/Cyan
    :naive_025 => RGB(0.4, 0.0, 1.0), # Purple
    :noisy_5 => RGB(0.1, 0.6433, 0.1), # Green
    :noisy_1 => RGB(0.0, 0.4056, 0.9787), # Blue
    :t012510_00 => RGB(0.8889, 0.2, 0.2), # Red
    :t012510_05 => RGB(0.78, 0.15, 0.8243) # Magenta
)

plt = plot_sim_result(sr1, [0.0, 1.0, 2.0, 5.0, 10.0]; label="T = {0, 1, 2, 5, 10}, t_p = 0.00", color=colors[:t012510_00])
plt = plot_sim_result!(plt, sr2, [0.0, 1.0, 2.0, 5.0, 10.0]; label="T = {0, 1, 2, 5, 10}, t_p = 0.05", color=colors[:t012510_05])

plt_rw = plot_sim_result_reward_only(sr1; label="T = {0, 1, 2, 5, 10}, t_p = 0.00", color=colors[:t012510_00])
plt_rw = plot_sim_result_reward_only!(plt_rw, sr2; label="T = {0, 1, 2, 5, 10}, t_p = 0.05", color=colors[:t012510_05])

plt_rw = plot!(plt_rw, 
    xlims=(1, num_trials),
    xticks=[1; 5:5:100]
)

plt_exp = plot_sim_result_exp_only(sr1, [0.0, 1.0, 2.0, 5.0, 10.0]; label="T = {0, 1, 2, 5, 10}, t_p = 0.00", color=colors[:t012510_00])
plt_exp = plot_sim_result_exp_only!(plt_exp, sr2, [0.0, 1.0, 2.0, 5.0, 10.0]; label="T = {0, 1, 2, 5, 10}, t_p = 0.05", color=colors[:t012510_05])


plt_exp = plot!(plt_exp, 
    xlims=(1, num_trials),
    xticks=[1; 5:5:100]
)
