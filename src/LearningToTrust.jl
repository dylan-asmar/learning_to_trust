# transition_y is extended in override_tag_tx.jl; opt out of precompile.
__precompile__(false)

module LearningToTrust

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
using Cairo
using Fontconfig
using Colors
using Plots
using Graphs
using MetaGraphs
using SARSOP
using StaticArrays
using Measures
using LaTeXStrings
using PGFPlotsX
using SparseArrays

include("load_sources.jl")

# Problem / agent registry
export AGENTS, TG_PROBS, TG_ASK_PROBS, RS_PROBS, RS_ASK_PROBS

# Suggesters
export AbstractSuggester, PolicySuggester, RuleSuggester, RandomSuggester, NoSuggester
export get_suggestion, tag_regional_heuristic_normal_map

# Results and utilities
export Result_Type, SimResult, SimResultTypeEval
export get_problem_and_policy, get_problem, get_stats, print_sim_result
export no_ask_action, action_known_state, action_map, suggestion_to_observation

# Simulations
export run_sim, run_sim_type_eval, run_sim_dynamic_suggester

# Plotting
export plot_sim_result, plot_sim_result!
export plot_sim_result_reward_only, plot_sim_result_reward_only!
export plot_sim_result_exp_only, plot_sim_result_exp_only!
export plot_static_line!

# Policy generation
export generate_problem_and_policy, generate_and_save_Q

end
