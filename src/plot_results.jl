using LinearAlgebra
using StatsBase
using Random
using Printf
using JLD2
using ProgressMeter
using Measures
using Plots

include("utils.jl")


function plot_sim_result(plt_sim_result::Result_Type, type_vals::Vector{Float64}; 
        label=nothing, exp_plot=true, plt_trials=nothing, color=nothing
    )
    if exp_plot
        # Create reward plot
        p1 = plot_sim_result_reward_only(plt_sim_result; label=label, plt_trials=plt_trials, color=color)
        
        # Create expected type plot
        p2 = plot_sim_result_exp_only(plt_sim_result, type_vals; label=label, plt_trials=plt_trials, color=color)
        
        plt = plot(p1, p2, layout=(2, 1), size=(1200, 1200))
    else
        # Only plot reward
        plt = plot_sim_result_reward_only(plt_sim_result; label=label, plt_trials=plt_trials, color=color)
    end
    
    return plt
end

function plot_sim_result!(plt::Plots.Plot, plt_sim_result::Result_Type, type_vals::Vector{Float64}; 
        label=nothing, exp_plot=true, plt_trials=nothing, color=nothing
    )
    if exp_plot
        # Add reward plot to first subplot
        plot_sim_result_reward_only!(plt.subplots[1], plt_sim_result; label=label, plt_trials=plt_trials, color=color)
        
        # Add expected type plot to second subplot
        plot_sim_result_exp_only!(plt.subplots[2], plt_sim_result, type_vals; label=label, plt_trials=plt_trials, color=color)
    else
        # Only add reward plot to the main plot
        plot_sim_result_reward_only!(plt, plt_sim_result; label=label, plt_trials=plt_trials, color=color)
    end
    
    return plt
end


function plot_sim_result_reward_only(plt_sim_result::Result_Type; 
    label=nothing, plt_trials=nothing, color=nothing
)
    per_trial_results = [zeros(plt_sim_result.num_sims) for _ in 1:plt_sim_result.num_trials_per_sim]
    for i in 1:plt_sim_result.num_trials_per_sim
        for j in 1:plt_sim_result.num_sims
            per_trial_results[i][j] = plt_sim_result.reward_per_trial_per_sim[j][i]
        end
    end

    mean_per_trial_results = [mean(per_trial_results[i]) for i in 1:plt_sim_result.num_trials_per_sim]
    std_err_per_trial_results = [std(per_trial_results[i]) / sqrt(plt_sim_result.num_sims) for i in 1:plt_sim_result.num_trials_per_sim]

    if isnothing(plt_trials)
        plt_trials = 1:length(mean_per_trial_results)
    end

    plt = plot(plt_trials, mean_per_trial_results[plt_trials];
        ribbon=std_err_per_trial_results[plt_trials] .* 1.96, 
        label=label, 
        color=color,
        title="Mean Reward vs Trial Number", 
        xlabel="Trial Number", 
        ylabel="Mean Reward",
        legend=:outertopright,
        margin = 10mm,
        size=(1200, 600)
    )

    return plt
end


function plot_sim_result_reward_only!(plt::Plots.Plot, plt_sim_result::Result_Type; 
    label=nothing, plt_trials=nothing, color=nothing
)
    per_trial_results = [zeros(plt_sim_result.num_sims) for _ in 1:plt_sim_result.num_trials_per_sim]

    for i in 1:plt_sim_result.num_trials_per_sim
        for j in 1:plt_sim_result.num_sims
            per_trial_results[i][j] = plt_sim_result.reward_per_trial_per_sim[j][i]
        end
    end

    mean_per_trial_results = [mean(per_trial_results[i]) for i in 1:plt_sim_result.num_trials_per_sim]
    std_err_per_trial_results = [std(per_trial_results[i]) / sqrt(plt_sim_result.num_sims) for i in 1:plt_sim_result.num_trials_per_sim]

    if isnothing(plt_trials)
        plt_trials = 1:length(mean_per_trial_results)
    end

    plot!(plt, plt_trials, mean_per_trial_results[plt_trials], ribbon=std_err_per_trial_results[plt_trials] .* 1.96, label=label, color=color)

    return plt
end

function plot_sim_result_reward_only!(plt::Plots.Subplot, plt_sim_result::Result_Type; 
    label=nothing, plt_trials=nothing, color=nothing
)
    per_trial_results = [zeros(plt_sim_result.num_sims) for _ in 1:plt_sim_result.num_trials_per_sim]

    for i in 1:plt_sim_result.num_trials_per_sim
        for j in 1:plt_sim_result.num_sims
            per_trial_results[i][j] = plt_sim_result.reward_per_trial_per_sim[j][i]
        end
    end

    mean_per_trial_results = [mean(per_trial_results[i]) for i in 1:plt_sim_result.num_trials_per_sim]
    std_err_per_trial_results = [std(per_trial_results[i]) / sqrt(plt_sim_result.num_sims) for i in 1:plt_sim_result.num_trials_per_sim]

    if isnothing(plt_trials)
        plt_trials = 1:length(mean_per_trial_results)
    end

    plot!(plt, plt_trials, mean_per_trial_results[plt_trials], ribbon=std_err_per_trial_results[plt_trials] .* 1.96, label=label, color=color)

    return plt
end


function plot_sim_result_exp_only(plt_sim_result::Result_Type, type_vals::Vector{Float64}; 
    label=nothing, plt_trials=nothing, color=nothing
)
    per_trial_sugg_type_exp = [zeros(plt_sim_result.num_sims) for _ in 1:plt_sim_result.num_trials_per_sim]
    for i in 1:plt_sim_result.num_trials_per_sim
        for j in 1:plt_sim_result.num_sims
            per_trial_sugg_type_exp[i][j] = dot(plt_sim_result.b_sugg_type_per_trial_per_sim[j][i], type_vals)
        end
    end

    if isnothing(plt_trials)
        plt_trials = 1:plt_sim_result.num_trials_per_sim
    end

    exp_label = label
    mean_per_trial_sugg_type_exp = [mean(per_trial_sugg_type_exp[i]) for i in 1:plt_sim_result.num_trials_per_sim]
    std_err_per_trial_sugg_type_exp = [std(per_trial_sugg_type_exp[i]) / sqrt(plt_sim_result.num_sims) for i in 1:plt_sim_result.num_trials_per_sim]

    type_min = min(minimum(type_vals), minimum(mean_per_trial_sugg_type_exp))
    max_upper_bound = mean_per_trial_sugg_type_exp .+ std_err_per_trial_sugg_type_exp .* 1.96
    type_max = max(maximum(type_vals), maximum(max_upper_bound))
    plt = plot(plt_trials, mean_per_trial_sugg_type_exp[plt_trials];
        ribbon=std_err_per_trial_sugg_type_exp[plt_trials] .* 1.96, 
        label=exp_label, 
        color=color,
        title="Expected Suggester Type vs Trial Number", 
        xlabel="Trial Number", 
        ylabel="Mean Expected Suggester Type",
        # ylims=(type_min, type_max),
        legend=:outertopright,
        margin = 10mm,
        size=(1200, 600)
    )
    return plt
end

function plot_sim_result_exp_only!(plt::Plots.Plot, plt_sim_result::Result_Type, type_vals::Vector{Float64}; 
    label=nothing, plt_trials=nothing, color=nothing
)
    per_trial_sugg_type_exp = [zeros(plt_sim_result.num_sims) for _ in 1:plt_sim_result.num_trials_per_sim]
    for i in 1:plt_sim_result.num_trials_per_sim
        for j in 1:plt_sim_result.num_sims
            per_trial_sugg_type_exp[i][j] = dot(plt_sim_result.b_sugg_type_per_trial_per_sim[j][i], type_vals)
        end
    end

    if isnothing(plt_trials)
        plt_trials = 1:plt_sim_result.num_trials_per_sim
    end

    exp_label = label
    mean_per_trial_sugg_type_exp = [mean(per_trial_sugg_type_exp[i]) for i in 1:plt_sim_result.num_trials_per_sim]
    std_err_per_trial_sugg_type_exp = [std(per_trial_sugg_type_exp[i]) / sqrt(plt_sim_result.num_sims) for i in 1:plt_sim_result.num_trials_per_sim]

    plot!(plt, plt_trials, mean_per_trial_sugg_type_exp[plt_trials], 
        ribbon=std_err_per_trial_sugg_type_exp[plt_trials] .* 1.96, 
        label=exp_label,
        color=color)

    return plt
end

function plot_sim_result_exp_only!(plt::Plots.Subplot, plt_sim_result::Result_Type, type_vals::Vector{Float64}; 
    label=nothing, plt_trials=nothing, color=nothing
)
    per_trial_sugg_type_exp = [zeros(plt_sim_result.num_sims) for _ in 1:plt_sim_result.num_trials_per_sim]
    for i in 1:plt_sim_result.num_trials_per_sim
        for j in 1:plt_sim_result.num_sims
            per_trial_sugg_type_exp[i][j] = dot(plt_sim_result.b_sugg_type_per_trial_per_sim[j][i], type_vals)
        end
    end

    if isnothing(plt_trials)
        plt_trials = 1:plt_sim_result.num_trials_per_sim
    end

    exp_label = label
    mean_per_trial_sugg_type_exp = [mean(per_trial_sugg_type_exp[i]) for i in 1:plt_sim_result.num_trials_per_sim]
    std_err_per_trial_sugg_type_exp = [std(per_trial_sugg_type_exp[i]) / sqrt(plt_sim_result.num_sims) for i in 1:plt_sim_result.num_trials_per_sim]

    plot!(plt, plt_trials, mean_per_trial_sugg_type_exp[plt_trials], 
        ribbon=std_err_per_trial_sugg_type_exp[plt_trials] .* 1.96, 
        label=exp_label,
        color=color)

    return plt
end


function plot_static_line!(plt::Plots.Plot, rew::Float64, ci::Float64; 
    label=nothing, start_trial=1, end_trial=1, color=nothing
)
    num_plot_trials = end_trial - start_trial + 1
    ys = fill(rew, num_plot_trials)
    rs = fill(ci, num_plot_trials)
    plot!(plt, start_trial:end_trial, ys, ribbon=rs, label=label, color=color)
end

function plot_static_line!(plt::Plots.Subplot, rew::Float64, ci::Float64; 
    label=nothing, start_trial=1, end_trial=1, color=nothing
)
    num_plot_trials = end_trial - start_trial + 1
    ys = fill(rew, num_plot_trials)
    rs = fill(ci, num_plot_trials)
    plot!(plt, start_trial:end_trial, ys, ribbon=rs, label=label, color=color)
end
