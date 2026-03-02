const Y_TEXT_OFFSET = 0.3

"""
    render(problem::TagMOMDPAT, step::NamedTuple; pre_act_text::String="")

Render a TagPOMDP step as a plot. If the step contains a belief, the belief will be plotted
using a color gradient of green for the belief of the target position and belief over the
robot position will be plotted as an orange robot with a faded robot representing smaller
belief. If the step contains a state, the robot and target will be plotted in their
respective positions. If the step contains an action, the action will be plotted in the
bottom center of the plot. `pre_act_text` can be used to add text before the action text.

- `problem::TagMOMDPAT`: The TagMOMDPAT to render
- `step::NamedTuple`: Step step to render with fields `b`, `s`, and `a`
- `pre_act_text::String`: Text to add before the action text
"""
function POMDPTools.render(problem::TagMOMDPAT, step::NamedTuple; 
    pre_act_text::String="",
    viz_types::Bool=true,
    text_below::Bool=true,
    plot_target::Bool=true,
    kwargs...
)
    plt = nothing
    plotted_robot = false

    if !isnothing(get(step, :b, nothing))
        plt = plot_tag(problem, step.b; viz_types=viz_types, text_below=text_below, kwargs...)
        plotted_robot = true
    else
        plt = plot_tag(problem; viz_types=viz_types, text_below=text_below, kwargs...)
    end

    if !isnothing(get(step, :s, nothing))
        r_pos = step.s[1].r_pos
        if r_pos == 0
            r_pos = step.s[2].t_pos
        end

        offset = (0.0, 0.0)
        if r_pos == step.s[2].t_pos
            offset = (0.0, 0.1)
        end
        t_x, t_y = get_prop(problem.mg, :node_pos_mapping)[step.s[2].t_pos]
        r_x, r_y = get_prop(problem.mg, :node_pos_mapping)[r_pos]
        if plot_target
            plt = plot_robot!(plt, (t_y, -t_x) .+ offset; color=RGB(0.8, 0.1, 0.1))
        end
        if !plotted_robot
            plt = plot_robot!(plt, (r_y, -r_x))
        end
        
        if viz_types
            # Display suggester type and number of asks remaining
            px_p_tick = px_per_tick(plt)
            fnt_size = Int(floor(px_p_tick / 5 / 1.3333333))
            num_cols = get_prop(problem.mg, :ncols)
            x_start = 0.4
            y_start = -0.75
            if !isnothing(get(step, :sugg_type, nothing))
                if typeof(step.sugg_type) <: Number
                    type_text = "Sugg λ: $(step.sugg_type)"
                else
                    type_text = step.sugg_type
                end
                plt = annotate!(plt, x_start, y_start, (text(type_text, "courier", fnt_size, :right, :vcenter, :black)))
            end
            
            if isnothing(get(step, :num_asks_remain, nothing))
                asks_remain_text = problem.num_asks == -1 ? "∞" : step.s[1].num_ask_remain
            else
                asks_remain_text = step.num_asks_remain
            end

            if asks_remain_text !== ""
                plt = annotate!(plt, x_start, y_start - Y_TEXT_OFFSET, (text("# Asks Rem: $(asks_remain_text)", "courier", fnt_size, :right, :vcenter, :black)))
            end
        end
    end

    if !isnothing(get(step, :a, nothing)) && text_below
        # Determine appropriate font size based on plot size
        px_p_tick = px_per_tick(plt)
        fnt_size = Int(floor(px_p_tick / 2 / 1.3333333))
        num_cols = get_prop(problem.mg, :ncols)
        num_rows = get_prop(problem.mg, :nrows)
        xc = 1.0
        yc = -(num_rows + 1.0)
        action_text = pre_act_text * "a = $(ACTION_NAMES[step.a])"
        plt = annotate!(plt, xc, yc, (text(action_text, fnt_size, :left, :vcenter, :black,)))
    end

    return plt
end

function plot_tag(problem::TagMOMDPAT; kwargs...)
    b = zeros(length(states(problem)))
    return plot_tag(problem, b, ordered_states(problem); kwargs...)
end
function plot_tag(problem::TagMOMDPAT, b::Vector{Float64}; kwargs...)
    @assert length(b) == length(states(problem)) "Belief must be the same length as the state list"
    return plot_tag(problem, b, ordered_states(problem); kwargs...)
end
function plot_tag(problem::TagMOMDPAT, b::DiscreteBelief; kwargs...)
    return plot_tag(problem, b.b, b.state_list; kwargs...)
end
function plot_tag(problem::TagMOMDPAT, b::SparseCat; kwargs...)
    state_vec = ordered_states(problem)
    bvec = zeros(length(state_vec))
    for (si, prob_i) in zip(b.vals, b.probs)
        bvec[stateindex(problem, si)] = prob_i
    end
    return plot_tag(problem, bvec, state_vec; kwargs...)
end

function plot_tag(problem::TagMOMDPAT, b::Vector, state_list::Vector{TagMOMDPState};
    color_grad=cgrad(:Greens_9),
    prob_color_scale=1.0,
    plt_size=(1000, 500),
    viz_types::Bool=true,
    text_below::Bool=true,
    plot_x_black::Bool=true
)
    @assert length(b) == length(state_list) "Belief and state list must be the same length"
    num_cells = get_prop(problem.mg, :num_grid_pos)
    node_pos_mapping = get_prop(problem.mg, :node_pos_mapping)

    map_str = map_str_from_metagraph(problem)
    map_str_mat = Matrix{Char}(undef, get_prop(problem.mg, :nrows), get_prop(problem.mg, :ncols))
    for (i, line) in enumerate(split(map_str, '\n'))
        map_str_mat[i, :] .= collect(line)
    end

    # Get the belief of the robot and the target in each cell
    grid_t_b = zeros(num_cells)
    grid_r_b = zeros(num_cells)
    for (ii, sᵢ) in enumerate(state_list)
        if sᵢ[1].r_pos == 0
            grid_r_b[sᵢ[2].t_pos] += b[ii] # At same location
        else
            grid_r_b[sᵢ[1].r_pos] += b[ii]    
        end
        grid_t_b[sᵢ[2].t_pos] += b[ii]
    end

    plt = plot(; legend=false, ticks=false, showaxis=false, grid=false, aspectratio=:equal, size=plt_size, trim=true)

    if text_below
        # Plot blank section at bottom for action text
        nc = get_prop(problem.mg, :ncols)
        nr = get_prop(problem.mg, :nrows)
        plt = plot!(plt, rect((nc+3.5)*0.5, 0.5, 0.5, 0.5, (nc+1)/2, -(nr+1.0)); linecolor=RGB(1.0, 1.0, 1.0), color=:white)
    end
    
    node_mapping = get_prop(problem.mg, :node_mapping)
    # Plot the grid
    for xi in 1:get_prop(problem.mg, :nrows)
        for yj in 1:get_prop(problem.mg, :ncols)
            if map_str_mat[xi, yj] == 'x'
                color = :black
                if !plot_x_black
                    continue
                end
            else
                cell_i = node_mapping[(xi, yj)]
                color_scale = grid_t_b[cell_i] * prob_color_scale
                if color_scale < 0.05
                    color = :white
                else
                    color = get(color_grad, color_scale)
                end
            end
            
            plt = plot!(plt, rect(0.5, 0.5, yj, -xi); color=color)
        end
    end
    # Determine scale of font based on plot size
    px_p_tick = px_per_tick(plt)
    fnt_size = Int(floor(px_p_tick / 4 / 1.3333333))

    # Plot the robot (tranparancy based on belief) and annotate the target belief as well
    for cell_i in 1:num_cells
        xi, yi = node_pos_mapping[cell_i]
        prob_text = round(grid_t_b[cell_i]; digits=2)
        if prob_text < 0.01
            prob_text = ""
        end
        plt = annotate!(yi, -xi, (text(prob_text, :black, :center, fnt_size)))
        if grid_r_b[cell_i] >= 1/num_cells - 1e-5
            plt = plot_robot!(plt, (yi, -xi); fillalpha=grid_r_b[cell_i])
        end
    end
    
    if viz_types
        # First let's get the marginal belief of the suggester type
        b_sugg_type = zeros(length(problem.types))
        for (ii, sᵢ) in enumerate(state_list)
            type_i = findfirst(isequal(sᵢ[2].sugg_type), problem.types)
            b_sugg_type[type_i] += b[ii]
        end
        b_sugg_type = b_sugg_type ./ sum(b_sugg_type)
        
        
        # Position on the left side of the plot
        x_left = -1.5
        y_top = -0.75 - 4 * Y_TEXT_OFFSET
        bar_width = 1.0
        bar_height = 0.25
        
        # Title for the belief section
        fnt_size_small = Int(floor(px_p_tick / 5 / 1.3333333))
        plt = annotate!(plt, x_left, y_top, (text("b(λ)", fnt_size_small, :left, :vcenter, :black)))
        
        # Plot bars for each type
        fnt_size_small = Int(floor(px_p_tick / 6 / 1.3333333))
        for (i, (type_val, belief_val)) in enumerate(zip(problem.types, b_sugg_type))
            y_bar = y_top - (i * (Y_TEXT_OFFSET + 0.05)) - 0.1
            
            # Background bar (white/light gray)
            plt = plot!(plt, rect(0.0, bar_width, bar_height/2, bar_height/2, x_left + 0.75, y_bar); 
                        color=:lightgray, linecolor=:black, linewidth=1)
            
            # Filled portion based on belief
            if belief_val > 0.01
                fill_width = bar_width * belief_val
                plt = plot!(plt, rect(0.0, fill_width, bar_height/2, bar_height/2, x_left + 0.75, y_bar); 
                            color=:darkgreen, linecolor=:black, linewidth=1, alpha=0.8)
                
                # Percentage text overlayed on the bar
                fnt_pct = Int(floor(px_p_tick / 6 / 1.3333333))
                percentage_text = "$(round(belief_val, digits=3))"
                plt = annotate!(plt, x_left + bar_width/2 + 0.75, y_bar, 
                                (text(percentage_text, fnt_pct, :center, :vcenter, :black)))
            end
            
            # Type label to the right of the bar
            plt = annotate!(plt, x_left, y_bar, 
                            (text("λ=$(type_val)", fnt_size_small, :left, :vcenter, :black)))
        end
        
        # Below the bars, add \mathbb{E}[λ]
        num_bars = length(problem.types)
        y_exp_lambda = y_top - ((num_bars + 1) * (Y_TEXT_OFFSET + 0.05)) - 0.1
        plt = annotate!(plt, x_left, y_exp_lambda, (text("E[λ] = $(round(sum(problem.types .* b_sugg_type), digits=2))", fnt_size_small, :left, :vcenter, :black)))
    end
    
    return plt
end

function plot_robot!(plt::Plots.Plot, (x, y); fillalpha=1.0, color=RGB(1.0, 0.627, 0.0))
    body_size = 0.3
    la = 0.1
    lb = body_size
    leg_offset = 0.3
    plot!(plt, ellip(x + leg_offset, y, la, lb); color=color, fillalpha=fillalpha)
    plot!(plt, ellip(x - leg_offset, y, la, lb); color=color, fillalpha=fillalpha)
    plot!(plt, circ(x, y, body_size); color=color, fillalpha=fillalpha)
    return plt
end

function rect(w, h, x, y)
    return rect(w, w, h, h, x, y)
end

function rect(wl, wr, ht, hb, x, y)
    return Shape(x .+ [wr, -wl, -wl, wr, wr], y .+ [ht, ht, -hb, -hb, ht])
end

function circ(x, y, r; kwargs...)
    return ellip(x, y, r, r; kwargs...)
end

function ellip(x, y, a, b; num_pts=25)
    angles = [range(0; stop=2π, length=num_pts); 0]
    xs = a .* sin.(angles) .+ x
    ys = b .* cos.(angles) .+ y
    return Shape(xs, ys)
end

function px_per_tick(plt)
    (x_size, y_size) = plt[:size]
    xlim = xlims(plt)
    ylim = ylims(plt)
    xlim_s = xlim[2] - xlim[1]
    ylim_s = ylim[2] - ylim[1]
    if xlim_s >= ylim_s
        px_p_tick = x_size / xlim_s
    else
        px_p_tick = y_size / ylim_s
    end
    return px_p_tick
end
