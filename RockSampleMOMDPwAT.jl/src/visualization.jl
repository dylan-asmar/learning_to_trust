function POMDPTools.render(momdp::RockSampleMOMDPAT, step;
    viz_rock_state=true,
    viz_belief=true,
    viz_types=true,
    pre_act_text="",
    num_asks_remain=nothing, 
    sugg_type=nothing,
    text_below=true,
)
        

    x_add = viz_types ? 3 : 1
    y_add = text_below ? 1 : 0

    nx, ny = momdp.map_size[1] + x_add, momdp.map_size[2] + y_add
    
    size_scale = 25cm
    max_n = max(nx, ny)
    set_default_graphic_size(size_scale * nx / max_n, size_scale * ny / max_n)
    
    
    cells = []
    for x in x_add:nx-1, y in 1:ny-y_add
        ctx = cell_ctx((x, y - 1 + y_add), (nx, ny))
        cell = compose(ctx, rectangle(), fill("white"))
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.15mm), stroke("black"), cells...)

    rocks = []
    for (i, (rx, ry)) in enumerate(momdp.rocks_positions)
        ctx = cell_ctx((rx + x_add - 1, ry + y_add - 1), (nx, ny))
        clr = "black"
        if viz_rock_state && get(step, :s, nothing) !== nothing
            clr = step[:s][2].rocks[i] ? "green" : "red"
        end
        rock = compose(ctx, ngon(0.5, 0.5, 0.3, 6), stroke(clr), fill("gray"), linewidth(0.15mm))
        push!(rocks, rock)
    end
    rocks = compose(context(), rocks...)
    exit_area = render_exit((nx, ny), y_add)
    
    upper_left_white_rect = nothing
    if viz_types
        upper_left_white_rect = compose(context(0.0, 0.0, 2 / nx, 1.0), rectangle(), fill("white"), stroke("white"))
    end
    
    agent = nothing
    action = nothing
    type_state = nothing
    if get(step, :s, nothing) !== nothing
        agent_ctx = cell_ctx(step[:s][1].pos .+ (x_add - 1, y_add - 1), (nx, ny))
        agent = render_agent(agent_ctx)
        if !isnothing(get(step, :a, nothing))
            action = render_action(momdp, step, x_add, y_add)
        end
        if viz_types
            type_state = render_type_info(momdp, step)
        end
    end
    
    action_text = nothing
    if text_below
        action_text = render_action_text(momdp, step, pre_act_text, x_add)
    end
    
    belief = nothing
    if viz_belief && (get(step, :b, nothing) !== nothing)
        belief = render_belief(momdp, step, x_add, y_add)
    end
    
    type_belief = nothing 
    if viz_types && !isnothing(get(step, :b, nothing)) # && length(momdp.types) > 1
        type_belief = render_type_belief(momdp, step)
    end
    
    return compose(context(), action, agent, action_text, belief, rocks, grid, exit_area, type_state, type_belief, upper_left_white_rect)
    # return compose(context(), grid)
    
end


function cell_ctx(xy, size)
    nx, ny = size
    x, y = xy
    return context((x - 1) / nx, (ny - y - 1) / ny, 1 / nx, 1 / ny)
end


function render_belief(momdp::RockSampleMOMDPAT, step, x_add, y_add)
    rock_beliefs = get_rock_beliefs(momdp, get(step, :b, nothing))
    nx, ny = momdp.map_size[1] + x_add, momdp.map_size[2] + y_add
    belief_outlines = []
    belief_fills = []
    belief_texts = []
    for (i, (rx, ry)) in enumerate(momdp.rocks_positions)
        ctx = cell_ctx((rx + x_add - 1, ry + y_add - 1), (nx, ny))
        belief_outline = compose(ctx, rectangle(0.1, 0.87, 0.8, 0.07), stroke("gray31"), fill("gray31"))
        belief_fill = compose(ctx, rectangle(0.1, 0.87, rock_beliefs[i] * 0.8, 0.07), stroke("lawngreen"), fill("lawngreen"))
        push!(belief_outlines, belief_outline)
        push!(belief_fills, belief_fill)
        
        # Text in the middle of the grid square at 2 decimals
        val_str = @sprintf("%0.2f", rock_beliefs[i])
        belief_text = compose(ctx, text(0.5, 0.5, val_str, hcenter, vcenter),
            stroke("white"),
            fill("white"),
            fontsize(20pt),
            font("Computer Modern"),
            linewidth(0.1mm)
            )
        push!(belief_texts, belief_text)
    end
    return compose(context(), belief_fills..., belief_outlines..., belief_texts...)
end

function get_rock_beliefs(momdp::RockSampleMOMDPAT{K}, b) where K
    rock_beliefs = zeros(Float64, K)
    for (sᵢ, bᵢ) in weighted_iterator(b)
        for (rock_idx, rock_state) in enumerate(sᵢ[2].rocks)
            if rock_state
                rock_beliefs[rock_idx] += bᵢ
            end
        end
    end
    return rock_beliefs
end

function get_type_beliefs(momdp::RockSampleMOMDPAT, b)
    # Calculate marginal belief over suggester types
    b_sugg_type = zeros(length(momdp.types))
    for (sᵢ, bᵢ) in weighted_iterator(b)
        type_i = findfirst(isequal(sᵢ[2].sugg_type), momdp.types)
        b_sugg_type[type_i] += bᵢ
    end
    b_sugg_type = b_sugg_type ./ sum(b_sugg_type)
    return b_sugg_type
end

function render_type_info(momdp::RockSampleMOMDPAT, step)
    
    # current_type = step[:s][2].sugg_type
    # num_asks = step[:s][1].num_ask_remain
    
    nx, ny = momdp.map_size
    nx += 3
    ny += 1
    
    ctx = context(0.0, 0.0, 2 / nx, 0.15)
    
    # Current suggester type
    if isnothing(get(step, :sugg_type, nothing))
        type_text = "Sugg λ: $(round(step[:s][2].sugg_type, digits=1))"
    else
        if typeof(step.sugg_type) <: Number
            type_text = "Sugg λ: $(step.sugg_type)"
        else
            type_text = step.sugg_type
        end
    end
    
    type_txt = compose(context(), text(0.05, 0.3, type_text, hleft, vcenter),
        stroke("black"),
        fill("black"),
        fontsize(20pt),
        font("Computer Modern")
        )
    
    # Number of asks remaining
    if isnothing(get(step, :num_asks_remain, nothing))
        asks_text = momdp.num_asks == -1 ? "∞" : step.s[1].num_ask_remain
    else
        asks_text = step.num_asks_remain
    end
    if asks_text == ""
        return compose(ctx, type_txt)
    end
    asks_text = "Asks Rem: $(asks_text)"
    asks_txt = compose(context(), text(0.05, 0.6, asks_text, hleft, vcenter),
        stroke("black"),
        fill("black"),
        fontsize(20pt),
        font("Computer Modern")
        )
        
    return compose(ctx, (type_txt, asks_txt))
end

function render_type_belief(momdp::RockSampleMOMDPAT, step)
    if get(step, :b, nothing) === nothing
        return nothing
    end
    
    b_sugg_type = get_type_beliefs(momdp, step.b)
    
    nx, ny = momdp.map_size
    nx += 3
    ny += 1
    
    b_type_header_txt = compose(context(), text(0.05, 0.07, "b(λ)", hleft, vtop),
        stroke("black"),
        fill("black"),
        fontsize(24pt),
        font("Computer Modern")
        )
        
    x_offset = 0.37
    y_offset = 0.08
    y_start = 0.05
    rec_w = 0.58
    rec_h = 0.045
    belief_outlines = []
    belief_fills = []
    belief_texts = []
    belief_vals_txt = []
    for (ii, type_i) in enumerate(momdp.types)    
        y_pos = y_start + ii * y_offset
        belief_outline = compose(context(), rectangle(x_offset, y_pos, rec_w, rec_h), stroke("gray"), fill("gray"))
        belief_fill = compose(context(), rectangle(x_offset, y_pos, b_sugg_type[ii] * rec_w, rec_h), stroke("darkgreen"), fill("darkgreen"))
        belief_text = compose(context(), text(0.05, y_pos + rec_h / 2, "λ=$(type_i)", hleft, vcenter),
            stroke("black"),
            fill("black"),
            fontsize(18pt),
            font("Computer Modern")
            )
        belief_val_txt = compose(context(), text(x_offset + rec_w / 2, y_pos + rec_h / 2, "$(round(b_sugg_type[ii], digits=2))", hcenter, vcenter),
            stroke("black"),
            fill("black"),
            fontsize(18pt),
            font("Computer Modern")
            )
        push!(belief_outlines, belief_outline)
        push!(belief_fills, belief_fill)
        push!(belief_texts, belief_text)
        push!(belief_vals_txt, belief_val_txt)
    end
    
    # Expected value
    y_pos = y_start + (length(momdp.types) + 1.5) * y_offset
    exp_val = sum(momdp.types .* b_sugg_type)
    ex_val_txt = compose(context(), text(0.05, y_pos, "E[λ] = $(round(exp_val, digits=2))", hleft, vtop),
        stroke("black"),
        fill("black"),
        fontsize(18pt),
        font("Computer Modern")
        )
    
    return compose(context(0.0, 0.15, 2 / nx, 0.85), (b_type_header_txt, belief_vals_txt..., belief_fills..., belief_outlines..., belief_texts..., ex_val_txt))
end

function render_exit(size, y_add)
    nx, ny = size
    rot = Rotation(pi / 2)
    txt = compose(context(), text(0.0, 0.5, "EXIT AREA", hleft, vcenter, rot),
        stroke("black"),
        fill("black"),
        fontsize(30pt),
        font("Palatino")
        )
    y_add_p = y_add == 0 ? 1 : 0
    ctx = context((nx - 1) / nx, 0, 1 / nx, (ny - 1 + y_add_p) / ny)
    return compose(compose(ctx, rectangle(), fill("white"), stroke("black"), linewidth(0.15mm)), txt)
end

function render_agent(ctx)
    center = compose(context(), circle(0.5, 0.5, 0.3), fill("orange"), stroke("black"), linewidth(0.15mm))
    lwheel = compose(context(), ellipse(0.2, 0.5, 0.1, 0.3), fill("orange"), stroke("black"), linewidth(0.15mm))
    rwheel = compose(context(), ellipse(0.8, 0.5, 0.1, 0.3), fill("orange"), stroke("black"), linewidth(0.15mm))
    return compose(ctx, center, lwheel, rwheel)
end

function render_action_text(momdp::RockSampleMOMDPAT, step, pre_act_text, x_add)
    n_basic_actions_mod = momdp.num_asks == 0 ? N_BASIC_ACTIONS - 1 : N_BASIC_ACTIONS
    actions = ["Sample", "North", "East", "South", "West", "Ask"]
    action_text = "Terminal"
    if get(step, :a, nothing) !== nothing
        if step.a <= n_basic_actions_mod
            action_text = actions[step.a]
        else
            action_text = "Sensing Rock $(step.a - n_basic_actions_mod)"
        end
    end
    action_text = pre_act_text * action_text
    
    nx, ny = momdp.map_size
    ny += 1
    nx += x_add
    if x_add == 1
        xs = 0
        w = 1
    else
        w = (nx - 2) / nx
        xs = 2 / nx
    end
    ctx = context(xs, (ny - 1) / ny, w, 1 / ny)
    x_off = 0.01
    y_off = 0.5
    font_size = 22pt
    txt = compose(context(), text(x_off, y_off, action_text, hleft, vcenter),
        stroke("black"),
        fill("black"),
        fontsize(font_size),
        font("Computer Modern")
        )
    return compose(ctx, txt, rectangle(), fill("white"), stroke("white"))
end

function render_action(momdp::RockSampleMOMDPAT, step, x_add, y_add)
    n_basic_actions_mod = momdp.num_asks == 0 ? N_BASIC_ACTIONS - 1 : N_BASIC_ACTIONS
    if step.a == BASIC_ACTIONS_DICT[:sample]
        ctx = cell_ctx(step.s[1].pos .+ (x_add - 1, y_add - 1), momdp.map_size .+ (x_add, y_add))
        if in(step.s[1].pos, momdp.rocks_positions)
            rock_ind = findfirst(isequal(step.s[1].pos), momdp.rocks_positions)
            clr = step.s[2].rocks[rock_ind] ? "green" : "red"
        else
            clr = "black"
        end
        return compose(ctx, ngon(0.5, 0.5, 0.1, 6), stroke("gray"), fill(clr))
    elseif step.a > n_basic_actions_mod
        rock_ind = step.a - n_basic_actions_mod
        rock_pos = momdp.rocks_positions[rock_ind]
        nx, ny = momdp.map_size[1] + x_add, momdp.map_size[2] + y_add
        y_offset = y_add == 0 ? 1 : 0
        rock_pos = ((rock_pos[1] + x_add - 1 - 0.5) / nx, (ny - rock_pos[2] - 0.5 + y_offset) / ny)
        rob_pos = ((step.s[1].pos[1] + x_add - 1 - 0.5) / nx, (ny - step.s[1].pos[2] - 0.5 + y_offset) / ny)
        return compose(context(), line([rob_pos, rock_pos]), stroke("orange"), linewidth(0.01w))
    end
    return nothing
end
