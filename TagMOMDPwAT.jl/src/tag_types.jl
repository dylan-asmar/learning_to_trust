struct TagMOMDPXState
    r_pos::Int
    num_ask_remain::Int
end

struct TagMOMDPYState
    t_pos::Int
    sugg_type::Float64
end

const TagMOMDPState = Tuple{TagMOMDPXState, TagMOMDPYState}

const ACTIONS_DICT = Dict(:north => 1, :east => 2, :south => 3, :west => 4, :tag => 5, :ask => 6)
const ACTION_NAMES = Dict(1 => "North", 2 => "East", 3 => "South", 4 => "West", 5 => "Tag", 6 => "Ask")

"""
    TagMOMDPAT <: MOMDP{TagMOMDPXState, TagMOMDPYState, Int, Int}

MOMDP type for the Tag POMDP.

# Fields
- `mg::MetaDiGraph`: metagraph representing the map
- `dist_matrix::Matrix{Float64}`: distance matrix for the metagraph
- `tag_reward::Float64`: reward for the agent tagging the opponent
- `tag_penalty::Float64`: reward for the agent using the tag action and not being in the same grid cell as the opponent
- `step_penalty::Float64`: reward for each movement action (negative = penalty)
- `ask_penalty::Float64`: reward for the agent asking the opponent for their position
- `Q_ask_array::Array{Float64, 3}`: 
- `discount_factor::Float64`: discount factor
- `move_away_probability::Float64`: probability associated with the opponent srategy. This probability is the chance it moves away versus stays in place.
- `transition_option::Symbol`: option for the transition function
- `num_asks::Int`:
- `types::Vector{Float64}`:
"""
struct TagMOMDPAT <: MOMDP{TagMOMDPXState, TagMOMDPYState, Int, Int}
    mg::MetaDiGraph
    dist_matrix::Matrix{Float64}
    tag_reward::Float64
    tag_penalty::Float64
    step_penalty::Float64
    ask_penalty::Float64
    Q_ask_array::Array{Float64, 3}
    discount_factor::Float64
    move_away_probability::Float64
    transition_option::Symbol
    num_asks::Int
    types::Vector{Float64}
    type_trans::Float64
    init_type_dist::Vector{Float64}
end

"""
    TagMOMDPAT(; kwargs...)

Returns a `TagMOMDPAT <: MOMDP{Int, TagMOMDPYState, Int, Int}`.
Default values are from the original paper:
Pineau, Joelle et al. “Point-based value iteration: An anytime algorithm for POMDPs.” IJCAI (2003).

The main difference in this implementation is the use of only 1 terminal state
and an opponent transition function that aims to keep the probability of moving away to the
specified value if there is a valid action (versus allowing the action and thus increasing
the probability of remaining in place). To use the transition function from the original
implementation, pass `orig_transition_fcn = true`.

# Keywords
- `map_str::String`: String representing the map, 'x' for walls, 'o' for open space.
    Default is the standard map from the original paper.\n
    Default: \"\"\"
    xxxxxoooxx\n
    xxxxxoooxx\n
    xxxxxoooxx\n
    oooooooooo\n
    oooooooooo\"\"\"
- `tag_reward::Float64`: Reward for the agent tagging the opponent, default = +10.0
- `tag_penalty::Float64`: Reward for the agent using the tag action and not being in the same grid cell as the opponent, default = -10.0
- `step_penalty::Float64`: Reward for each movement action, default = -1.0
- `ask_penalty::Float64`: Reward for the agent asking the opponent for their position, default = -1.0
- `discount_factor::Float64`: Discount factor, default = 0.95
- `move_away_probability::Float64`: Probability associated with the opponent srategy. This probability is the chance it moves away, default = 0.8
- `transition_option::Symbol`: Option for the transition function. Options are `:orig` and `:modified`. Default is `:modified`.
- `num_asks::Int`: Number of asks the agent can make. Default is -1, which means the agent can't make any asks. -2 means the agent can make infinite asks.
- `types::Vector{Float64}`: 
"""
function TagMOMDPAT(;
    map_str::String = """xxxxxoooxx
                         xxxxxoooxx
                         xxxxxoooxx
                         oooooooooo
                         oooooooooo
                         """,
    tag_reward::Float64 = 10.0,
    tag_penalty::Float64 = -10.0,
    step_penalty::Float64 = -1.0,
    ask_penalty::Float64 = -1.0,
    Q_ask_array::Union{Array{Float64, 3}, Nothing} = nothing,
    discount_factor::Float64 = 0.95,
    move_away_probability::Float64 = 0.8,
    transition_option::Symbol = :modified,
    num_asks::Int=0,
    types::Vector{Float64} = [1.0],
    type_trans::Float64 = 0.0,
    init_type_dist::Union{Vector{Float64}, Nothing} = nothing
)
    # Remove whitespace from map_str but leaeve line breaks and check for valid charactors
    map_str = replace(map_str, r"[ \t]+" => "")
    if !all(c -> c ∈ ('x', 'o', '\n'), map_str)
        error("Invalid charactor in map_str. Only 'x', 'o', and '\\n' are allowed.")
    end

    # Create metagraph
    mg = create_metagraph_from_map(map_str)

    # Create distance matrix for the metagraph
    dist_matrix = floyd_warshall_shortest_paths(mg).dists

    @assert num_asks >= -1 "`num_asks` must be -1, or a non-negative integer"
    
    num_q_actions = length(ACTIONS_DICT) - 1 # -1 for the ask action
    
    if isnothing(Q_ask_array)
        Q_ask_array = zeros(get_prop(mg, :num_grid_pos) + 1, get_prop(mg, :num_grid_pos), num_q_actions)
    end 
    
    assert_msg = "Q_ask_array must have size $(get_prop(mg, :num_grid_pos) + 1) x $(get_prop(mg, :num_grid_pos)) x $num_q_actions, got $(size(Q_ask_array))"
    
    @assert size(Q_ask_array)[1] == get_prop(mg, :num_grid_pos) + 1 assert_msg
    @assert size(Q_ask_array)[2] == get_prop(mg, :num_grid_pos) assert_msg
    @assert size(Q_ask_array)[3] == num_q_actions assert_msg

    if isnothing(init_type_dist)
        init_type_dist = ones(length(types)) / length(types)
    end
    @assert length(init_type_dist) == length(types) "init_type_dist must have the same length as types"
    @assert isapprox(sum(init_type_dist), 1.0, atol=1e-6) "init_type_dist must sum to 1.0"
    
    return TagMOMDPAT(
        mg, dist_matrix, tag_reward, tag_penalty, step_penalty, ask_penalty, Q_ask_array,
        discount_factor, move_away_probability, transition_option, num_asks, types, type_trans, init_type_dist
    )
end

"""
    create_metagraph_from_map(map_str::String)

Returns a `MetaDiGraph` representing the map. 'x' for walls, 'o' for open space.

Properties of the graph:
- `:nrows`: number of rows in the map
- `:ncols`: number of columns in the map
- `:num_grid_pos`: number of open spaces in the map
- `:node_mapping`: dictionary mapping (i, j) position in the map to node number
- `:node_pos_mapping`: dictionary mapping node number to (i, j) position in the map

Properties of the edges:
- `:action`: action associated with the edge (e.g. :north, :south, :east, :west)

# Example mat_str for the original TagPOMDP (the one in the original paper)
xxxxxoooxx\n
xxxxxoooxx\n
xxxxxoooxx\n
oooooooooo\n
oooooooooo\n
"""
function create_metagraph_from_map(map_str::String)
    lines = split(map_str, '\n')
    if lines[end] == ""
        pop!(lines)
    end

    @assert all(length(line) == length(lines[1]) for line in lines) "Map is not rectangular"

    nrows, ncols = length(lines), length(lines[1])
    num_o = count(c -> c == 'o', map_str)

    g = SimpleDiGraph(num_o)

    node_mapping = Dict{Tuple{Int, Int}, Int}()
    node_pos_mapping = Dict{Int, Tuple{Int, Int}}()
    node_counter = 1

    # Map each open area to a unique node number in the graph
    for (i, line) in enumerate(lines)
        for (j, char) in enumerate(line)
            if char == 'o'
                node_mapping[(i, j)] = node_counter
                node_pos_mapping[node_counter] = (i, j)
                node_counter += 1
            end
        end
    end

    # Create MetaGraph
    mg = MetaDiGraph(g)

    set_prop!(mg, :nrows, nrows)
    set_prop!(mg, :ncols, ncols)
    set_prop!(mg, :num_grid_pos, num_o)
    set_prop!(mg, :node_mapping, node_mapping)
    set_prop!(mg, :node_pos_mapping, node_pos_mapping)

    # Add edges based on possible moves and set action properties
    for (i, line) in enumerate(lines)
        for (j, char) in enumerate(line)
            if char == 'o'
                current_node = node_mapping[(i, j)]
                # North
                if i > 1 && lines[i - 1][j] == 'o'
                    north_node = node_mapping[(i - 1, j)]
                    add_edge!(mg, current_node, north_node)
                    set_prop!(mg, current_node, north_node, :action, :north)
                end
                # South
                if i < nrows && lines[i + 1][j] == 'o'
                    south_node = node_mapping[(i + 1, j)]
                    add_edge!(mg, current_node, south_node)
                    set_prop!(mg, current_node, south_node, :action, :south)
                end
                # East
                if j < ncols && lines[i][j + 1] == 'o'
                    east_node = node_mapping[(i, j + 1)]
                    add_edge!(mg, current_node, east_node)
                    set_prop!(mg, current_node, east_node, :action, :east)
                end
                # West
                if j > 1 && lines[i][j - 1] == 'o'
                    west_node = node_mapping[(i, j - 1)]
                    add_edge!(mg, current_node, west_node)
                    set_prop!(mg, current_node, west_node, :action, :west)
                end
            end
        end
    end
    return mg
end

"""
    map_str_from_metagraph(pomdp::TagMOMDPAT)

Returns a string representing the map. 'x' for walls, 'o' for open space. Uses the
`node_mapping` property of the metagraph to determine which nodes are open spaces.
"""
function map_str_from_metagraph(problem::TagMOMDPAT)
    nrows = get_prop(problem.mg, :nrows)
    ncols = get_prop(problem.mg, :ncols)
    node_mapping = get_prop(problem.mg, :node_mapping)
    lines = Vector{String}(undef, nrows)
    for i in 1:nrows
        line = Vector{Char}(undef, ncols)
        for j in 1:ncols
            if (i, j) in keys(node_mapping)
                line[j] = 'o'
            else
                line[j] = 'x'
            end
        end
        lines[i] = String(line)
    end
    return join(lines, '\n')
end

POMDPs.isterminal(::TagMOMDPAT, s::TagMOMDPState) = false
POMDPs.discount(problem::TagMOMDPAT) = problem.discount_factor

MOMDPs.is_y_prime_dependent_on_x_prime(::TagMOMDPAT) = false
MOMDPs.is_x_prime_dependent_on_y(::TagMOMDPAT) = true
MOMDPs.is_initial_distribution_independent(::TagMOMDPAT) = true


function Base.show(io::IO, problem::TagMOMDPAT)
    println(io, "TagPOMDPProblem")
    for name in fieldnames(typeof(problem))
        if name == :mg
            print(io, "\t", name, ": ")
            print(io, typeof(getfield(problem, name)), ", $(nv(getfield(problem, name))) nodes, $(ne(getfield(problem, name))) edges\n")
        elseif name == :dist_matrix
            d_mat_size = size(getfield(problem, name))
            print(io, "\t", name, ": ")
            print(io, typeof(getfield(problem, name)), "$d_mat_size\n")
        elseif name == :Q_ask_array
            q_array_size = size(getfield(problem, name))
            print(io, "\t", name, ": ")
            print(io, typeof(getfield(problem, name)), "$q_array_size\n")
        else
            print(io, "\t", name, ": ", getfield(problem, name), "\n")
        end
    end

    # Print the map as a string
    map_str = map_str_from_metagraph(problem)
    print(io, "\tmap:\n")
    lines = split(map_str, '\n')
    for line in lines
        print(io, "\t\t", line, "\n")
    end
end
