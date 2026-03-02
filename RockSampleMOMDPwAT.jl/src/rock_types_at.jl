const RSPosAT = Tuple{Int, Int}
const RSRocksAT{K} = NTuple{K, Bool} where {K}

struct RockSampleMOMDPXState
    pos::RSPosAT
    num_ask_remain::Int
end

struct RockSampleMOMDPYState{K}
    rocks::RSRocksAT{K}
    sugg_type::Float64
end

const RockSampleMOMDPState{K} = Tuple{RockSampleMOMDPXState, RockSampleMOMDPYState{K}}

const N_BASIC_ACTIONS = 6
const BASIC_ACTIONS_DICT = Dict(
    :sample => 1,
    :north => 2,
    :east => 3,
    :south => 4,
    :west => 5,
    :ask => 6)

const ACTION_DIRS_AT = ((0, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    (0, 0))

struct RockSampleMOMDPAT{K} <: MOMDP{RockSampleMOMDPXState, RockSampleMOMDPYState{K}, Int, Int}
    map_size::Tuple{Int,Int}
    rocks_positions::NTuple{K,RSPosAT}
    init_pos::RSPosAT
    sensor_efficiency::Float64
    ask_cost::Float64
    Q_ask_array::Array{Float64, 3}
    bad_rock_penalty::Float64
    good_rock_reward::Float64
    step_penalty::Float64
    sensor_use_penalty::Float64
    exit_reward::Float64
    discount_factor::Float64
    num_asks::Int
    types::Vector{Float64}
    type_trans::Float64
    init_type_dist::Vector{Float64}
end


function RockSampleMOMDPAT(;
    map_size::Tuple{Int,Int}=(5, 5),
    rocks_positions::Vector{Tuple{Int, Int}}=[(1, 1), (3, 3), (4, 4)],
    init_pos::RSPosAT=(1, 1),
    sensor_efficiency::Float64=20.0,
    ask_cost::Float64=-1.0,
    Q_ask_array::Union{Array{Float64, 3}, Nothing}=nothing,
    bad_rock_penalty::Float64=-10.0,
    good_rock_reward::Float64=10.0,
    step_penalty::Float64=0.,
    sensor_use_penalty::Float64=0.,
    exit_reward::Float64=10.,
    discount_factor::Float64=0.95,
    num_asks::Int=0,
    types::Vector{Float64}=[1.0],
    type_trans::Float64=0.0,
    init_type_dist::Union{Vector{Float64}, Nothing}=nothing
)
    
    @assert num_asks >= -1 "`num_asks` must be -1, or a non-negative integer"
 
    K = length(rocks_positions)
    
    rock_positions_ntuple = Tuple(rocks_positions)
    
    num_q_actions = N_BASIC_ACTIONS + K - 1 # -1 for the ask action
    num_q_x_states = (map_size[1] * map_size[2] + 1)
    num_q_y_states = 2^K
    
    if isnothing(Q_ask_array)
        Q_ask_array = zeros(num_q_x_states, num_q_y_states, num_q_actions)
    end 
    
    assert_msg = "Q_ask_array must have size $(num_q_x_states) x $(num_q_y_states) x $num_q_actions, got $(size(Q_ask_array))"
    
    @assert size(Q_ask_array)[1] == num_q_x_states assert_msg
    @assert size(Q_ask_array)[2] == num_q_y_states assert_msg
    @assert size(Q_ask_array)[3] == num_q_actions assert_msg

    if isnothing(init_type_dist)
        init_type_dist = ones(length(types)) / length(types)
    end
    
    @assert length(init_type_dist) == length(types) "init_type_dist must have the same length as types"
    @assert isapprox(sum(init_type_dist), 1.0, atol=1e-6) "init_type_dist must sum to 1.0"
    
    return RockSampleMOMDPAT{K}(map_size, rock_positions_ntuple, init_pos, sensor_efficiency, ask_cost, Q_ask_array, bad_rock_penalty, good_rock_reward, step_penalty, sensor_use_penalty, exit_reward, discount_factor, num_asks, types, type_trans, init_type_dist)
end


POMDPs.isterminal(::RockSampleMOMDPAT, s::RockSampleMOMDPState) = s[1].pos == (-1, -1)
POMDPs.discount(momdp::RockSampleMOMDPAT) = momdp.discount_factor

MOMDPs.is_y_prime_dependent_on_x_prime(::RockSampleMOMDPAT) = false
MOMDPs.is_x_prime_dependent_on_y(::RockSampleMOMDPAT) = false
MOMDPs.is_initial_distribution_independent(::RockSampleMOMDPAT) = true 
function Base.show(io::IO, momdp::RockSampleMOMDPAT{K}) where {K}
    println(io, "RockSampleMOMDPAT{$K}")
    for name in fieldnames(typeof(momdp))
        if name == :Q_ask_array
            q_array_size = size(getfield(momdp, name))
            print(io, "\t", name, ": ")
            print(io, typeof(getfield(momdp, name)), "$q_array_size\n")
        else
            print(io, "\t", name, ": ", getfield(momdp, name), "\n")
        end
    end
end
