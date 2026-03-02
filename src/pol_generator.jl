using POMDPs
using POMDPTools
using MOMDPs
using SARSOP

using TagMOMDPProblemAT
using RockSampleMOMDPProblemAT

using JLD2
using ProgressMeter
using Printf

include("constants.jl")
include("utils.jl")

"""
    generate_problem_and_policy( problem; timeout=600, kwargs...)

Generates and saves the problem and policy. If more problems are defined, update the
constants in constants.jl. Uses the default solver and passes kwargs to the solver.

# Arguments
- `problem::Symbol`: Which problem to generate. See consstants.jl for options.

# Keyword Arguments
- `timeout=600`: time line for the SARSOP solver
- `trial_improvement_factor=0.01`: trial improvement factor for the SARSOP solver
- Additional kew word arguments are passed to the solver

# Returns
- `nothing`
"""
function generate_problem_and_policy(
    problem::Symbol;
    timeout=600.0,
    trial_improvement_factor=0.01,
    kwargs...
)
    momdp, save_str = get_problem(problem)

    solver = SARSOPSolver(; 
        timeout=timeout, 
        trial_improvement_factor=trial_improvement_factor,
        kwargs...
    )
    pol = solve(solver, momdp)

    save_str = "policies/" * save_str
    @save(save_str, pol)
    println("Complete! Saved as: $save_str")
    return nothing
end

"""
    generate_and_save_Q(problem::Symbol)

Generates and saves the action value function in the form of a matrix Q(s, a). Uses the
`get_problem_and_policy` function defined in utils.jl. This process followed from section
20.3 in Kochenderfer, Mykel J., Tim A. Wheeler, and Kyle H. Wray. Algorithms for decision
making. Mit Press, 2022.

# Returns
- `Q::Matrix{Float64}(length(state_space), length(A))`: action value function as a matrix
"""
function generate_and_save_Q(problem::Symbol)
    momdp, pol, load_str = get_problem_and_policy(problem)
    nx = pol.n_states_x
    ny = pol.n_states_y
    
    states_space_x = ordered_states_x(momdp)
    states_space_y = ordered_states_y(momdp)
    
    A = actions(momdp)

    Q = zeros(nx, ny, length(A))
    
    num_its = nx * ny
    p = Progress(num_its; desc="Calculating action value matrix", barlen=50, showspeed=true)

    for x_idx in 1:nx
        Threads.@threads for y_idx in 1:ny
        # for y_idx in 1:ny
            xi = states_space_x[x_idx]
            yi = states_space_y[y_idx]
            by = SparseCat([yi], [1.0])
            qa = actionvalues(pol, by, xi)
            Q[x_idx, y_idx, :] = qa
            next!(p)
        end
    end
    
    # Count number of non-zero elements in Q
    nnz_Q = count(abs.(Q) .>= 1e-100)
    Q[abs.(Q) .<= 1e-100] .= 0
    @info """Q Info
    Non-zeros: $nnz_Q
    Zeros: $(prod(size(Q)) - nnz_Q)
    Memory usage: $(sizeof(Q) / 1024^2) MB"""    
    
    save_str = load_str * "_Q.jld2"
    @save(save_str, Q)
    
end
