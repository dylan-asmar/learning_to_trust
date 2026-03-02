# Learning To Trust: Bayesian Adaptation To Varying Suggester Reliability In Sequential Decision Making

This repository contains the code for reproducible experiments supporting our paper *Learning To Trust: Bayesian Adaptation To Varying Suggester Reliability In Sequential Decision Making*.


## Table of Contents
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Problem Domains](#problem-domains)
- [Suggester Types](#suggester-types)
- [Simulation Functions](#simulation-functions)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Generating Policies](#generating-policies)
- [Troubleshooting](#troubleshooting)
- [Code Structure](#code-structure)


## Quick Start

**Prerequisites:** Julia v1.10+ (tested on v1.10 and v1.11)

```bash
git clone <repo_url>
cd learning_to_trust--supp_material
```

```julia
# Activate environment and load dependencies
julia> include("setup.jl")

# Run a basic simulation
julia> _, π_sugg, _ = get_problem_and_policy(:tag)
julia> suggester = PolicySuggester(π_sugg, 3.0)
julia> run_sim(:tag_inf_0_1_2_5_10; no_ask_problem=:tag, suggester=suggester, num_sims=1, visualize=false)
```

**Note:** The setup process will automatically handle local package dependencies and may take a few minutes on first run due to package compilation.

### Verification

To verify the setup worked correctly, you should see:
```
Setup complete! You can now run simulations.
Try: _, π_sugg, _ = get_problem_and_policy(:tag)
```

Test basic functionality:
```julia
# This should load successfully and print policy information
julia> _, π_sugg, _ = get_problem_and_policy(:tag)
# Output: Loading problem and policy...complete!

# This should run a quick simulation
julia> suggester = PolicySuggester(π_sugg, 3.0)
julia> result = run_sim(:tag_inf_0_1_2_5_10; no_ask_problem=:tag, suggester=suggester, num_sims=1)
julia> print_sim_result(result)
```

## Environment Setup

The setup process activates the environment, adds the RockSample and Tag MOMDPs as development dependencies, loads all required packages, and includes necessary source files.

```julia
julia> include("setup.jl")
```

The setup script will:
1. Activate the project environment
2. Install local packages (`RockSampleMOMDPwAT.jl` and `TagMOMDPwAT.jl`) as development dependencies
3. Install all required Julia packages
4. Load all packages and source files
5. Display completion message with usage example

**First-time setup may take 5-10 minutes** due to package installation and compilation.

## Problem Domains

The repository includes two POMDP/MOMDP domains with the ability for the agent to ask for a suggestion and maintain a belief about the suggester type.

### RockSample Standard Problems
- **`:rs84`** - RockSample(8,4,10,-1): 8×8 grid, 4 rocks, sensor efficiency 10, step penalty -1
- **`:rs78`** - RockSample(7,8,20,0): 7×7 grid, 8 rocks, sensor efficiency 20, no step penalty

### Tag Standard Problems  
- **`:tag`** - Tag domain with asking capability disabled

### Problem Variants
Each base problem has variants for different suggester type assumptions:
- `*_inf_1`, `*_inf_2`, `*_inf_5` - Infinite number of asks allowed and a single suggester type (λ=1, 2, 5).
- `*_inf_0_1_2_5_10` - Infinite asks, mixed suggester types(λ ∈ {0,1,2,5,10})
- `*_inf_0_1_2_5_10_w05` - Mixed types with dynamic type transitions (5% chance)
- Additional variants such as `:tag_1_5`, `:tag_1_0_1_2_5_10` etc. are available. See `src/constants.jl` for the full list.

## Suggester Types

The codebase supports multiple types of suggesters that provide action recommendations:

### PolicySuggester
Uses a policy and parameter λ for noisy softmax sampling over action values:
```julia
_, π_sugg, _ = get_problem_and_policy(:tag)
suggester = PolicySuggester(π_sugg, 3.0)  # λ=3.0 
```

### Other Suggester Types
- **`RandomSuggester`** - Returns random actions
- **`NoSuggester`** - Returns -1 (no suggestion)  
- **`RuleSuggester`** - Allows custom rule functions

### Creating Custom Suggesters
Extend `AbstractSuggester` and implement `get_suggestion`:
```julia
struct CustomSuggester <: AbstractSuggester
    λ::Float64
    # your fields
end

function get_suggestion(suggester::CustomSuggester, momdp, s, rng)
    # your logic
    return action
end
```

## Simulation Functions

There are three main simulation functions, each designed for different scenarios:
- `run_sim` (`src/run_sims.jl`) — Standard simulation with a fixed suggester.
- `run_sim_type_eval` (`src/run_sims_type_eval.jl`) — Static type evaluation, where the agent maintains beliefs over suggester types. (Reproduces Table 1).
- `run_sim_dynamic_suggester` (`src/run_sims_type_eval_dynamic.jl`) — Dynamic type scenarios, with agent beliefs adapting to type transitions. (Reproduces Figures 1 & 2).

### `run_sim()` - Basic Simulations
The foundational simulation function that runs standard MOMDP experiments. It handles:
- Single or multiple simulation runs
- Visualization and verbose output for debugging
- Basic statistics collection (rewards, steps, suggestions)
- Integration with different suggester types

### `run_sim_type_eval()` - Static Type Evaluation
Designed for experiments where the agent maintains a fixed belief about suggester types throughout the episode. This function is essential for reproducing **Table 1** results from the paper.

### `run_sim_dynamic_suggester()` - Dynamic Type Scenarios  
Handles time-varying suggester scenarios where the suggester type can change during episodes. Critical for reproducing **Figures 1 & 2** from the paper.

## Reproducing Paper Results

### Table 1: Static Suggester Type Evaluation

Use `run_sim_type_eval()` for experiments with static suggester types:

```julia
# Agent assumes λ=5.0 suggester, but actual suggester is λ=3.0
prob = :tag
prob_type = :tag_inf_5
_, π_sugg, _ = get_problem_and_policy(prob)
suggester = PolicySuggester(π_sugg, 3.0)

sim_result = run_sim_type_eval(
    prob, 
    prob_type; 
    num_trials=50, 
    max_steps=500, 
    num_sims=200, 
    agent=:normal, 
    seed=45, 
    suggester=suggester, 
    ν=1.0  # not used for normal agent
)

# Print results
print_sim_result(sim_result)
```

#### Key Arguments for `run_sim_type_eval()`:
- **`problem`**: Standard problem (`:tag`, `:rs84`, `:rs78`) - the base POMDP without type considerations
- **`type_problem`**: Problem with integrated type state space (`:tag_inf_5`, `:rs84_inf_0_1_2_5_10`, etc.) - defines what types the agent can reason about
- **`agent`**: Agent behavior type:
  - `:normal` - Optimal agent that maintains beliefs about suggester types
  - `:perfect` - Agent with perfect knowledge of suggester type
  - `:random` - Random action selection
  - `:naive` - Simple agent that follows suggestions with probability `ν`
- **`suggester`**: The actual suggester providing recommendations (can differ from agent's belief)
- **`ν`**: Used only with `:naive` agent - probability of following suggestions

#### Example Agent Configurations:
```julia
# Get policy first
_, π, _ = get_problem_and_policy(:tag)

# Noisy agent assuming single type λ=5.0, actual suggester λ=3.0
run_sim_type_eval(:tag, :tag_inf_5; agent=:normal, suggester=PolicySuggester(π, 3.0))

# Naive agent with 80% suggestion following rate
run_sim_type_eval(:tag, :tag_inf_5; agent=:naive, ν=0.8, suggester=PolicySuggester(π, 3.0))

# Agent with multiple types and the actual suggester is λ=1.0
run_sim_type_eval(:tag, :tag_inf_0_1_2_5_10; agent=:normal, suggester=PolicySuggester(π, 1.0))
```

**Performance Note:** For paper reproduction, use the full parameter values shown above. For testing, consider smaller values (e.g., `num_sims=10, num_trials=10`) to reduce runtime.

### Figures 1 & 2: Dynamic Suggester Types

For dynamic suggester experiments (changing types over time):

```julia
_, π_sugg, _ = get_problem_and_policy(:tag)

# Define time-varying suggesters
suggesters = Dict(
    0 => PolicySuggester(π_sugg, 3.0),
    15 => PolicySuggester(π_sugg, 5.0),
    25 => PolicySuggester(π_sugg, 1.0),
    40 => PolicySuggester(π_sugg, 15.0),
    50 => PolicySuggester(π_sugg, 2.0),
    60 => PolicySuggester(π_sugg, 0.0),
    75 => PolicySuggester(π_sugg, 10.0)
)

sr1 = run_sim_dynamic_suggester(
    :tag,
    :tag_inf_0_1_2_5_10;
    num_trials=100,
    max_steps=20000,
    num_sims=100,
    agent=:normal,
    seed=45,
    suggesters=suggesters
)

sr2 = run_sim_dynamic_suggester(
    :tag,
    :tag_inf_0_1_2_5_10_w05;
    num_trials=100,
    max_steps=20000,
    num_sims=100,
    agent=:normal,
    seed=45,
    suggesters=suggesters
)
```

#### Key Arguments for `run_sim_dynamic_suggester()`:
- **`problem`**: Standard problem (`:tag`, `:rs84`, `:rs78`) - base POMDP
- **`type_problem`**: Problem with type state space - defines the agent's type reasoning capability:
  - `:tag_inf_0_1_2_5_10` - Agent reasons about types {0,1,2,5,10} with static transitions
  - `:tag_inf_0_1_2_5_10_w05` - Same types but with 5% transition probability between types
- **`suggesters`**: Dictionary mapping timesteps to suggester instances
  - Keys: timestep when suggester changes
  - Values: `AbstractSuggester` instances (typically `PolicySuggester` with different λ values)
- **`agent`**: Agent type (same as `run_sim_type_eval`)

#### Plotting Dynamic Suggester Results

The repository includes plotting utilities for plotting the results of the simulations:

```julia
using Colors
# Plot combined results
colors = Dict(
    :t012510_00 => RGB(0.8889, 0.2, 0.2),    # Red
    :t012510_05 => RGB(0.78, 0.15, 0.8243)   # Magenta
)

plt = plot_sim_result(sr1, [0.0, 1.0, 2.0, 5.0, 10.0]; 
                     label="T = {0, 1, 2, 5, 10}, t_p = 0.00", 
                     color=colors[:t012510_00])
plt = plot_sim_result!(plt, sr2, [0.0, 1.0, 2.0, 5.0, 10.0]; 
                      label="T = {0, 1, 2, 5, 10}, t_p = 0.05", 
                      color=colors[:t012510_05])
```

# Generating Policies

The function to generate and save policies and a function to gnereat the action value function as a matrix is in `pol_generator.jl`. To generate and save a policy, call `generate_problem_and_policy` with the problem of interest. Parameters can be passed to the SARSOP solver by keywords. Due to space limitations of the supplementary material, The RockSample(7,8) policies and action value fucntions are not provided.

```julia
# Generate and save a policy
julia> generate_problem_and_policy(:tag; timeout=600)

# Generate and save the Q-matrix (required for ask-enabled domains)
julia> generate_and_save_Q(:tag)
```

- Policies are saved under `policies/` and loaded automatically by simulation functions.
- Q-matrices are required whenever the problem includes an ask action.

## Troubleshooting

### Common Issues

**Package Loading Errors:**
If you encounter errors about packages not being found (e.g., `RockSampleMOMDPProblemAT` or `TagMOMDPProblemAT`), try:
```julia
# Restart Julia and run setup again
julia> include("setup.jl")
```

**Dependency Conflicts:**
If you encounter package version conflicts:
```julia
# Clear the environment and reinstall
julia> using Pkg; Pkg.activate("."); Pkg.resolve()
```

**Missing Policies:**
The repository includes pre-computed policies in the `policies/` directory. If policies are missing, you can generate them:
```julia
julia> generate_problem_and_policy(:tag; timeout=600)
julia> generate_and_save_Q(:tag)
```

## Code Structure

### Core Files
- **`setup.jl`** - Environment setup and package loading
- **`src/constants.jl`** - Problem and agent type definitions
- **`src/utils.jl`** - Core utilities including `get_problem()` and `get_problem_and_policy()`
- **`src/suggesters.jl`** - Suggester type definitions and implementations

### Simulation Functions
- **`src/run_sims.jl`** - Main simulation function `run_sim()`
- **`src/run_sims_type_eval.jl`** - Static type evaluation `run_sim_type_eval()`
- **`src/run_sims_type_eval_dynamic.jl`** - Dynamic type experiments `run_sim_dynamic_suggester()`

### Policy Management
- **`src/pol_generator.jl`** - Policy generation `generate_problem_and_policy()` and action value function generation `generate_and_save_Q()`
- **`policies/`** - Pre-computed policies and Q-matrices

### Domain Implementations
- **`RockSampleMOMDPwAT.jl/`** - RockSample MOMDP implementation
- **`TagMOMDPwAT.jl/`** - Tag MOMDP implementation

### Visualization and Analysis
- **`src/plot_results.jl`** - Plotting functions for result analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{asmar2025learning,
  title={Learning to Trust: Bayesian Adaptation to Varying Suggester Reliability in Sequential Decision Making}, 
  author={Dylan M. Asmar and Mykel J. Kochenderfer},
  year={2025},
  eprint={2511.12378},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2511.12378}
}
```