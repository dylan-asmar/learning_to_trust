using Pkg

# Activate the main project environment
Pkg.activate("./")

# Ensure development packages are added
println("Ensuring local packages are available...")
Pkg.develop(path="./RockSampleMOMDPwAT.jl/")
Pkg.develop(path="./TagMOMDPwAT.jl/")

# Instantiate to get all dependencies
println("Installing dependencies...")
Pkg.instantiate()
Pkg.build()

# Load all required packages first
println("Loading packages...")
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
using Revise

println("Loading source files...")
include("src/constants.jl")
include("src/suggesters.jl")
include("src/utils.jl")
include("src/run_sims.jl")
include("src/run_sims_type_eval.jl")
include("src/run_sims_type_eval_dynamic.jl")
include("src/override_tag_tx.jl")
include("src/plot_results.jl")
include("src/pol_generator.jl")

println("Setup complete! You can now run simulations.")
println("Try: _, π_sugg, _ = get_problem_and_policy(:tag)")
