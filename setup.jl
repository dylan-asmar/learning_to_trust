using Pkg

const PROJECT_ROOT = @__DIR__
Pkg.activate(PROJECT_ROOT)

println("Installing dependencies...")
Pkg.instantiate()
Pkg.precompile()

println("Loading LearningToTrust...")
using LearningToTrust

println("Setup complete! You can now run simulations.")
println("Try: _, π_sugg, _ = get_problem_and_policy(:tag)")
