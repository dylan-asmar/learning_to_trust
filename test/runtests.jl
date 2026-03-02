using Pkg

# Change working directory to the project root so relative paths (e.g. "policies/") work.
const PROJECT_ROOT = dirname(@__DIR__)
cd(PROJECT_ROOT)
Pkg.activate(PROJECT_ROOT; io=devnull)

# ── Package loading 
using Test
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
using StaticArrays
using Measures

# ── Source files ─────────────────────────────────────────────────────────────
# Include the three top-level simulation files; each one internally includes
# its own dependencies (constants, utils, suggesters, …).  The type-eval file
# also includes override_tag_tx.jl, which patches transition_y for TagMOMDPAT
# and must be in effect before any Tag simulations are executed.
#
# NOTE: Julia's include() resolves paths relative to the *file being parsed*,
# not the working directory.  We therefore build absolute paths via PROJECT_ROOT.
const SRC = joinpath(PROJECT_ROOT, "src")
include(joinpath(SRC, "run_sims.jl"))
include(joinpath(SRC, "run_sims_type_eval.jl"))          # also includes override_tag_tx
include(joinpath(SRC, "run_sims_type_eval_dynamic.jl"))

# ═══════════════════════════════════════════════════════════════════════════════
@testset verbose=true "Learning to Trust - Supplementary Material" begin

# ─────────────────────────────────────────────────────────────────────────────
@testset "1  Constants" begin
    # Agent symbols
    @test :normal  in AGENTS
    @test :perfect in AGENTS
    @test :random  in AGENTS
    @test :naive   in AGENTS
    @test :scaled  in AGENTS
    @test :noisy   in AGENTS
    @test length(AGENTS) == 6

    # Tag problem list
    @test :tag                    in TG_PROBS
    @test :tag_inf_0_1_2_5_10     in TG_PROBS
    @test :tag_inf_0_1_2_5_10_w05 in TG_PROBS
    @test :tag_inf_1              in TG_PROBS
    @test :tag_inf_2              in TG_PROBS
    @test :tag_inf_5              in TG_PROBS
    @test :tag_1_5                in TG_PROBS

    # Base :tag has no ask action, so it should NOT be in TG_ASK_PROBS
    @test !(:tag in TG_ASK_PROBS)
    @test :tag_inf_1 in TG_ASK_PROBS
    @test :tag_inf_5 in TG_ASK_PROBS
    @test :tag_inf_0_1_2_5_10 in TG_ASK_PROBS

    # RS problem list
    @test :rs84                    in RS_PROBS
    @test :rs84_inf_1              in RS_PROBS
    @test :rs84_inf_2              in RS_PROBS
    @test :rs84_inf_5              in RS_PROBS
    @test :rs84_inf_0_1_2_5_10    in RS_PROBS
    @test :rs84_inf_0_1_2_5_10_w05 in RS_PROBS
    @test :rs78                    in RS_PROBS

    # Verify the known members and the known non-member.
    @test :tag_inf_1  in TG_ASK_PROBS
    @test :tag_inf_5  in TG_ASK_PROBS
    @test :tag_inf_10 in TG_ASK_PROBS
    @test !(:tag in TG_ASK_PROBS)        # base problem has no ask

    # All RS ask-problems are in RS_PROBS
    @test all(p in RS_PROBS for p in RS_ASK_PROBS)
    @test :rs84_inf_1 in RS_ASK_PROBS
    @test :rs84_inf_5 in RS_ASK_PROBS
    @test !(:rs84 in RS_ASK_PROBS)       # base RS problem has no ask
end  # testset 1

# ─────────────────────────────────────────────────────────────────────────────
@testset "2  Suggester Types" begin

    @testset "NoSuggester" begin
        s = NoSuggester(0.0)
        @test s isa AbstractSuggester
        @test s isa NoSuggester
        @test s.λ == 0.0

        rng = MersenneTwister(1)
        tag_momdp = TagMOMDPAT(; num_asks=0, types=[1.0])
        st = rand(rng, initialstate(tag_momdp))
        @test get_suggestion(s, tag_momdp, st, rng) == -1
    end

    @testset "RandomSuggester" begin
        s = RandomSuggester(1.0)
        @test s isa AbstractSuggester
        @test s isa RandomSuggester
        @test s.λ == 1.0

        rng = MersenneTwister(2)
        tag_momdp = TagMOMDPAT(; num_asks=0, types=[1.0])
        st = rand(rng, initialstate(tag_momdp))
        na = length(actions(tag_momdp))
        for _ in 1:20
            sug = get_suggestion(s, tag_momdp, st, rng)
            @test 1 <= sug <= na
        end
    end

    @testset "RuleSuggester" begin
        rule_fn = (momdp, s, λ, rng) -> 1   # always suggest action 1
        s = RuleSuggester(rule_fn, 2.0)
        @test s isa AbstractSuggester
        @test s isa RuleSuggester
        @test s.λ == 2.0

        rng = MersenneTwister(3)
        tag_momdp = TagMOMDPAT(; num_asks=0, types=[1.0])
        st = rand(rng, initialstate(tag_momdp))
        @test get_suggestion(s, tag_momdp, st, rng) == 1
    end

    @testset "PolicySuggester (loaded policy)" begin
        momdp_tag, π_tag, _ = get_problem_and_policy(:tag)
        s = PolicySuggester(π_tag, 3.0)
        @test s isa AbstractSuggester
        @test s isa PolicySuggester
        @test s.λ == 3.0
        @test s.policy === π_tag

        # Softmax suggestion should be a valid action index
        rng = MersenneTwister(4)
        st = rand(rng, initialstate(momdp_tag))
        sug = get_suggestion(s, momdp_tag, st, rng)
        @test 1 <= sug <= length(actions(momdp_tag))
    end

    @testset "PolicySuggester λ=0 (uniform random)" begin
        momdp_tag, π_tag, _ = get_problem_and_policy(:tag)
        s = PolicySuggester(π_tag, 0.0)   # all softmax weights equal → uniform
        rng = MersenneTwister(5)
        na = length(actions(momdp_tag))
        for _ in 1:10
            st = rand(rng, initialstate(momdp_tag))
            sug = get_suggestion(s, momdp_tag, st, rng)
            @test 1 <= sug <= na
        end
    end
end  # testset 2

# ─────────────────────────────────────────────────────────────────────────────
@testset "3  Policy Loading" begin

    @testset "3.1  Tag domain" begin

        @testset ":tag (no ask)" begin
            momdp, pol, load_str = get_problem_and_policy(:tag)
            @test momdp isa TagMOMDPAT
            @test momdp.num_asks == 0
            @test pol isa MOMDPAlphaVectorPolicy
            @test load_str == "policies/tag_m"
            @test length(momdp.types) == 1
        end

        @testset ":tag_inf_1 (λ=1)" begin
            momdp, pol, _ = get_problem_and_policy(:tag_inf_1)
            @test momdp isa TagMOMDPAT
            @test momdp.num_asks == -1          # infinite asks
            @test momdp.types == [1.0]
        end

        @testset ":tag_inf_2 (λ=2)" begin
            momdp, pol, _ = get_problem_and_policy(:tag_inf_2)
            @test momdp isa TagMOMDPAT
            @test momdp.types == [2.0]
        end

        @testset ":tag_inf_5 (λ=5)" begin
            momdp, pol, _ = get_problem_and_policy(:tag_inf_5)
            @test momdp isa TagMOMDPAT
            @test momdp.types == [5.0]
        end

        @testset ":tag_1_5 (1 ask, λ=5)" begin
            momdp, pol, _ = get_problem_and_policy(:tag_1_5)
            @test momdp isa TagMOMDPAT
            @test momdp.num_asks == 1
        end

        @testset ":tag_inf_0_1_2_5_10 (mixed types)" begin
            momdp, pol, _ = get_problem_and_policy(:tag_inf_0_1_2_5_10)
            @test momdp isa TagMOMDPAT
            @test momdp.num_asks == -1
            @test momdp.types == [0.0, 1.0, 2.0, 5.0, 10.0]
            @test momdp.type_trans == 0.0
        end

        @testset ":tag_inf_0_1_2_5_10_w05 (mixed + transitions)" begin
            momdp, pol, _ = get_problem_and_policy(:tag_inf_0_1_2_5_10_w05)
            @test momdp isa TagMOMDPAT
            @test momdp.types == [0.0, 1.0, 2.0, 5.0, 10.0]
            @test momdp.type_trans ≈ 0.05
        end

        @testset ":tag_1_0_1_2_5_10 (1 ask, mixed)" begin
            momdp, pol, _ = get_problem_and_policy(:tag_1_0_1_2_5_10)
            @test momdp isa TagMOMDPAT
            @test momdp.num_asks == 1
            @test length(momdp.types) == 5
        end

        @testset ":tag_1_0_1_2_5_10_w05 (1 ask, mixed + transitions)" begin
            momdp, pol, _ = get_problem_and_policy(:tag_1_0_1_2_5_10_w05)
            @test momdp isa TagMOMDPAT
            @test momdp.num_asks == 1
            @test momdp.type_trans ≈ 0.05
        end
    end  # Tag domain

    @testset "3.2  RockSample(8,4) domain" begin
        # The README notes that RS(7,8) policies are NOT provided due to space limits.
        # All RS(8,4) policies ARE provided.

        @testset ":rs84 (no ask)" begin
            momdp, pol, load_str = get_problem_and_policy(:rs84)
            @test momdp isa RockSampleMOMDPAT
            @test momdp.num_asks == 0
            @test pol isa MOMDPAlphaVectorPolicy
            @test load_str == "policies/rs_8-4-10-1"
            @test momdp.map_size == (8, 8)
            @test length(momdp.rocks_positions) == 4
        end

        @testset ":rs84_inf_1 (λ=1)" begin
            momdp, pol, _ = get_problem_and_policy(:rs84_inf_1)
            @test momdp isa RockSampleMOMDPAT
            @test momdp.num_asks == -1
            @test momdp.types == [1.0]
        end

        @testset ":rs84_inf_2 (λ=2)" begin
            momdp, pol, _ = get_problem_and_policy(:rs84_inf_2)
            @test momdp isa RockSampleMOMDPAT
            @test momdp.types == [2.0]
        end

        @testset ":rs84_inf_5 (λ=5)" begin
            momdp, pol, _ = get_problem_and_policy(:rs84_inf_5)
            @test momdp isa RockSampleMOMDPAT
            @test momdp.types == [5.0]
        end

        @testset ":rs84_inf_0_1_2_5_10 (mixed types)" begin
            momdp, pol, _ = get_problem_and_policy(:rs84_inf_0_1_2_5_10)
            @test momdp isa RockSampleMOMDPAT
            @test momdp.num_asks == -1
            @test momdp.types == [0.0, 1.0, 2.0, 5.0, 10.0]
        end

        @testset ":rs84_inf_0_1_2_5_10_w05 (mixed + transitions)" begin
            momdp, pol, _ = get_problem_and_policy(:rs84_inf_0_1_2_5_10_w05)
            @test momdp isa RockSampleMOMDPAT
            @test momdp.types == [0.0, 1.0, 2.0, 5.0, 10.0]
            @test momdp.type_trans ≈ 0.05
        end
    end  # RS domain
end  # testset 3

# ─────────────────────────────────────────────────────────────────────────────
@testset "4  Utility Functions" begin

    @testset "get_stats" begin
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ave, sd, se, ci = get_stats(data)
        all_data = vcat(data...)
        @test ave ≈ mean(all_data)
        @test sd  ≈ std(all_data)
        @test se  ≈ sd / sqrt(length(all_data))
        @test ci  ≈ 1.96 * se
    end

    @testset "suggestion_to_observation" begin
        rs_momdp, _, _ = get_problem_and_policy(:rs84_inf_1)
        # RS: suggestion index → observation index = suggestion + 3
        @test suggestion_to_observation(rs_momdp, 1) == 4
        @test suggestion_to_observation(rs_momdp, 0) == 3

        tag_momdp, _, _ = get_problem_and_policy(:tag_inf_1)
        # Tag: suggestion index → observation index = suggestion + 2
        @test suggestion_to_observation(tag_momdp, 1) == 3
        @test suggestion_to_observation(tag_momdp, 0) == 2
    end
end  # testset 4

# ─────────────────────────────────────────────────────────────────────────────
#  README QUICK START
#  ─────────────────
#  _, π_sugg, _ = get_problem_and_policy(:tag)
#  suggester = PolicySuggester(π_sugg, 3.0)
#  run_sim(:tag_inf_0_1_2_5_10; no_ask_problem=:tag, suggester=suggester,
#           num_sims=1, visualize=false)
@testset "5  README Quick Start" begin
    _, π_sugg, _ = get_problem_and_policy(:tag)
    suggester = PolicySuggester(π_sugg, 3.0)

    @test suggester isa PolicySuggester
    @test suggester.λ == 3.0

    result = run_sim(
        :tag_inf_0_1_2_5_10;
        no_ask_problem = :tag,
        suggester      = suggester,
        num_sims       = 1,
        num_trials     = 1,
        max_steps      = 50,
        visualize      = false,
        seed           = 42,
    )

    @test result isa SimResult
    @test result.problem       == :tag_inf_0_1_2_5_10
    @test result.no_ask_problem == :tag
    @test result.num_sims      == 1
    @test result.agent         == :normal

    # print_sim_result must not throw
    @test_nowarn print_sim_result(result)
end  # testset 5

# ─────────────────────────────────────────────────────────────────────────────
@testset "6  run_sim – Tag domain" begin

    # Shared setup
    _, π_sugg_tag, _ = get_problem_and_policy(:tag)

    @testset "NoSuggester" begin
        result = run_sim(
            :tag_inf_5;
            no_ask_problem = :tag,
            suggester      = NoSuggester(0.0),
            num_sims       = 2,
            num_trials     = 2,
            max_steps      = 50,
            visualize      = false,
            seed           = 10,
        )
        @test result isa SimResult
        @test result.problem   == :tag_inf_5
        @test result.num_sims  == 2
        @test result.suggester_type == NoSuggester

        # Belief vectors must be valid probability distributions
        for sim_bvec in result.b_sugg_type_per_trial_per_sim
            for b in sim_bvec
                @test all(b .>= 0)
                @test sum(b) ≈ 1.0  atol=1e-6
            end
        end
    end

    @testset "PolicySuggester (λ=3)" begin
        suggester = PolicySuggester(π_sugg_tag, 3.0)
        result = run_sim(
            :tag_inf_5;
            no_ask_problem = :tag,
            suggester      = suggester,
            num_sims       = 2,
            num_trials     = 2,
            max_steps      = 50,
            visualize      = false,
            seed           = 20,
        )
        @test result isa SimResult
        @test result.λ_sugg == 3.0
    end

    @testset "RandomSuggester" begin
        result = run_sim(
            :tag_inf_1;
            no_ask_problem = :tag,
            suggester      = RandomSuggester(0.0),
            num_sims       = 1,
            num_trials     = 1,
            max_steps      = 30,
            visualize      = false,
            seed           = 30,
        )
        @test result isa SimResult
        @test result.suggester_type == RandomSuggester
    end

    @testset "mixed types (:tag_inf_0_1_2_5_10)" begin
        suggester = PolicySuggester(π_sugg_tag, 5.0)
        result = run_sim(
            :tag_inf_0_1_2_5_10;
            no_ask_problem = :tag,
            suggester      = suggester,
            num_sims       = 2,
            num_trials     = 2,
            max_steps      = 50,
            visualize      = false,
            seed           = 40,
        )
        @test result isa SimResult
        @test length(result.b_sugg_type_per_trial_per_sim[1][1]) == 5  # 5 types
    end

    @testset "perfect agent" begin
        suggester = PolicySuggester(π_sugg_tag, 5.0)
        result = run_sim(
            :tag_inf_5;
            no_ask_problem = :tag,
            suggester      = suggester,
            num_sims       = 1,
            num_trials     = 1,
            max_steps      = 30,
            agent          = :perfect,
            visualize      = false,
            seed           = 50,
        )
        @test result isa SimResult
        @test result.agent == :perfect
    end
end  # testset 6

# ─────────────────────────────────────────────────────────────────────────────
@testset "7  run_sim – RockSample(8,4) domain" begin
    _, π_sugg_rs, _ = get_problem_and_policy(:rs84)

    @testset "PolicySuggester (:rs84_inf_5)" begin
        result = run_sim(
            :rs84_inf_5;
            no_ask_problem = :rs84,
            suggester      = PolicySuggester(π_sugg_rs, 5.0),
            num_sims       = 2,
            num_trials     = 1,
            max_steps      = 50,
            visualize      = false,
            seed           = 60,
        )
        @test result isa SimResult
        @test result.problem       == :rs84_inf_5
        @test result.no_ask_problem == :rs84
    end

    @testset "mixed types (:rs84_inf_0_1_2_5_10)" begin
        result = run_sim(
            :rs84_inf_0_1_2_5_10;
            no_ask_problem = :rs84,
            suggester      = PolicySuggester(π_sugg_rs, 2.0),
            num_sims       = 2,
            num_trials     = 1,
            max_steps      = 50,
            visualize      = false,
            seed           = 70,
        )
        @test result isa SimResult
        @test length(result.b_sugg_type_per_trial_per_sim[1][1]) == 5
    end
end  # testset 7

# ─────────────────────────────────────────────────────────────────────────────
#  TABLE 1 REPRODUCTION
#  ─────────────────────
#  run_sim_type_eval is the function used to generate Table 1 in the paper.
#  The paper uses num_trials=50, max_steps=500, num_sims=200; here we use
#  minimal parameters to keep the test suite fast.
@testset "8  run_sim_type_eval – Table 1" begin
    _, π_sugg, _ = get_problem_and_policy(:tag)
    base_suggester = PolicySuggester(π_sugg, 3.0)

    sim_kwargs = (
        num_trials = 2,
        max_steps  = 50,
        num_sims   = 2,
        seed       = 45,
    )

    @testset "normal agent – single type (:tag_inf_5)" begin
        # README example: agent assumes λ=5, actual λ=3
        result = run_sim_type_eval(:tag, :tag_inf_5;
            agent     = :normal,
            suggester = base_suggester,
            ν         = 1.0,
            sim_kwargs...
        )
        @test result isa SimResultTypeEval
        @test result.agent        == :normal
        @test result.problem      == :tag
        @test result.type_problem == :tag_inf_5
        @test result.num_sims     == 2
        @test result.num_trials_per_sim == 2
        @test result.ν            == 1.0
        @test length(result.total_reward_per_sim)           == 2
        @test length(result.reward_per_trial_per_sim)       == 2
        @test length(result.reward_per_trial_per_sim[1])    == 2
        @test length(result.b_sugg_type_per_trial_per_sim)  == 2
        @test_nowarn print_sim_result(result)
    end

    @testset "naive agent – 80% follow rate" begin
        result = run_sim_type_eval(:tag, :tag_inf_5;
            agent     = :naive,
            suggester = base_suggester,
            ν         = 0.8,
            sim_kwargs...
        )
        @test result isa SimResultTypeEval
        @test result.agent == :naive
        @test result.ν    == 0.8
    end

    @testset "perfect agent" begin
        result = run_sim_type_eval(:tag, :tag_inf_5;
            agent     = :perfect,
            suggester = base_suggester,
            ν         = 1.0,
            sim_kwargs...
        )
        @test result isa SimResultTypeEval
        @test result.agent == :perfect
    end

    @testset "random agent" begin
        result = run_sim_type_eval(:tag, :tag_inf_5;
            agent     = :random,
            suggester = base_suggester,
            ν         = 1.0,
            sim_kwargs...
        )
        @test result isa SimResultTypeEval
        @test result.agent == :random
    end

    @testset "normal agent – mixed types (:tag_inf_0_1_2_5_10)" begin
        # Actual λ=1, agent reasons over {0,1,2,5,10}
        result = run_sim_type_eval(:tag, :tag_inf_0_1_2_5_10;
            agent     = :normal,
            suggester = PolicySuggester(π_sugg, 1.0),
            ν         = 1.0,
            sim_kwargs...
        )
        @test result isa SimResultTypeEval
        @test result.type_problem == :tag_inf_0_1_2_5_10
        # 5 types → belief vector has length 5
        @test length(result.b_sugg_type_per_trial_per_sim[1][1]) == 5
        # All beliefs are valid probability distributions
        for sim_bvec in result.b_sugg_type_per_trial_per_sim
            for b in sim_bvec
                @test all(b .>= 0)
                @test sum(b) ≈ 1.0  atol=1e-6
            end
        end
    end

    @testset "RS84 – normal agent (:rs84_inf_5)" begin
        _, π_sugg_rs, _ = get_problem_and_policy(:rs84)
        result = run_sim_type_eval(:rs84, :rs84_inf_5;
            agent     = :normal,
            suggester = PolicySuggester(π_sugg_rs, 3.0),
            ν         = 1.0,
            num_trials = 1,
            max_steps  = 50,
            num_sims   = 2,
            seed       = 55,
        )
        @test result isa SimResultTypeEval
        @test result.problem      == :rs84
        @test result.type_problem == :rs84_inf_5
    end
end  # testset 8

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURES 1 & 2 REPRODUCTION
#  ──────────────────────────
#  run_sim_dynamic_suggester handles time-varying suggester scenarios.
#  The paper uses num_trials=100, max_steps=20000, num_sims=100; we use
#  minimal parameters for the test suite.
@testset "9  run_sim_dynamic_suggester – Figs 1 & 2" begin
    _, π_sugg, _ = get_problem_and_policy(:tag)

    # Time-varying suggesters (simplified from the README example)
    suggesters = Dict{Int, AbstractSuggester}(
        0  => PolicySuggester(π_sugg, 3.0),
        5  => PolicySuggester(π_sugg, 5.0),
        10 => PolicySuggester(π_sugg, 1.0),
    )

    dyn_kwargs = (
        num_trials = 2,
        max_steps  = 200,
        num_sims   = 2,
        agent      = :normal,
        seed       = 45,
        suggesters = suggesters,
    )

    @testset "static type transitions (:tag_inf_0_1_2_5_10)" begin
        sr = run_sim_dynamic_suggester(:tag, :tag_inf_0_1_2_5_10; dyn_kwargs...)
        @test sr isa SimResultTypeEval
        @test sr.problem      == :tag
        @test sr.type_problem == :tag_inf_0_1_2_5_10
        @test sr.num_sims     == 2
        @test length(sr.total_reward_per_sim) == 2
        @test_nowarn print_sim_result(sr)
    end

    @testset "dynamic type transitions (:tag_inf_0_1_2_5_10_w05)" begin
        sr = run_sim_dynamic_suggester(:tag, :tag_inf_0_1_2_5_10_w05; dyn_kwargs...)
        @test sr isa SimResultTypeEval
        @test sr.problem      == :tag
        @test sr.type_problem == :tag_inf_0_1_2_5_10_w05
    end
end  # testset 9

# ─────────────────────────────────────────────────────────────────────────────
@testset "10  SimResult / SimResultTypeEval structure" begin

    _, π_sugg, _ = get_problem_and_policy(:tag)
    suggester = PolicySuggester(π_sugg, 3.0)

    result = run_sim(
        :tag_inf_5;
        no_ask_problem = :tag,
        suggester      = suggester,
        num_sims       = 3,
        num_trials     = 2,
        max_steps      = 30,
        visualize      = false,
        seed           = 1,
    )

    @test result.num_sims           == 3
    @test result.num_trials_per_sim == 2
    @test result.seed               == 1
    @test result.max_steps          == 30

    @test length(result.reward_per_trial_per_sim)       == 3
    @test length(result.reward_per_trial_per_sim[1])    == 2
    @test length(result.steps_per_trial_per_sim)        == 3
    @test length(result.asks_per_trial_per_sim)         == 3
    @test length(result.total_reward_per_sim)           == 3
    @test length(result.total_steps_per_sim)            == 3
    @test length(result.total_asks_per_sim)             == 3
    @test length(result.b_sugg_type_per_trial_per_sim)  == 3
    @test all(length(bv) == 2 for bv in result.b_sugg_type_per_trial_per_sim)

    # Steps-per-trial should be positive
    for s in vcat(result.steps_per_trial_per_sim...)
        @test s > 0
    end

    # Asks per trial should be non-negative
    for a in vcat(result.asks_per_trial_per_sim...)
        @test a >= 0
    end

    # show / print should not error
    @test_nowarn show(devnull, result)

    # SimResultTypeEval
    te_result = run_sim_type_eval(:tag, :tag_inf_5;
        agent          = :normal,
        suggester      = suggester,
        ν              = 1.0,
        num_sims       = 2,
        num_trials     = 2,
        max_steps      = 30,
        seed           = 2,
    )
    @test te_result isa SimResultTypeEval
    @test te_result.num_sims           == 2
    @test te_result.num_trials_per_sim == 2
    @test te_result.ν                  == 1.0
    @test_nowarn show(devnull, te_result)
end  # testset 10

# ─────────────────────────────────────────────────────────────────────────────
@testset "11  Error Handling" begin
    # Unknown problem symbol
    @test_throws ErrorException get_problem_and_policy(:totally_unknown_problem)

    # run_sim with an invalid agent symbol should error
    _, π_sugg, _ = get_problem_and_policy(:tag)
    @test_throws ErrorException run_sim(
        :tag_inf_5;
        no_ask_problem = :tag,
        suggester      = PolicySuggester(π_sugg, 1.0),
        agent          = :bad_agent_name,
        num_sims       = 1,
        num_trials     = 1,
        max_steps      = 10,
    )

    # run_sim_type_eval with invalid agent
    @test_throws ErrorException run_sim_type_eval(:tag, :tag_inf_5;
        agent          = :unknown_agent,
        suggester      = PolicySuggester(π_sugg, 1.0),
        num_sims       = 1,
        num_trials     = 1,
        max_steps      = 10,
    )
end  # testset 11

end  # top-level testset
