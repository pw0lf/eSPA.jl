using Test
using eSPA
using LinearAlgebra
include("../src/utils.jl")

@testset "initializing" begin
    K = 5
    D = 10
    T = 20
    M = 3
    X = randn(D, T)

    @testset "init discrete" begin
        W = zeros(D)
        gamma = zeros(K, T)
        initialize_discrete!(K, gamma, W, D, T)
        # Test if W is positiv and sums up to 1
        @test sum(W) ≈ 1.0 atol=0.01
        @test sum(W .< 0) == 0
        @testset "Gamma is stochastic and discrete" begin
            for t in 1:T
                count_ones = count(gamma[:, t] .== 1)
                @test count_ones == 1

                count_zeros = count(gamma[:, t] .== 0)
                @test count_zeros == K - 1
            end
        end
    end

    @testset "init plus" begin
        W = zeros(D)
        S = zeros(D, K)
        lambda = zeros(M, K)
        initialize_plus!(X, K, W, S, lambda, D, T, M)

        # Test if W is positiv and sums up to 1
        @test sum(W) ≈ 1.0 atol=0.01
        @test sum(W .< 0) == 0

        # Test S?

        # Test if lambda is stochastic 
        @testset "lambda stochastic" begin
            for m in 1:M
                @test sum(lambda[:, m]) ≈ 1 atol=0.01
                @test sum(lambda[:, m] .< 0) == 0
            end
        end
    end

    @testset "init GOAL" begin
        G = 8
        R = zeros(D, G)
        gamma = zeros(K, T)
        initialize_GOAL!(K, G, gamma, R, D, T)

        @testset "Gamma is stochastic and discrete" begin
            for t in 1:T
                count_ones = count(gamma[:, t] .== 1)
                @test count_ones == 1

                count_zeros = count(gamma[:, t] .== 0)
                @test count_zeros == K - 1
            end
        end

        @test sum(R' * R - Matrix(1.0I, G, G)) ≈ 0 atol=0.01
    end

    @testset "init fuzzy" begin end
end

@testset "loss functions" begin
    D = 10
    T = 10
    M = 3
    eps_E = 2.0
    eps_CL = 3.0

    X = Matrix(1.0I, D, D)
    S = X[:, 1:5]
    gamma = hcat(Matrix(1I, 5, 5), Matrix(1I, 5, 5))
    W = 1/D .* ones(D)

    Pi = [1 1 0 0 0 1 1 0 0 0;
          0 0 1 1 0 0 0 1 1 0;
          0 0 0 0 1 0 0 0 0 1]

    lambda = [1 1 0 0 0;
              0 0 1 1 0;
              0 0 0 0 1]

    @testset "loss eSPA" begin
        L1, L2, L3 = losseSPA(X, eps_E, eps_CL, gamma, W, S, lambda, Pi, D, T, M)
        @test L2 ≈ 1/D atol=0.01
        @test L1 ≈ (1/D) * log(1/D) * eps_E atol=0.01
        @test L3 ≈ 0.0 atol=0.01
    end

    @testset "loss GOAL" begin
        G = 10
        R = Matrix(1.0I, G, G)
        L1, L2 = lossGOAL(X, eps_CL, gamma, R, S, lambda, Pi, D, T, M)
        @test L1 ≈ 1.0 atol=0.01
        @test L2 ≈ 0.0 atol=0.01
    end
end

@testset "fuzzy steps" begin end

@testset "GOAL steps" begin
    @testset "sstep" begin
        D = 5
        T = 10
        K = 5
        G = 5
        R = Matrix(1.0I, G, G)
        X = zeros(5, 10)
        for t in 1:T
            X[:, t] = t*ones(D)
        end
        gamma = hcat(Matrix(1.0I, D, D), Matrix(1.0I, D, D)[end:-1:1, :])
        S = zeros(D, K)
        sstep_goal!(X, K, gamma, S, R, D)
        @test sum(S .!= 11/2) == 0
    end

    @testset "gammastep" begin
        T = 8
        D = 2
        K = 2
        M = 2
        X = [-0.5 -1.5 -1.0 -1.0 0.5 1.5 1.0 1.0; 0 0 0.5 -0.5 0 0 0.5 -0.5]
        S = [-1 1; 0 0]
        Pi = [1 1 1 1 0 0 0 0; 0 0 0 0 1 1 1 1]
        lambda = [1 0; 0 1]
        G = 2
        R = Matrix(1.0I, G, G)
        eps_CL = 1.0
        tol = 0.001
        gamma = zeros(K, T)
        gammastep_goal!(X, K, eps_CL, tol, gamma, S, R, Pi, lambda, T, M)
        gamma_true = [1 1 1 1 0 0 0 0; 0 0 0 0 1 1 1 1]
        @test gamma == gamma_true
    end

    @testset "rstep" begin
        T = 6
        D = 3
        G = 2
        X = [1 1 1 0 0 0; 0 0 0 -1 -1 -1; 0 0 0 0 0 0]
        gamma = [1 1 1 0 0 0; 0 0 0 1 1 1]
        S = [1 -1; 1 0]
        R = zeros(D, G)

        rstep_goal!(X, G, gamma, S, R, D)

        R_true = [1 0; 0 -1; 0 0]
        for d in 1:D, g in 1:G
            @test R[d, g] .≈ R_true[d, g] atol = 0.01
        end
    end
end

@testset "discrete steps" begin
    @testset "sstep" begin
        D = 5
        T = 10
        K = 5
        X = zeros(5, 10)
        for t in 1:T
            X[:, t] = t*ones(D)
        end
        gamma = hcat(Matrix(1.0I, D, D), Matrix(1.0I, D, D)[end:-1:1, :])
        S = zeros(D, K)
        sstep_discrete!(X, K, gamma, S, D)
        @test sum(S .!= 11/2) == 0
    end

    @testset "lambdastep" begin
        T = 6
        M = 3
        K = 3
        Pi = hcat(Matrix(1.0I, M, M), Matrix(1.0I, M, M))
        gamma = hcat(Matrix(1.0I, M, M), Matrix(1.0I, M, M))
        lambda = zeros(M, K)
        lambdastep_discrete!(K, gamma, lambda, Pi, M)
        @test lambda ≈ Matrix(1.0I, 3, 3) atol=0.01
    end

    @testset "gammastep" begin
        T = 8
        D = 2
        K = 2
        M = 2
        X = [-0.5 -1.5 -1.0 -1.0 0.5 1.5 1.0 1.0; 0 0 0.5 -0.5 0 0 0.5 -0.5]
        S = [-1 1; 0 0]
        Pi = [1 1 1 1 0 0 0 0; 0 0 0 0 1 1 1 1]
        lambda = [1 0; 0 1]
        W = [0.5, 0.5]
        eps_CL = 1.0
        tol = 0.001
        gamma = zeros(K, T)
        gammastep_discrete!(X, K, eps_CL, tol, gamma, W, S, lambda, Pi, T, M)
        gamma_true = [1 1 1 1 0 0 0 0; 0 0 0 0 1 1 1 1]
        @test gamma == gamma_true
    end

    @testset "wstep" begin
        D = 10
        T = 10
        M = 3
        eps_E = 2.0

        X = Matrix(1.0I, D, D)
        S = X[:, 1:5]
        gamma = hcat(Matrix(1I, 5, 5), Matrix(1I, 5, 5))
        W = zeros(D)
        wstep_discrete!(X, eps_E, gamma, W, S, D, T)
        for d in 1:D
            @test W[d] ≈ 1/D atol=0.01
        end
        @test sum(W .≈ 1/D) == D
    end
end

@testset "cluster utils" begin
    K = 3
    T = 10
    gamma = [1 1 1 0 0 0 0 0 0 0; 0 0 0 1 1 1 1 0 0 0; 0 0 0 0 0 0 0 1 1 1]
    @test check_clustersizes(K, gamma, T) == [3, 4, 3]

    no_empty_cluster!(K, gamma, T)
    @test check_clustersizes(K, gamma, T) == [3, 4, 3]

    gamma = [1 1 1 1 1 1 1 0 0 0; 0 0 0 0 0 0 0 1 1 1; 0 0 0 0 0 0 0 0 0 0]
    split_cluster!(gamma, 3, 1)
    @test check_clustersizes(K, gamma, T) == [4, 3, 3]

    gamma = [1 1 1 1 1 1 1 0 0 0; 0 0 0 0 0 0 0 1 1 1; 0 0 0 0 0 0 0 0 0 0]
    no_empty_cluster!(K, gamma, T)
    @test check_clustersizes(K, gamma, T) == [4, 3, 3]
end

@testset "predictions" begin
    @testset "discrete" begin
        T = 8
        D = 2
        K = 2
        M = 2
        X = [-0.5 -1.5 -1.0 -1.0 0.5 1.5 1.0 1.0; 0 0 0.5 -0.5 0 0 0.5 -0.5]
        S = [-1 1; 0 0]
        lambda = [1 0; 0 1]
        W = [0.5, 0.5]
        gamma = zeros(K, T)

        prediction_gamma_discrete!(gamma, X, W, S, K, T)

        gamma_true = [1 1 1 1 0 0 0 0; 0 0 0 0 1 1 1 1]
        @test gamma == gamma_true
    end

    @testset "goal" begin
        T = 8
        D = 2
        K = 2
        M = 2
        X = [-0.5 -1.5 -1.0 -1.0 0.5 1.5 1.0 1.0; 0 0 0.5 -0.5 0 0 0.5 -0.5]
        S = [-1 1; 0 0]
        Pi = [1 1 1 1 0 0 0 0; 0 0 0 0 1 1 1 1]
        lambda = [1 0; 0 1]
        G = 2
        R = Matrix(1.0I, G, G)
        gamma = zeros(K, T)
        prediction_gamma_GOAL!(gamma, X, R, S, K, T)
        gamma_true = [1 1 1 1 0 0 0 0; 0 0 0 0 1 1 1 1]
        @test gamma == gamma_true
    end

    @testset "fuzzy" begin end
end
