mutable struct eSPAdiscrete
    # Hyperparameters
    K::Int
    eps_CL::Float64
    eps_E::Float64
    tol::Float64

    # Parameters
    gamma::AbstractMatrix
    W::AbstractVector
    S::AbstractMatrix
    lambda::AbstractMatrix
    Pi::AbstractMatrix

    # Variables of the input data
    D::Int
    T::Int
    M::Int
end

function eSPAdiscrete(K::Int, eps_CL::Float64, eps_E::Float64, tol::Float64)
    if eps_CL < 0.0
        throw(ArgumentError("eps_CL must be non-negative"))
    end
    if eps_E < 0.0
        throw(ArgumentError("eps_E must be non-negative"))
    end
    if tol < 0.0
        throw(ArgumentError("tol must be non-negative"))
    end
    gamma = [;;] # K rows, T columns
    W = [] # D rows
    S = [;;] # D rows, K columns  
    lambda = [;;] # M rows, K columns
    Pi = [;;] # M rows, T columns

    D = 0
    T = 0
    M = 0
    return eSPAdiscrete(K, eps_CL, eps_E, tol, gamma, W, S, lambda, Pi, D, T, M)
end

function fit!(model::eSPAdiscrete, X::AbstractMatrix, y::AbstractVector)
    model.D, model.T = size(X)
    T_y = size(y)[1]

    model.M = maximum(y)

    if model.T != T_y
        throw(ArgumentError("The number of samples in X and y must be the same"))
    end

    # Pi is one-hot-encoding for y
    model.Pi = zeros(model.M, model.T)

    for (i, m) in enumerate(y)
        model.Pi[m, i] = 1
    end

    model.gamma = zeros(model.K, model.T)
    model.W = zeros(model.D)

    initialize_discrete!(model.K, model.gamma, model.W, model.D, model.T)

    model.S = zeros(model.D, model.K)
    model.lambda = zeros(model.M, model.K)

    i = 1
    L = Inf
    L_delta = Inf

    clusters = []
    push!(clusters, copy(model.S))
    while L_delta > model.tol
        no_empty_cluster!(model.K, model.gamma, model.T)
        sstep_discrete!(X, model.K, model.gamma, model.S, model.D)
        lambdastep_discrete!(model.K, model.gamma, model.lambda, model.Pi, model.M)
        gammastep_discrete!(X, model.K, model.eps_CL, model.tol, model.gamma, model.W,
                            model.S, model.lambda, model.Pi, model.T, model.M)
        wstep_fuzzy!(X, model.eps_E, model.gamma, model.W, model.S, model.D, model.T)
        L1, L2,
        L3 = losseSPA(X, model.eps_E, model.eps_CL, model.gamma, model.W, model.S,
                      model.lambda, model.Pi, model.D, model.T, model.M)
        L_new = L1 + L2 - L3
        L_delta = abs(L - L_new)
        L = L_new
        println(i, ", Loss: ", L_new, " | $L1, $L2, $(-L3)")
        #println("delta: $L_delta")
        #println("W: $(model.W)")
        i += 1
        push!(clusters, copy(model.S))
    end
    return clusters
end

function predict(model::eSPAdiscrete, X::AbstractMatrix)
    D, T = size(X)
    if model.D != D
        throw(ArgumentError("The input samples should have the same dimension D as the Trainingdata. Input = $D, Trainingdata = $(model.D)."))
    end
    gamma = zeros(model.K, T)
    prediction_gamma_discrete!(gamma, X, model.W, model.S, model.K, T)
    Pi = model.lambda * gamma
    pred = argmax(Pi, dims = 1)
    pred = map(x -> x[1], pred)[1, :]
    return pred
end
