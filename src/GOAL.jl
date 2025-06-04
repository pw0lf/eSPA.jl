"""
    GOAL(K::Int, eps_CL::Float64, G::Int, tol::Float64, max_iter::Int)

Implementation of GOAL [Vecchi+2024]

# Fields
- `K::Int`: Number of Clusters
- `eps_CL::Float64`: Hyperparameter for the classifyer loss
- `G::Int`: Dimension of Gauge
- `tol::Float64`: Break-condition for optimizaiton
- `max_iter::Int`: Maximum number of optimization iterations

"""
mutable struct GOAL
    # Hyperparameters
    K::Int
    eps_CL::Float64
    G::Int
    tol::Float64
    max_iter::Int

    # Parameters
    gamma::AbstractMatrix
    R::AbstractMatrix
    S::AbstractMatrix
    lambda::AbstractMatrix
    Pi::AbstractMatrix

    # Variables of the input data
    D::Int
    T::Int
    M::Int
end

function GOAL(K::Int, eps_CL::Float64, G::Int, tol::Float64, max_iter::Int)
    if eps_CL < 0.0
        throw(ArgumentError("eps_CL must be non-negative"))
    end
    if tol < 0.0
        throw(ArgumentError("tol must be non-negative"))
    end
    gamma = [;;] # K rows, T columns
    R = [;;] # D rows, G columns
    S = [;;] # D rows, K columns  
    lambda = [;;] # M rows, K columns
    Pi = [;;] # M rows, T columns

    D = 0
    T = 0
    M = 0
    return GOAL(K, eps_CL, G, tol, max_iter, gamma, R, S, lambda, Pi, D, T, M)
end

"""
    fit!(model::GOAL, X::AbstractMatrix, y::AbstractVector)

Train GOAL with Data.

# Arguments:
- `model::GOAL`: Model instance to train.
- `X::AbstractMatrix`: Data matrix which includes the features. Rows contain the features, Columns the data point.
- `y::AbstractVector`: Data labels. The labels should be Integers between 1 and M.
"""
function fit!(model::GOAL, X::AbstractMatrix, y::AbstractVector)
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
    model.R = zeros(model.D, model.G)
    model.S = zeros(model.G, model.K)
    model.lambda = zeros(model.M, model.K)

    initialize_GOAL!(model.K, model.G, model.gamma, model.R, model.D, model.T)

    i = 1
    L = Inf
    L_delta = Inf

    while L_delta > model.tol && i <= model.max_iter
        sstep_goal!(X, model.K, model.gamma, model.S, model.R, model.D)
        gammastep_goal!(X, model.K, model.eps_CL, model.tol, model.gamma, model.S, model.R,
                        model.Pi, model.lambda, model.T, model.M)
        no_empty_cluster!(model.K, model.gamma, model.T)
        lambdastep_discrete!(model.K, model.gamma, model.lambda, model.Pi, model.M)
        rstep_goal!(X, model.G, model.gamma, model.S, model.R, model.D)

        L1,
        L2 = lossGOAL(X, model.eps_CL, model.gamma, model.R, model.S, model.lambda,
                      model.Pi, model.D, model.T, model.M)
        L_new = L1 - L2
        L_delta = abs(L - L_new)
        L = L_new
        println(i, ", Loss: ", L_new, " | $L1, $(-L2)")
        i += 1
    end
end

"""
    predict(model::GOAL, X::AbstractMatrix)

Calculate predictions.

# Arguments:
- `model::GOAL`: Trained instance of GOAL.
- `X::AbstractMatrix`: Data matrix which includes the features. Rows contain the features, Columns the data point.
"""
function predict(model::GOAL, X::AbstractMatrix)
    D, T = size(X)
    if model.D != D
        throw(ArgumentError("The input samples should have the same dimension D as the Trainingdata. Input = $D, Trainingdata = $(model.D)."))
    end
    gamma = zeros(model.K, T)
    prediction_gamma_GOAL!(gamma, X, model.R, model.S, model.K, T)
    Pi = model.lambda * gamma
    pred = argmax(Pi, dims = 1)
    pred = map(x -> x[1], pred)[1, :]
    return pred
end
