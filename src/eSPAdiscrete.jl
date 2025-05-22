using DataFrames
"""
    eSPAdiscrete(K::Int, eps_CL::Float64, eps_E::Float64, tol::Float64, max_iter::Int)

Discrete version of eSPA [Horenko2020]

# Fields
- `K::Int`: Number of Clusters
- `eps_CL::Float64`: Hyperparameter for the classifyer loss
- `eps_E::Float64`: Hyperparameter for the feature selection loss
- `tol::Float64`: Break-condition for optimizaiton

"""
mutable struct eSPAdiscrete
    # Hyperparameters
    K::Int
    eps_CL::Float64
    eps_E::Float64
    tol::Float64
    max_iter::Int

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

function eSPAdiscrete(K::Int, eps_CL::Float64, eps_E::Float64, tol::Float64,max_iter::Int)
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
    return eSPAdiscrete(K, eps_CL, eps_E, tol, max_iter, gamma, W, S, lambda, Pi, D, T, M)
end

"""
    fit!(model::eSPAdiscrete, X::AbstractMatrix, y::AbstractVector)

Train eSPAdiscrete with Data.

# Arguments:
- `model::eSPAdiscrete`: Model instance to train.
- `X::AbstractMatrix`: Data matrix which includes the features. Rows contain the features, Columns the data point.
- `y::AbstractVector`: Data labels. The labels should be Integers between 1 and M.
"""
function fit!(model::eSPAdiscrete, X::AbstractMatrix, y::AbstractVector)
    start_time = time_ns()
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

    opt_times = DataFrame(i=Int[],no_empty_cluster=Int[],sstep=Int[],lambdastep=Int[],gammastep=Int[],wstep=Int[],loss=Int[])
    start_optimization = time_ns()
    while (L_delta > model.tol) && (i <= model.max_iter)
        time_1 = time_ns()
        no_empty_cluster!(model.K, model.gamma, model.T)
        time_2 = time_ns()
        sstep_discrete!(X, model.K, model.gamma, model.S, model.D)
        time_3 = time_ns()
        lambdastep_discrete!(model.K, model.gamma, model.lambda, model.Pi, model.M)
        time_4 = time_ns()
        gammastep_discrete!(X, model.K, model.eps_CL, model.tol, model.gamma, model.W,
                            model.S, model.lambda, model.Pi, model.T, model.M)
        time_5 = time_ns()
        wstep_fuzzy!(X, model.eps_E, model.gamma, model.W, model.S, model.D, model.T)
        time_6 = time_ns()
        L1, L2,
        L3 = losseSPA(X, model.eps_E, model.eps_CL, model.gamma, model.W, model.S,
                      model.lambda, model.Pi, model.D, model.T, model.M)
        time_7 = time_ns()
        L_new = L1 + L2 - L3
        L_delta = abs(L - L_new)
        L = L_new
        println(i, ", Loss: ", L_new, " | $L1, $L2, $(-L3)")
        
        timing_results = (;i = i,no_empty_cluster=time_2 - time_1,sstep=time_3-time_2,lambdastep=time_4-time_3,gammastep=time_5-time_4,wstep=time_6-time_5,loss=time_7-time_6)
        push!(opt_times,timing_results)
        i += 1
    end
    end_time = time_ns()
    return start_time, start_optimization, end_time, opt_times
end

"""
    predict(model::eSPAdiscrete, X::AbstractMatrix)

Calculate predictions.

# Arguments:
- `model::eSPAdiscrete`: Trained instance of eSPAdiscrete.
- `X::AbstractMatrix`: Data matrix which includes the features. Rows contain the features, Columns the data point.
"""
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
