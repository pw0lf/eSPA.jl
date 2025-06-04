"""
    eSPAhybrid(K::Int, eps_CL::Float64, eps_E::Float64, tol::Float64, max_iter::Int, num_fuzzy_steps::Int)

# Fields
- `K::Int`: Number of Clusters
- `eps_CL::Float64`: Hyperparameter for the classifyer loss
- `eps_E::Float64`: Hyperparameter for the feature selection loss
- `tol::Float64`: Break-condition for discrete optimizaiton
- `max_iter::Int`: 
- `num_fuzzy_steps::Int`: NUmber of fuzzy steps after discrete steps are finished

"""
mutable struct eSPAhybrid
    # Hyperparameters
    K::Int
    eps_CL::Float64
    eps_E::Float64
    tol::Float64
    max_iter::Int
    num_fuzzy_steps::Int

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

function eSPAhybrid(K::Int, eps_CL::Float64, eps_E::Float64, tol::Float64, max_iter::Int,
                    num_fuzzy_steps::Int)
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
    return eSPAhybrid(K, eps_CL, eps_E, tol, max_iter, num_fuzzy_steps, gamma, W, S, lambda,
                      Pi, D, T, M)
end

"""
    fit!(model::eSPAhybrid, X::AbstractMatrix, y::AbstractVector)

Train eSPAhybrid with Data.

# Arguments:
- `model::eSPAhybrid`: Model instance to train.
- `X::AbstractMatrix`: Data matrix which includes the features. Rows contain the features, Columns the data point.
- `y::AbstractVector`: Data labels. The labels should be Integers between 1 and M.
"""
function fit!(model::eSPAhybrid, X::AbstractMatrix, y::AbstractVector)
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
    model.S = zeros(model.D, model.K)
    model.lambda = zeros(model.M, model.K)

    initialize_plus!(X, model.K, model.W, model.S, model.lambda, model.D, model.T, model.M)

    i = 1
    L = Inf
    L_delta = Inf

    opt_times_discr = DataFrame(i = Int[], no_empty_cluster = Int[], sstep = Int[],
                                lambdastep = Int[], gammastep = Int[], wstep = Int[],
                                loss = Int[])
    start_optimization = time_ns()
    println("Starting discrete phase")
    while L_delta > model.tol && i <= model.max_iter
        time_1 = time_ns()
        gammastep_discrete!(X, model.K, model.eps_CL, model.tol, model.gamma, model.W,
                            model.S, model.lambda, model.Pi, model.T, model.M)
        time_2 = time_ns()
        no_empty_cluster!(model.K, model.gamma, model.T)
        time_3 = time_ns()
        wstep_discrete!(X, model.eps_E, model.gamma, model.W, model.S, model.D, model.T)
        time_4 = time_ns()
        sstep_discrete!(X, model.K, model.gamma, model.S, model.D)
        time_5 = time_ns()
        lambdastep_discrete!(model.K, model.gamma, model.lambda, model.Pi, model.M)
        time_6 = time_ns()

        L1, L2,
        L3 = losseSPA(X, model.eps_E, model.eps_CL, model.gamma, model.W, model.S,
                      model.lambda, model.Pi, model.D, model.T, model.M)
        time_7 = time_ns()
        L_new = L1 + L2 - L3
        L_delta = abs(L - L_new)
        L = L_new
        println(i, ", Loss: ", L_new, " | $L1, $L2, $(-L3)")
        #println("delta: $L_delta")
        #println("W: $(model.W)")
        timing_results = (; i = i, no_empty_cluster = time_3 - time_2,
                          sstep = time_5-time_4, lambdastep = time_6-time_5,
                          gammastep = time_2-time_1, wstep = time_4-time_3,
                          loss = time_7-time_6)
        push!(opt_times_discr, timing_results)
        i += 1
    end
    opt_times_fuzzy = DataFrame(i = Int[], sstep = Int[], lambdastep = Int[],
                                gammastep = Int[], wstep = Int[], loss = Int[])
    start_fuzzy = time_ns()
    println("Starting fuzzy phase")
    for fuzzy_step in 1:model.num_fuzzy_steps
        #TODO: Change
        #no_empty_cluster!(model.K,model.gamma,model.T) #TODO: Check if this is needed
        time_1 = time_ns()
        sstep_fuzzy!(model.D, model.W, model.T, model.S, model.gamma, X)
        time_2 = time_ns()
        lambdastep_fuzzy!(model.K, model.gamma, model.lambda, model.Pi, model.M, model.T)
        time_3 = time_ns()
        gammastep_fuzzy!(model.T, model.M, model.Pi, model.lambda, model.eps_CL, model.W, X,
                         model.S, model.K, model.gamma)
        time_4 = time_ns()
        wstep_fuzzy!(X, model.eps_E, model.gamma, model.W, model.S, model.D, model.T)
        time_5 = time_ns()
        L1, L2,
        L3 = losseSPA(X, model.eps_E, model.eps_CL, model.gamma, model.W, model.S,
                      model.lambda, model.Pi, model.D, model.T, model.M)
        time_6 = time_ns()
        L_new = L1 + L2 - L3
        L_delta = abs(L - L_new)
        L = L_new
        println(fuzzy_step, ", Loss: ", L_new, " | $L1, $L2, $(-L3)")
        timing_results = (; i = fuzzy_step, sstep = time_2-time_1,
                          lambdastep = time_3-time_2, gammastep = time_4-time_3,
                          wstep = time_5-time_4, loss = time_6-time_5)
        push!(opt_times_fuzzy, timing_results)
    end
    end_time = time_ns()
    return start_time, start_optimization, start_fuzzy, end_time, opt_times_discr,
           opt_times_fuzzy
end

"""
    predict(model::eSPAhybrid, X::AbstractMatrix)

Calculate predictions.

# Arguments:
- `model::eSPAhybrid`: Trained instance of eSPAhybrid.
- `X::AbstractMatrix`: Data matrix which includes the features. Rows contain the features, Columns the data point.
"""
function predict(model::eSPAhybrid, X::AbstractMatrix)
    D, T = size(X)
    if model.D != D
        throw(ArgumentError("The input samples should have the same dimension D as the Trainingdata. Input = $D, Trainingdata = $(model.D)."))
    end
    gamma = zeros(model.K, T)
    if model.num_fuzzy_steps <= 0
        prediction_gamma_discrete!(gamma, X, model.W, model.S, model.K, T)
    else
        prediction_gamma_fuzzy!(gamma, X, model.W, model.S, model.K, T)
    end
    Pi = model.lambda * gamma
    pred = argmax(Pi, dims = 1)
    pred = map(x -> x[1], pred)[1, :]
    return pred
end
