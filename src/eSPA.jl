module eSPA

struct eSPAdiscrete
    # Hyperparameters
    K::Int
    eps_CL::Float
    eps_E::Float
    tol::Float

    # Parameters
    gamma::AbstractMatrix
    W::AbstractMatrix
    S::AbstractMatrix
    lambda::AbstractMatrix

    # Variables of the input data
    D::Int
    T::Int
    M::Int
end

function eSPAdiscrete(K::Int, eps_CL::Float, eps_E::Float, tol::Float)
    if eps_CL < 0.0
        throw(ArgumentError("eps_CL must be non-negative"))
    end
    if eps_E < 0.0
        throw(ArgumentError("eps_E must be non-negative"))
    end
    if tol < 0.0
        throw(ArgumentError("tol must be non-negative"))
    end
    gamma = Matrix{Float64}() # T rows, K columns
    W = Matrix{Float64}() # D rows, 1 column
    S = Matrix{Float64}() # K rows, D columns 
    lambda = Matrix{Float64}() # K rows, M columns

    D = 0
    T = 0
    M = 0
    return eSPAdiscrete(K, eps_CL, eps_E, tol, gamma, W, S, lambda,D,T,M)
end

function fit!(model::eSPAdiscrete, X::AbstractMatrix, y::AbstractMatrix)

    model.T, model.D = size(X)
    T_y, model.M = size(y)

    if model.T != T_y
        throw(ArgumentError("The number of samples in X and y must be the same"))
    end

    model.gamma = randn(model.T,model.K)
    model.W = randn(model.D,1)

    model.S = zeros(model.K,model.D)
    model.lambda = zeros(model.K,model.M)

    i = 1
    L = Inf 
    L_delta = Inf
    while L_delta > model.tol
        model.S = sstep_discrete!(model.S,X,model.gamma,model.D,model.K)
        model.lambda = lambdastep_discrete!(model.lambda,y,model.gamma)
        model.gamma = gammastep_discrete!(model.gamma,X,y,model.S,model.W,model.lambda,model.eps_CL,model.M,model.tol)
        model.W = wstep(X,model.S,model.gamma)
        L_new = losseSPA(X,y,model.gamma,model.S,model.W,model.lambda)
        L_delta = L - L_new
        L = L_new
        i += 1
    end

end

function predict(model::eSPAdiscrete, X::AbstractMatrix)

end

function losseSPA(X,y,gamma,S,W,lambda)

end

function sstep_discrete!(S,X,gamma,D,K)
    for k in 1:K,d in 1:D
        S[k,d] = sum(X[:,d] .* gamma[:,k])/sum(gamma[:,k])
    end
end

function lambdastep_discrete!(lambda,y,gamma)
    lambda_temp = y' * gamma
    for k in 1:K,m in 1:M
        lambda[k,m] = lambda_temp[k,m]/sum(lambda_temp[k,:])
    end
end

function gammastep_discrete!(gamma,X,y,S,W,lambda,eps_CL,M, tol)
    for t in 1:T,k in 1:K
        temp_results = zeros(K)
        for j in 1:K
            temp_results[j] = -eps_CL/M * sum(y[t,:].* max.(log.(lambda[j,:]), tol)) + sum(W[:,1].* (X[t,:]-S[j,:]).^2)
            if argmin(temp_results) == k
                gamma[t,k] = 1
            else
                gamma[t,k] = 0
            end
        end
    end
end

function wstep(X,S,gamma)

end

end # module eSPA
