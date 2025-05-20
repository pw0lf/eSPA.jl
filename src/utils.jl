using LinearAlgebra
using JuMP
using Ipopt

######## initializations ########

function initialize_discrete!(K, gamma, W, D, T)
    W[:] = rand(D)
    normalize!(W, 1)

    random_assignment = vcat(collect(1:K), rand(1:K, T - K))
    for (i, k) in enumerate(random_assignment)
        gamma[k, i] = 1
    end
end

function initialize_plus!(X, K, W, S, lambda, D, T, M)
    W[:] = rand(D)
    normalize!(W, 1)

    S[:, :] = X[:, rand(1:T, K)]

    lambda[:, :] = rand(M,K)
    for m in 1:M
        lambda[:, m] = normalize!(lambda[:, m], 1)
    end
end

function initialize_fuzzy!(K, gamma, W, D, T)
    W[:] = rand(D)
    normalize!(W, 1)

    for t in 1:T
        gamma[:, t] = normalize(rand(K), 1)
    end
end

function initialize_GOAL!(K, G, gamma, R, D, T)
    random_assignment = vcat(collect(1:K), rand(1:K, T - K))
    for (i, k) in enumerate(random_assignment)
        gamma[k, i] = 1
    end

    rand_matrix = randn(D, G)
    Q, _ = qr(rand_matrix)
    R[:, :] = Q[:, 1:G]
end

######## loss functions ########

function losseSPA(X, eps_E, eps_CL, gamma, W, S, lambda, Pi, D, T, M)
    loss1 = eps_E/D * sum(W[:] .* log.(W[:]))
    #println("loss1 ",loss1)

    loss2 = 0
    S_mul_gamma = S*gamma
    for d in 1:D
        tmp = 0
        for t in 1:T
            tmp += (X[d, t] - (S_mul_gamma)[d, t])^2
        end
        loss2 += W[d] * tmp
    end
    loss2 = loss2/T
    #println("loss2 ",loss2)

    loss3 = 0
    for m in 1:M
        for t in 1:T
            tmp2 = sum(lambda[m, :] .* gamma[:, t])
            if tmp2 != 0.0
                loss3 += Pi[m, t] * log(tmp2) #TODO: Fixed the -Inf problem by if-clause, but it this right?
            else                              # -Inf Problem: the Prob for a data sample to be labeled m is 0
                loss3 += -100
            end
        end
    end
    loss3 = eps_CL/(T*M) * loss3
    #println("loss3: ",loss3)

    return loss1, loss2, loss3
end

function lossGOAL(X, eps_CL, gamma, R, S, lambda, Pi, D, T, M)
    RSGamma = R * S * gamma
    loss1 = 0
    for d in 1:D, t in 1:T
        loss1 += (X[d, t] - RSGamma[d, t])^2
    end
    loss1 = loss1/T

    loss2 = 0
    for m in 1:M
        for t in 1:T
            tmp2 = sum(lambda[m, :] .* gamma[:, t])
            if tmp2 != 0.0
                loss2 += Pi[m, t] * log(tmp2) #TODO: Fixed the -Inf problem by if-clause, but it this right?
            end                              # -Inf Problem: the Prob for a data sample to be labeled m is 0
        end
    end
    loss2 = eps_CL/(T*M) * loss2

    return loss1, loss2
end

######## fuzzy steps ########
#TODO: Alles steps bis auf W-step

function wstep_fuzzy!(X, eps_E, gamma, W, S, D, T)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    @variable(model, w[1:D] >= 0)

    function obj_function(w)
        val = 0
        for d in 1:D
            tmp = 0
            for t in 1:T
                tmp += (X[d, t] - (S*gamma)[d, t])^2
            end
            val += w[d] * (eps_E * log(w[d]) + D/T * tmp)
        end
        return val/D
    end
    @objective(model, Min, obj_function(w))
    @constraint(model, sum(w) == 1)
    optimize!(model)
    W[:] = value.(w)
end

function lambdastep_fuzzy!(K, gamma, lambda, Pi, M, T)
    function obj_function(l)
        val = 0
        for m in 1:M, t in 1:T
            val += Pi[m, t] * log(sum(l[m, :] .* gamma[:, t]))
        end
        return val
    end

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    @variable(model, l[1:M, 1:K] >= 0)
    @objective(model, Min, obj_function(l))
    @constraint(model, [i = 1:K], sum(l[:, i]) == 1)
    optimize!(model)
    lambda[:, :] = value.(l)
end

function gammastep_fuzzy!(T, M, Pi, lambda, eps_CL, W, X, S, K, gamma)
    for t in 1:T
        function obj_function(gt)
            val1 = 0
            for m in 1:M
                val1 += Pi[m, t] * log(sum(lambda[m, :] .* gt))
            end
            val1 = eps_CL/M * val1

            return sum(W .* (X[:, t] .- (S * gt))) - val1
        end
        model = Model(Ipopt.Optimizer)
        set_optimizer_attribute(model, "print_level", 0)
        @variable(model, gt[1:K] >= 0)
        @objective(model, Min, obj_function(gt))
        @constraint(model, sum(gt) == 1)
        optimize!(model)
        #optimize!(model; quiet=true)
        gamma[:, t] = value.(gt)
    end
end

#TODO: test since it's written with chatgpt
function sstep_fuzzy!(D, W, T, S, gamma, X)
    for d in 1:D
        #Wd = Diagonal(fill(W[d], T))
        A = sqrt(W[d]) .* gamma'   # T x K
        b = sqrt(W[d]) .* X[d, :]
        S[d, :] = (A \ b)'  
        #S[d, :] = (gamma * Wd * gamma') \ (gamma * Wd * X[d, :])
    end
end

######## GOAL steps ########

function sstep_goal!(X, K, gamma, S, R, D)
    RS = zeros(D, K)
    for k in 1:K, d in 1:D
        RS[d, k] = sum(X[d, :] .* gamma[k, :])/sum(gamma[k, :])
    end
    S[:, :] = R' * RS
end

function gammastep_goal!(X, K, eps_CL, tol, gamma, S, R, Pi, lambda, T, M)
    gamma[:, :] = zeros(K, T)
    for t in 1:T
        temp_results = zeros(K)
        for j in 1:K
            temp_results[j] = -eps_CL/M * sum(Pi[:, t] .* max.(log.(lambda[:, j]), tol)) +
                              sum((X[:, t]-(R*S)[:, j]) .^ 2)
        end
        k = argmin(temp_results)
        gamma[k, t] = 1
    end
end

#Re-use lambda discrete

function rstep_goal!(X, G, gamma, S, R, D)
    XgammaS = X * gamma' * S'
    U, temp_sigma, Vt = svd(XgammaS, full = true)
    sigma = zeros(D, G)
    A_inv = zeros(G, G)
    for g in 1:G
        sigma[g, g] = temp_sigma[g]
        A_inv[g, g] = 1/temp_sigma[g]
    end
    I = sigma * A_inv
    R[:, :] = U * I * Vt
end

######## discrete steps ########

function sstep_discrete!(X, K, gamma, S, D)
    for k in 1:K, d in 1:D
        S[d, k] = sum(X[d, :] .* gamma[k, :])/sum(gamma[k, :])
    end
end

function lambdastep_discrete!(K, gamma, lambda, Pi, M)
    lambda_temp = Pi * gamma'
    for k in 1:K, m in 1:M
        lambda[m, k] = lambda_temp[m, k]/sum(lambda_temp[:, k])
    end
end

function wstep_discrete!(X, eps_E, gamma, W, S, D, T)
    sum_a = zeros(D)
    for t in 1:T
        sum_a += (X[:, t] - (S*gamma)[:, t]) .^ 2
    end
    a = exp.(-(D/(T*eps_E)) .* sum_a .- ones(D))

    W[:] = a ./ sum(a)
end

function gammastep_discrete!(X, K, eps_CL, tol, gamma, W, S, lambda, Pi, T, M)
    gamma[:, :] = zeros(K, T)
    for t in 1:T
        temp_results = zeros(K)
        for j in 1:K
            temp_results[j] = -eps_CL/M * sum(Pi[:, t] .* max.(log.(lambda[:, j]), tol)) +
                              sum(W[:] .* (X[:, t]-S[:, j]) .^ 2)
        end
        k = argmin(temp_results)
        gamma[k, t] = 1
    end
end

# Problem: Manche Cluster haben garkeine Elemente
# Idee1: Ist ein Cluster leer, so nehme den Datenpunkt der am weitesten von seinem Center entfernt ist als neues Cluster
# Idee2: Splite größtes Cluster in 2.
# Idee3: Verschiebe zufälligen Datenpunkt in das leere CLuster

# Hier wird Idee 2 implementiert:

######## cluster utils ########

function no_empty_cluster!(K, gamma, T)
    cs = check_clustersizes(K, gamma, T)
    while (0.0 in cs)
        empty_clst = findall(==(0.0), cs)[1]
        big_clst = argmax(cs)
        split_cluster!(gamma, empty_clst, big_clst)
        cs = check_clustersizes(K, gamma, T)
    end
end

function check_clustersizes(K, gamma, T)
    cluster_size = zeros(K)
    cluster_assignment = argmax(gamma, dims = 1)
    cluster_assignment = map(x -> x[1], cluster_assignment)
    for t in 1:T
        cluster_size[cluster_assignment[t]] += 1
    end
    return cluster_size
end

function split_cluster!(gamma, empty_clst, big_clst)
    ind_big_clst = findall(==(1), gamma[big_clst, :])
    ind_new = ind_big_clst[2:2:end]
    gamma[big_clst, ind_new] .= 0
    gamma[empty_clst, ind_new] .= 1
end

######## predictions ########

function prediction_gamma_discrete!(gamma, X, W, S, K, T)
    for t in 1:T
        temp_results = zeros(K)
        for j in 1:K
            temp_results[j] = sum(W[:] .* ((X[:, t] - S[:, j]) .^ 2))
        end
        k = argmin(temp_results)
        gamma[k, t] = 1
    end
end

function prediction_gamma_fuzzy!(gamma, X, W, S, K, T)
    for t in 1:T
        function obj_function(gt)
            return sum(W .* (X[:, t] .- (S * gt)))
        end
        model = Model(Ipopt.Optimizer)
        set_optimizer_attribute(model, "print_level", 0)
        @variable(model, gt[1:K] >= 0)
        @objective(model, Min, obj_function(gt))
        @constraint(model, sum(gt) == 1)
        optimize!(model)
        gamma[:, t] = value.(gt)
    end
end

function prediction_gamma_GOAL!(gamma, X, R, S, K, T)
    for t in 1:T
        temp_results = zeros(K)
        for j in 1:K
            temp_results[j] = sum((X[:, t] - (R*S)[:, j]) .^ 2)
        end
        k = argmin(temp_results)
        gamma[k, t] = 1
    end
end
