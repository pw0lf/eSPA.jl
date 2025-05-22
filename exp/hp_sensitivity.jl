using eSPA
using Random
using ArgParse
using Hyperopt
using MLJ
using MLJBase

Random.seed!(50)
include("synthdata2.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--dim","-d"
            help = "Dimension of dataset"
            arg_type = Int
            default = 0
            required = true

        "--num_samples", "-n"
            help = "Number of samples of the dataset"
            arg_type = Int
            default = 0
            required = true

        "--iter","-i"
            help = "Number of iterations for hyperparameter-tuning"
            arg_type = Int
            default = 0
            required = true
    end
    return parse_args(s)
end

args = parse_commandline()

d = args["dim"]
n = args["num_samples"]
iter = args["iter"]

X, Y = make_synth_data2(n, d)

train_indices, test_indices = partition(1:size(X, 2), 0.8, shuffle = true)
X_train = X[:, train_indices]
X_test = X[:, test_indices]
y_train = Y[train_indices]
y_test = Y[test_indices]

function train_and_eval_model(K,eps_CL,eps_E,tol)
    model = eSPAplus(K,eps_CL,eps_E,tol)
    eSPA.fit!(model, X_train, y_train)
    y_pred = eSPA.predict(model, X_test)
    return accuracy(y_pred, y_test)
end

#TODO: Anpassen, so dass Grid search gemacht wird
ho = @hyperopt for i = iter,
            K = LinRange(5,200,200-4),
            eps_E = LinRange(1,500,1000),
            eps_CL = LinRange(1,500,1000),
            tol = exp10.(-LinRange(1,8,1000))
    @show cost = train_and_eval_model(K,eps_CL,eps_E,tol)
end

println(ho.maximizer)
println(ho.maximum)
