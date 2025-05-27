using eSPA
using Random
using Hyperopt
using MLJ
using MLJBase
using Dates
using CSV
using JSON
using ArgParse
using DataFrames
using IterTools


Random.seed!(50)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
experiment_dir = "exp_hp_sens" * timestamp
mkpath(experiment_dir)

csv_path = joinpath(experiment_dir,"results.csv")
header = DataFrame(K=Int[],eps_CL=Float64[],eps_E=Float64[],tol=Float64[],time=Float64[],acc=Float64[])
CSV.write(csv_path,header)

df = CSV.read("darwin.csv", DataFrame)
X = Matrix{Float64}(df[:, 2:end-1])
X = X'
label_map = Dict("P" => 1, "H" => 2)
Y = [label_map[label] for label in df[:,end]]

train_indices, test_indices = MLJBase.partition(1:size(X, 2), 0.8, shuffle = true)
X_train = X[:, train_indices]
X_test = X[:, test_indices]
y_train = Y[train_indices]
y_test = Y[test_indices]

global iteration_counter = 0
function train_and_eval_model(K,eps_CL,eps_E,tol)
    global iteration_counter += 1
    model = eSPAplus(K,eps_CL,eps_E,tol,100)
    start_time, start_optimization, end_time, opt_times = eSPA.fit!(model, X_train, y_train)
    y_pred = eSPA.predict(model, X_test)
    acc = accuracy(y_pred, y_test)
    complete_time = (end_time-start_time)/1e9
    #Track data
    result_row = DataFrame(K=[K],eps_CL = [eps_CL],eps_E=[eps_E],tol=[tol],time=[complete_time],acc=[acc])
    CSV.write(csv_path, result_row; append=true)
    return acc
end

best_K = 34
best_eps_CL = 0.952750047242729
best_eps_E = 0.012826498305280605
best_tol = 3.47168681892656e-8


_K = (34-20):(34+20)
_eps_E = LinRange(best_eps_E*10,best_eps_E/10,1000)
_eps_CL = LinRange(best_eps_CL*10,best_eps_CL/10,1000)
_tol = LinRange(best_tol*10^-2,best_tol*10^2,1000)

println(collect(_K))
println(collect(_eps_CL))
println(collect(_eps_E))
println(collect(_tol))

for K in _K
    train_and_eval_model(K,best_eps_CL,best_eps_E,best_tol)
end
for eps_CL in _eps_CL
    train_and_eval_model(best_K,eps_CL,best_eps_E,best_tol)
end
for eps_E in _eps_E
    train_and_eval_model(best_K,best_eps_CL,eps_E,best_tol)
end
for tol in _tol
    train_and_eval_model(best_K,best_eps_CL,best_eps_E,tol)
end



