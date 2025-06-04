using eSPA
using Random
using MLDatasets
using MLJ
using MLJBase
using DataFrames
using Plots
Random.seed!(50)

include("synthdata2.jl")

d= 4
n = 13 * d + 2
n = 100

X, Y = make_synth_data2(n, d)

train_indices, test_indices = partition(1:size(X, 2), 0.8, shuffle = true)
X_train = X[:, train_indices]
X_test = X[:, test_indices]
y_train = Y[train_indices]
y_test = Y[test_indices]


model = eSPAhybrid(21, 0.01, 0.1, 0.0001, 100 ,3)
start_time, start_optimization, end_time, opt_times = eSPA.fit!(model, X_train, y_train)
y_pred = eSPA.predict(model, X_test)
println("Acc: ", accuracy(y_pred, y_test))
println((end_time-start_time)/1e9)
println(opt_times)
