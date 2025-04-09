using eSPA
using Random
using MLDatasets
using MLJ
using MLJBase
using DataFrames
using Plots
Random.seed!(50)

include("synthdata2.jl")

d=8
n = 13 * d + 2
n = 1000

X, Y = make_synth_data2(n, d)

train_indices, test_indices = partition(1:size(X, 2), 0.8, shuffle = true)
X_train = X[:, train_indices]
X_test = X[:, test_indices]
y_train = Y[train_indices]
y_test = Y[test_indices]

#model = eSPAdiscrete(35, 20., 500.0, 0.0001)
model = GOAL(12, 500.0, 7, 0.1)
clusters = eSPA.fit!(model, X_train, y_train)
y_pred = eSPA.predict(model, X_test)
println("Acc: ", accuracy(y_pred, y_test))

println(size(clusters))
scatter(X_test[1, :], X_test[2, :], zcolor = y_pred, label = "", alpha = 0.5)
#for cls in clusters
#    scatter!(cls,label="",marker = :xcross,markersize=8)
#end

scatter!(model.S[1, :], model.S[2, :], label = "", marker = :diamond, markersize = 4,
         color = :red)

function plot_it(i, x, y)
    cls = clusters[i]
    scatter(X_test[x, (y_test .== 1)], X_test[y, (y_test .== 1)],
            zcolor = y_pred[(y_test .== 1)], label = "", alpha = 0.5, marker = :circle)
    scatter!(X_test[x, (y_test .== 2)], X_test[y, (y_test .== 2)],
             zcolor = y_pred[(y_test .== 2)], label = "", alpha = 0.5, marker = :square)
    scatter!(cls[x, :], cls[y, :], label = "", marker = :diamond, markersize = 4,
             color = :red)
end
