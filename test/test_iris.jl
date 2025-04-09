using eSPA
using Random
using MLDatasets
using MLJ
using MLJBase
using DataFrames
Random.seed!(50)

X, Y = MLDatasets.Iris(as_df = false)[:]

label_map = Dict(val => i for (i, val) in enumerate(unique(Y)))
Y = [label_map[s] for s in Y]
Y = Y'[:, 1]

train_indices, test_indices = partition(1:size(X, 2), 0.8, shuffle = true)
X_train = X[:, train_indices]
X_test = X[:, test_indices]
y_train = Y[train_indices]
y_test = Y[test_indices]

model = eSPAfuzzy(5, 1.0, 3.0, 0.0005)
eSPA.fit!(model, X_train, y_train)
y_pred = eSPA.predict(model, X_test)
println("Acc: ", accuracy(y_pred, y_test))
