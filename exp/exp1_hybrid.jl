using eSPA
using Random
using Hyperopt
using MLJ
using MLJBase
using Dates
using CSV
using JSON

Random.seed!(50)
include("synthdata2.jl")

D = [4,10,50]
N = [100,1000]
iter = 2

for d in D, n in N

    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    experiment_dir = "exp1_discrete_$(d)_$(n)" * timestamp
    mkpath(experiment_dir) 

    X, Y = make_synth_data2(n, d)

    train_indices, test_indices = partition(1:size(X, 2), 0.8, shuffle = true)
    X_train = X[:, train_indices]
    X_test = X[:, test_indices]
    y_train = Y[train_indices]
    y_test = Y[test_indices]

    global iteration_counter = 0

    function train_and_eval_model(K,eps_CL,eps_E,tol,num_fuzzy_steps)

        global iteration_counter += 1

        model = eSPAhybrid(K,eps_CL,eps_E,tol,num_fuzzy_steps)
        start_time, start_optimization, end_time, opt_times = eSPA.fit!(model, X_train, y_train)
        y_pred = eSPA.predict(model, X_test)
        acc = accuracy(y_pred, y_test)
        #Track data
        params = Dict(
            "complete_time" => (end_time-start_time)/1e9,
            "opt_time" => (end_time-start_optimization)/1e9,
            "accuracy" => acc,
            "hps" => Dict("K" => K, "eps_CL" => eps_CL,"eps_E" => eps_E, "tol" => tol,"num_fuzzy_steps" => num_fuzzy_steps)
            )
        json_file = joinpath(experiment_dir, "params_$(iteration_counter).json")
        open(json_file, "w") do io
            JSON.print(io, params)
        end

        csv_file = joinpath(experiment_dir, "opt_times_$(iteration_counter).csv")
        CSV.write(csv_file, opt_times)

        return acc
    end

    ho = @hyperopt for i = iter,
                K = 5:(minimum([200,Int(0.8*n)])),
                eps_E = LinRange(1,500,1000),
                eps_CL = LinRange(1,500,1000),
                tol = exp10.(-LinRange(1,10,1000)),
                num_fuzzy_steps = 5:15
        cost = train_and_eval_model(K,eps_CL,eps_E,tol)
    end

    hp_tune_results = Dict(
        "max_acc" => ho.maximum,
        "best_params" => ho.maximizer
    )

    json_file = joinpath(experiment_dir, "hp_tune.json")
        open(json_file, "w") do io
            JSON.print(io, hp_tune_results)
        end

end
