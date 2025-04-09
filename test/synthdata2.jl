using Random
using DataFrames
using LinearAlgebra

function make_synth_data2(n,d)
    return vcat(first_two_genes(n,[-5,-5],[5,5],1.),other_genes(n,d-2)), make_labels(n)
end

function first_two_genes(n,center_1,center_2,random_noise)
    cs = Int(n/4)
    d = norm(center_1 .- center_2)

    circle1 = (randn(2,cs) .* random_noise) .+ center_1
    circle2 = (randn(2,cs) .* random_noise) .+ center_2

    T = randn(cs) .* (2 * pi)
    R = (randn(cs) .* random_noise) .+ d/2
    ring1 = zeros(2,cs)
    ring1[:,:] .= center_1
    for i in 1:(cs)
        ring1[:,i] += [R[i]*cos(T[i]),R[i]*sin(T[i])]
    end

    T = randn(cs) .* (2 * pi)
    R = (randn(cs) .* random_noise) .+ d/2
    ring2 = zeros(2,cs)
    ring2[:,:] .= center_2
    for i in 1:(cs)
        ring2[:,i] += [R[i]*cos(T[i]),R[i]*sin(T[i])]
    end


    return hcat(circle1,circle2,ring1,ring2)
end

function make_labels(n)
    cs = Int(n/4)
    pos = ones(Int,2*cs)
    neg = ones(Int,2*cs) .+ 1
    return vcat(pos,neg)
end

function other_genes(n,d)
    return randn(d,n) .* (rand(d) .* 5)
end