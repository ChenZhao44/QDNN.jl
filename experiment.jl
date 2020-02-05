using Plots
using MLDatasets
using Images

# load full training set
rawdata_x, rawdata_y = MNIST.traindata();
# load full test set
# test_x,  test_y  = MNIST.testdata();

data_x = Array{Float64,1}[]
data_y = Int[]
data_ind = Int[]
for i = 1:60000
    if rawdata_y[i] < 2
        push!(data_x, Array{Float64,1}(imresize(rawdata_x[:,:,i], 8, 8)[:]))
        push!(data_y, rawdata_y[i])
        push!(data_ind, i)
    end
end

nData = size(data_y,1)

using Flux

include("layer.jl")

W1 = (rand(160) .- 0.5) * 2π
n1 = 8
pairs1 = [i => i+1 for i = 1:n1-1]
push!(pairs1, n1 => 1)

b2 = rand(24) .- 0.5
W2 = (rand(96) .- 0.5) * 2π
n2 = 6
pairs2 = [i => i+1 for i = 1:n2-1]
push!(pairs2, n2 => 1)

b3 = rand(12) .- 0.5
W3 = (rand(28) .- 0.5) * 2π
n3 = 4
pairs3 = [i => i+1 for i = 1:n3-1]
push!(pairs3, n3 => 1)

cir_1 = build_circuit(n1, 8, 20, pairs1);
cir_2 = build_circuit(n2, 4, 16, pairs2);
cir_3 = build_circuit(n3, 3, 7, pairs3);

H1_X = [put(n1, i=>X) for i = 1:n1]
H1_Y = [put(n1, i=>Y) for i = 1:n1]
H1_Z = [put(n1, i=>Z) for i = 1:n1]
Hami_1 = [H1_X; H1_Y; H1_Z]

H2_Y = [put(n2, i=>Y) for i = 1:n2]
H2_Z = [put(n2, i=>Z) for i = 1:n2]
Hami_2 = [H2_Y; H2_Z];

H3 = [Array{ComplexF64,2}(zeros(2,2)) for i = 1:2]
for i = 1:2
    H3[i][i,i] = 1
end

Hami_3 = [concentrate(4, matblock(H3[i]), (1)) for i = 1:2];

using FileIO

# save("data/Round_0.jld", "W1", W1, "b2", b2, "W2", W2, "b3", b3, "W3", W3)
W1, b2, W2, b3, W3 = load("data/Round_0.jld", "W1", "b2", "W2", "b3", "W3");

function loss(x,y)
    h1 = forward(cir_1, x, W1, Hami_1)
    h2 = forward_2(cir_2, h1, b2, W2, Hami_2)
    y_p = forward_2(cir_3, h2, b3, W3, Hami_3)
    l = sum((y-y_p).^2)
    m_y, i_y = findmax(y)
    m_y_p, i_y_p = findmax(y_p)
    corr = 0
    if i_y == i_y_p
        corr = 1
    end
    return l, corr
end

function loss()
    l = 0
    corr = 0
    for i = 1:nData
        x = data_x[i]
        y = zeros(2)
        y[data_y[i]+1] = 1
        l_x, corr_x = loss(x, y)
        l += l_x
        corr += corr_x
#         print([i, l])
    end
    return l/nData, corr/nData
end

# @time l, corr = loss()
# save("data/Round_0.jld", "W1", W1, "W2", W2, "W3", W3, "l", l, "corr", corr)

function get_gradient(x,y)
    h1 = forward(cir_1, x, W1, Hami_1)
    h2 = forward_2(cir_2, h1, b2, W2, Hami_2)
    y_p = forward_2(cir_3, h2, b3, W3, Hami_3)

    L_y = 2*(y_p - y)

    y_h2, y_b3, y_W3 = back_propagation_2(cir_3, h2, b3, W3, Hami_3)

    L_h2 = y_h2 * L_y
    L_b3 = y_b3 * L_y
    L_W3 = y_W3 * L_y

    h2_h1, h2_b2, h2_W2 = back_propagation_2(cir_2, h1, b2, W2, Hami_2)

    L_h1 = h2_h1 * L_h2
    L_b2 = h2_b2 * L_h2
    L_W2 = h2_W2 * L_h2

    h1_x, h1_W1 = back_propagation(cir_1, x, W1, Hami_1)
    L_W1 = h1_W1 * L_h1

    return L_W1, L_b2, L_W2, L_b3, L_W3
end

function get_gradient()
    grad_W1 = zeros(size(W1))
    grad_b2 = zeros(size(b2))
    grad_W2 = zeros(size(W2))
    grad_b3 = zeros(size(b3))
    grad_W3 = zeros(size(W3))
    for j = 1:nbatch
        i = rand(1:nData)
        x = data_x[i]
        y = zeros(2)
        y[data_y[i]+1] = 1
        L_W1, L_b2, L_W2, L_b3, L_W3 = get_gradient(x, y)
        grad_W1 += L_W1
        grad_b2 += L_b2
        grad_W2 += L_W2
        grad_b3 += L_b3
        grad_W3 += L_W3
    end
    grad_W1 = grad_W1 ./ nbatch
    grad_b2 = grad_b2 ./ nbatch
    grad_W2 = grad_W2 ./ nbatch
    grad_b3 = grad_b3 ./ nbatch
    grad_W3 = grad_W3 ./ nbatch
    return grad_W1, grad_b2, grad_W2, grad_b3, grad_W3
end

function get_gradient_test()
    grad_W1 = zeros(size(W1))
    grad_b2 = zeros(size(b2))
    grad_W2 = zeros(size(W2))
    grad_b3 = zeros(size(b3))
    grad_W3 = zeros(size(W3))
    for j = 1:nbatch
        i = j
        x = data_x[i]
        y = zeros(2)
        y[data_y[i]+1] = 1
        L_W1, L_b2, L_W2, L_b3, L_W3 = get_gradient(x, y)
        grad_W1 += L_W1
        grad_b2 += L_b2
        grad_W2 += L_W2
        grad_b3 += L_b3
        grad_W3 += L_W3
    end
    grad_W1 = grad_W1 ./ nbatch
    grad_b2 = grad_b2 ./ nbatch
    grad_W2 = grad_W2 ./ nbatch
    grad_b3 = grad_b3 ./ nbatch
    grad_W3 = grad_W3 ./ nbatch
    return grad_W1, grad_b2, grad_W2, grad_b3, grad_W3
end

function train_test(step, history_L, history_Corr)
    round = size(history_L, 1) - 1
    for k in (round+1):(round+step)
        println("Round: ", k)

        print("computing gradient!\n")
        grad_W1, grad_b2, grad_W2, grad_b3, grad_W3 = get_gradient_test()
        println("max of grad_W1 = ", findmax(abs.(grad_W1)))
        println("max of grad_b2 = ", findmax(abs.(grad_b2)))
        println("max of grad_W2 = ", findmax(abs.(grad_W2)))
        println("max of grad_b3 = ", findmax(abs.(grad_b3)))
        println("max of grad_W3 = ", findmax(abs.(grad_W3)))

        Optimise.update!(opt_W1, W1, grad_W1)
        Optimise.update!(opt_b2, b2, grad_b2)
        Optimise.update!(opt_W2, W2, grad_W2)
        Optimise.update!(opt_b3, b3, grad_b3)
        Optimise.update!(opt_W3, W3, grad_W3)

        l, corr = loss()
        push!(history_L, l)
        println("loss = ", l)
        push!(history_Corr, corr)
        println("correct rate = ", corr)

#         save("data/Round_$(k).jld", "W1", W1, "W2", W2, "W3", W3, "l", l, "corr", corr)
#         save("data/history_L.jld", "history_L", history_L)
#         save("data/history_Corr.jld", "history_Corr", history_Corr)
    end
end

using Plots

using JLD
using Flux.Optimise

function train(step, history_L, history_Corr)
    round = size(history_L, 1) - 1
    for k in (round+1):(round+step)
        println("Round: ", k)

        print("computing gradient!\n")
        grad_W1, grad_b2, grad_W2, grad_b3, grad_W3 = get_gradient()
        println("max of grad_W1 = ", findmax(grad_W1))
        println("max of grad_b2 = ", findmax(grad_b2))
        println("max of grad_W2 = ", findmax(grad_W2))
        println("max of grad_b3 = ", findmax(grad_b3))
        println("max of grad_W3 = ", findmax(grad_W3))

        Optimise.update!(opt_W1, W1, grad_W1)
        Optimise.update!(opt_b2, b2, grad_b2)
        Optimise.update!(opt_W2, W2, grad_W2)
        Optimise.update!(opt_b3, b3, grad_b3)
        Optimise.update!(opt_W3, W3, grad_W3)

        l, corr = loss()
        push!(history_L, l)
        println("loss = ", l)
        push!(history_Corr, corr)
        println("correct rate = ", corr)

        save("data/Round_$(k).jld", "W1", W1, "b2", b2, "W2", W2, "b3", b3, "W3", W3, "l", l, "corr", corr)
        save("data/history_L.jld", "history_L", history_L)
        save("data/history_Corr.jld", "history_Corr", history_Corr)
    end
end

η = 0.01
opt_W1 = ADAM(η)
opt_b2 = ADAM(η)
opt_W2 = ADAM(η)
opt_b3 = ADAM(η)
opt_W3 = ADAM(η)

W1, b2, W2, b3, W3 = load("data/Round_0.jld", "W1", "b2", "W2", "b3", "W3");

nbatch = 240

@time l, corr = loss()
history_L = [l]
history_Corr = [corr]
train(200, history_L, history_Corr)

η = 0.001
opt_W1 = ADAM(η)
opt_b2 = ADAM(η)
opt_W2 = ADAM(η)
opt_b3 = ADAM(η)
opt_W3 = ADAM(η)

train(200, history_L, history_Corr)

plot(0:400,[history_L history_Corr])
