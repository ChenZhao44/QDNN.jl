using QDNN
using Yao

n1 = 8
ql_1 = QNNL{Float64}(8, 8, 17, "XYZ")

n2 = 6
ql_2 = QNNL{Float64}(6, 4, 14, "YZ")

n3 = 4
H3 = [Array{ComplexF64,2}(zeros(2,2)) for i = 1:2]
for i = 1:2
    H3[i][i,i] = 1
end
Hami_3 = [subroutine(4, matblock(H3[i]), (1)) for i = 1:2];
ql_3 = QNNL{Float64}(4, 3, 8, Hami_3; no_bias = true)

qm = QDNNModel([ql_1, ql_2, ql_3])

x = rand(64)
ZZ = forward(qm, x)
bp = back_propagation(qm, x, [0, 1.0])

using MLDatasets
using Images

# load full training set
rawdata_x, rawdata_y = MNIST.traindata();
# load full test set
# test_x,  test_y  = MNIST.testdata();

data_x = Vector{Float64}[]
data_y = Vector{Float64}[]
data_ind = Int[]
for i = 1:60000
    if rawdata_y[i] < 2
        push!(data_x, Array{Float64,1}(imresize(rawdata_x[:,:,i], 8, 8)[:]))
        y = zeros(2)
        y[rawdata_y[i]+1] = 1
        push!(data_y, y)
        push!(data_ind, i)
    end
end

using Flux.Optimise
opt = ADAM(0.1)

function train(qm::QDNNModel, data_x, data_y, iter::Integer, nbatch::Integer, opt)
    l = loss(qm, data_x, data_y)
    save("data01/loss_0.jld", "l", l)
    save("data01/qm_0.jld", "qm", qm)
    println("At first, loss = $(sum(l) / length(l))")
    for i = 1:iter
        println("Iteration $(i):")
        println("Computing gradient...")
        grad = get_gradient(qm, data_x, data_y, nbatch)
        println("Updating...")
        para_update!(qm, grad, opt)
        l = loss(qm, data_x, data_y)
        save("data01/loss_$(i).jld", "l", l)
        save("data01/qm_$(i).jld", "qm", qm)
        println("loss = $(sum(l) / length(l))")
    end
end

using FileIO
train(qm, data_x, data_y, 200, 240, opt)
