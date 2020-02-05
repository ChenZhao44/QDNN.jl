using QDNN
using Yao
using Test

@testset "QDNN.jl" begin
    # Write your own tests here.
end

@testset "circuit.jl" begin
    ql = encoder_circuit(2,2)
end

@testset "layer.jl" begin
    cir_e = encoder_circuit(3, 5)
    cir_t = transform_circuit(3, 8)
    w = parameters(cir_t)
    H1_Y = [put(3, i=>Y) for i = 1:3]
    b = zeros(3)

    ql = QNNL{3, Float64}(cir_e, cir_t, w, H1_Y, b)
    x = rand(15)
    forward(ql, x)
    back_propagation(ql, x)
end

@testset "model.jl" begin

end
