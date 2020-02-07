using LinearAlgebra
using Yao

include("circuit.jl")

export QNNL, forward, back_propagation
export Layer

abstract type Layer{T} end

struct QNNL{T} <: Layer{T}
	encoder::YaoBlocks.ChainBlock
	transform::YaoBlocks.ChainBlock
	params::Array{T}
	Hami::Array{<:YaoBlocks.AbstractBlock, 1}
	bias::Array{T,1}
end

function QNNL{T}(n::Integer, l1::Integer, l2::Integer, w::Array{T}, b::Array{T,1}, Hami::Array{<:YaoBlocks.AbstractBlock, 1}) where {T}
	cir_e = QDNN.encoder_circuit(n, l1)
	cir_t = QDNN.transform_circuit(n, l2)
	QNNL{T}(cir_e, cir_t, w, Hami, b)
end

function QNNL{T}(n::Integer, l1::Integer, l2::Integer, Hami::Array{<:YaoBlocks.AbstractBlock, 1}; no_bias = false) where {T}
	cir_e = QDNN.encoder_circuit(n, l1)
	cir_t = QDNN.transform_circuit(n, l2)
	w = (rand(T, size(parameters(cir_t))) .- 0.5) * 2π
	if no_bias
		b = T[]
	else
		b = rand(T, size(Hami)) .- 0.5
	end
	QNNL{T}(cir_e, cir_t, w, Hami, b)
end

function QNNL{T}(n::Integer, l1::Integer, l2::Integer, w::Array{T}, b::Array{T,1}, Hs::String) where {T}
	Hami = YaoBlocks.ChainBlock[]
	for c in Hs
		Hami = [Hami; genH(n, c)]
	end
	cir_e = QDNN.encoder_circuit(n, l1)
	cir_t = QDNN.transform_circuit(n, l2)
	QNNL{T}(cir_e, cir_t, w, Hami, b)
end


function QNNL{T}(n::Integer, l1::Integer, l2::Integer, Hs::String; no_bias = false) where T
	Hami = YaoBlocks.ChainBlock[]
	for c in Hs
		Hami = [Hami; genH(n, c)]
	end
	cir_e = QDNN.encoder_circuit(n, l1)
	cir_t = QDNN.transform_circuit(n, l2)
	w = (rand(T, size(parameters(cir_t))) .- 0.5) * 2π
	if no_bias
		b = T[]
	else
		b = rand(T, size(Hami)) .- 0.5
	end
	QNNL{Float64}(cir_e, cir_t, w, Hami, b)
end

function forward(ql::QNNL{T}, x::Array{T}) where {N, T}
	cir = chain(ql.encoder, ql.transform)
	n = nqubits(cir)
	dispatch!(cir, [x; ql.params])

	m = size(ql.Hami, 1)
	psi = zero_state(n)
	psi |> cir
	y = [real(expect((ql.Hami[i]), psi)) for i = 1:m]
	if size(ql.bias, 1) > 0
		y += ql.bias
	end
	return y
end

function back_propagation(ql::QNNL{T}, x::Array{T}) where {N,T}
	m = size(ql.Hami, 1)
	s1 = size(x, 1)
	s2 = size(ql.params, 1)
	s3 = size(ql.bias, 1)

	if s3 > 0
		L_b = 1
	else
		L_b = 0
	end

	L_x = zeros(s1, m)
	L_w = zeros(s2, m)


	for i = 1:s1
		x[i] += π/2
		y_pos = forward(ql, x)
		x[i] -= π
		y_neg = forward(ql, x)
		x[i] += π/2

		L_x[i, :] = (y_pos - y_neg) / 2
	end

	for i = 1:s2
		ql.params[i] += π/2
		y_pos = forward(ql, x)
		ql.params[i] -= π
		y_neg = forward(ql, x)
		ql.params[i] += π/2

		L_w[i, :] = (y_pos - y_neg) / 2
	end

	return L_x, L_w, L_b
end
