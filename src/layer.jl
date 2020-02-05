using LinearAlgebra

include("circuit.jl")

export QNNL, forward, back_propagation
export Layer

abstract type Layer end

struct QNNL{N,T} <: Layer
	encoder::YaoBlocks.ChainBlock{N}
	transform::YaoBlocks.ChainBlock{N}
	params::Array{T}
	Hami::Array{YaoBlocks.ChainBlock{N}, 1}
	bias::Array{T,1}
end

function forward(ql::QNNL{N, T}, x::Array{T}) where {N, T}
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

function back_propagation(ql::QNNL{N,T}, x::Array{T}) where {N,T}
	m = size(ql.Hami, 1)
	s1 = size(x, 1)
	s2 = size(ql.params, 1)
	s3 = size(ql.bias, 1)

	if s3 > 0
		L_b = 0
	else
		L_b = 1
	end

	L_x = zeros(s1, m)
	L_w = zeros(s2, m)


	for i = 1:s1
		x[i] += π/2
		y_pos = forward(ql, x)
		x[i] -= π
		y_neg = forward(ql. x)
		x[i] += π/2

		L_x[i, :] = (y_pos - y_neg) / 2julia identity matrix
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
