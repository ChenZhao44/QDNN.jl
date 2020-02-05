export encoder_circuit, transform_circuit

layerX(nbit::Integer) = chain(nbit, put(i=>Rx(0)) for i = 1:nbit)
layerZ(nbit::Integer) = chain(nbit, put(i=>Rz(0)) for i = 1:nbit)
entangler(pairs::Array{Pair{T,T},1} where T) = chain(control(ctrl, target=>X) for (ctrl, target) in pairs)

function encoder_circuit(n::Integer, l::Integer, pairs::Array{Pair{T,T},1} where T)
	circuit = chain(n)

	if l >= 1
		push!(circuit, layerX(n))
	end
	if l >= 2
		push!(circuit, layerZ(n))
	end
	for i in 3:l
			if i%3 == 0
					push!(circuit, entangler(pairs))
			end
			if i%3 == 1
					push!(circuit, layerX(n))
			else
					push!(circuit, layerZ(n))
			end
	end
	push!(circuit, entangler(pairs))

	return circuit
end

function encoder_circuit(n::Integer, l::Integer)
	pairs = [i => i+1 for i = 1:(n-1)]
	push!(pairs, n=>1)
	return encoder_circuit(n, l, pairs)
end

function transform_circuit(n::Integer, l::Integer, pairs::Array{Pair{T,T},1} where T)
	circuit = chain(n)
	for i in 1:l
			if i%3 == 2
					push!(circuit, layerX(n))
			else
					push!(circuit, layerZ(n))
			end
			if i%3 == 0 && i != l
					push!(circuit, entangler(pairs))
			end
	end
	return circuit
end

function transform_circuit(n::Integer, l::Integer)
	pairs = [i => i+1 for i = 1:(n-1)]
	push!(pairs, n=>1)
	return transform_circuit(n, l, pairs)
end

function genH(n::Integer, c::Char)
	if c == 'X'
		return [chain(n, put(n, i=>X)) for i = 1:n]
	elseif c == 'Y'
		return [chain(n, put(n, i=>Y)) for i = 1:n]
	elseif c == 'Z'
		return [chain(n, put(n, i=>Z)) for i = 1:n]
	end
end
