layerX(nbit::Integer) = chain(nbit, put(i=>Rx(0)) for i = 1:nbit)
layerZ(nbit::Integer) = chain(nbit, put(i=>Rz(0)) for i = 1:nbit)
entangler(pairs::Array{Pair{T,T},1} where T) = chain(control(ctrl, target=>X) for (ctrl, target) in pairs)

function build_circuit(n::Integer, l1::Integer, l2::Integer, pairs::Array{Pair{T,T},1} where T)
		circuit = chain(n)

		push!(circuit, layerX(n))
		push!(circuit, layerZ(n))
		for i in 3:l1
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

		for i in 1:l2
				if i%3 == 0
						push!(circuit, entangler(pairs))
				end
				if i%3 == 1
						push!(circuit, layerX(n))
				else
						push!(circuit, layerZ(n))
				end
		end

		return circuit
end

function build_circuit_2(n, l1, pairs)
		circuit = chain(n)

		push!(circuit, layerX(n))
		push!(circuit, layerZ(n))
		for i in 3:l1
				if i%3 == 0
						push!(circuit, entangler(pairs))
				end
				if i%3 == 1
						push!(circuit, layerX(n))
				else
						push!(circuit, layerZ(n))
				end
		end

		return circuit
end

function build_circuit(n::Integer, l1::Integer, l2::Integer)
	pairs = [i => i+1 for i = 1:(n-1)]
	return build_circuit(n, l1, l2, pairs)
end
