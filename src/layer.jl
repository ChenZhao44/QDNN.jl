include("circuit.jl")

function forward(cir, x, w, Hami)
	n = nqubits(cir)
	dispatch!(cir, [x; w])
	m = size(Hami, 1)
	y = zeros(m)
	psi = zero_state(n)
	psi |> cir

	for i = 1:m
		y[i] = real(expect(Hami[i], psi))
	end

	return y
end

function forward_2(cir, x, b, w, Hami)
	n = nqubits(cir)
	dispatch!(cir, [x+b; w])
	m = size(Hami, 1)
	y = zeros(m)
	psi = zero_state(n)
	psi |> cir

	for i = 1:m
		y[i] = real(expect(Hami[i], psi))
	end

	return y
end

function forward(cir, x, w)
	n = nqubits(cir)
	dispatch!(cir, [x; w])
	m = 2^n
	y = zeros(m)
	psi = zero_state(n)
	psi |> cir

	y = probs(psi)
	return y
end

function forward_2(cir, x, b, w)
	n = nqubits(cir)
	dispatch!(cir, [x+b; w])
	m = 2^n
	y = zeros(m)
	psi = zero_state(n)
	psi |> cir

	y = probs(psi)
	return y
end

function back_propagation(cir, x, w, Hami)
	m = size(Hami, 1)
	s1 = size(x, 1)
	s2 = size(w, 1)

	L_x = zeros(s1, m)
	L_w = zeros(s2, m)

	for i = 1:s1
		x[i] += π/2
		y_pos = forward(cir, x, w, Hami)
		x[i] -= π
		y_neg = forward(cir, x, w, Hami)
		x[i] += π/2

		L_x[i, :] = y_pos - y_neg
	end

	for i = 1:s2
		w[i] += π/2
		y_pos = forward(cir, x, w, Hami)
		w[i] -= π
		y_neg = forward(cir, x, w, Hami)
		w[i] += π/2

		L_w[i, :] = y_pos - y_neg
	end

	return L_x, L_w
end

function back_propagation_2(cir, x, b, w, Hami)
	m = size(Hami, 1)
	s1 = size(x, 1)
	s2 = size(w, 1)

	L_x = zeros(s1, m)
	L_w = zeros(s2, m)

	for i = 1:s1
		x[i] += π/2
		y_pos = forward_2(cir, x, b, w, Hami)
		x[i] -= π
		y_neg = forward_2(cir, x, b, w, Hami)
		x[i] += π/2

		L_x[i, :] = y_pos - y_neg
	end

	L_b = copy(L_x)

	for i = 1:s2
		w[i] += π/2
		y_pos = forward_2(cir, x, b, w, Hami)
		w[i] -= π
		y_neg = forward_2(cir, x, b, w, Hami)
		w[i] += π/2

		L_w[i, :] = y_pos - y_neg
	end

	return L_x, L_b, L_w
end

function back_propagation(cir, x, w)
	n = nqubits(cir)
	m = 2^n
	s1 = size(x, 1)
	s2 = size(w, 1)

	L_x = zeros(s1, m)
	L_w = zeros(s2, m)

	for i = 1:s1
		x[i] += π/2
		y_pos = forward(cir, x, w)
		x[i] -= π
		y_neg = forward(cir, x, w)
		x[i] += π/2

		L_x[i, :] = y_pos - y_neg
	end

	for i = 1:s2
		w[i] += π/2
		y_pos = forward(cir, x, w)
		w[i] -= π
		y_neg = forward(cir, x, w)
		w[i] += π/2

		L_w[i, :] = y_pos - y_neg
	end

	return L_x, L_w
end

function back_propagation_2(cir, x, b, w)
	n = nqubits(cir)
	m = 2^n
	s1 = size(x, 1)
	s2 = size(w, 1)

	L_x = zeros(s1, m)
	L_w = zeros(s2, m)

	for i = 1:s1
		x[i] += π/2
		y_pos = forward_2(cir, x, b, w)
		x[i] -= π
		y_neg = forward_2(cir, x, b, w)
		x[i] += π/2

		L_x[i, :] = y_pos - y_neg
	end

	L_b = copy(L_x)

	for i = 1:s2
		w[i] += π/2
		y_pos = forward_2(cir, x, b, w)
		w[i] -= π
		y_neg = forward_2(cir, x, b, w)
		w[i] += π/2

		L_w[i, :] = y_pos - y_neg
	end

	return L_x, L_b, L_w
end