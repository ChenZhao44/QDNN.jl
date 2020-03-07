using Flux

export forward, back_propagation, loss, get_gradient, para_update!
export QDNNModel

struct QDNNModel{T}
    layers::Array{L, 1} where {L<:Layer{T}}
    # loss::Function
end

function forward(model::QDNNModel{T}, x::Array{T}) where {T}
    y = x
    Z = [x]
    for layer in model.layers
        y = QDNN.forward(layer, y)
        push!(Z, y)
    end
    return Z
end

function back_propagation(model::QDNNModel{T}, x::Array{T}, L_y::Array{T}) where {T}
    Z = forward(model, x)
    L_x = L_y
    nl = size(model.layers, 1)
    grad = Vector{Vector{T}}[]

    for l = nl:-1:1
        layer = model.layers[l]
        L_xx, L_ww, L_bb = QDNN.back_propagation(layer, Z[l])
        L_w = L_ww * L_x
        if L_bb != 0
            L_b = L_bb * L_x
        else
            L_b = T[]
        end
        pushfirst!(grad, [L_w, L_b])

        L_x = L_xx * L_x
    end

    return grad
end

# function MSE(y1::Vector{T}, y2::Vector{T}) where {T}
#     n = size(y1, 1)
#     return sum((y1-y2).^2)/n, 2/n .* (y1-y2)
# end

function SSE(y1::Vector{T}, y2::Vector{T}) where {T}
    return sum((y1-y2).^2), 2 .* (y1-y2)
end

function loss(qm::QDNNModel{T}, x::Vector{T}, y::Vector{T}) where {T}
    fwd = forward(qm, x)
    y_p = fwd[end]
    l, L_y = SSE(y_p, y)
    return l
end

function loss(qm::QDNNModel{T}, data_x::Vector{Vector{T}}, data_y::Vector{Vector{T}}) where {T}
    nData = size(data_x, 1)
    l = zeros(nData)
    for i = 1:nData
        x = data_x[i]
        y = data_y[i]
        l_x = loss(qm, x, y)
        l[i] = l_x
    end
    return l
end

function get_gradient(qm::QDNNModel{T}, x::Vector{T}, y::Vector{T}) where {T}
    fwd = forward(qm, x)
    y_p = fwd[end]
    l, L_y = SSE(y_p, y)

    grad = back_propagation(qm, x, L_y)
    return grad
end

function get_gradient(qm::QDNNModel{T}, data_x::Vector{Vector{T}}, data_y::Vector{Vector{T}}, nbatch::Integer) where T
    grad = nothing
    nData = size(data_x, 1)
    for j = 1:nbatch
        i = rand(1:nData)
        x = data_x[i]
        y = data_y[i]
        gradi = get_gradient(qm, x, y)
        if grad == nothing
            grad = gradi
        else
            grad += gradi
        end
    end
    grad = grad ./ nbatch
    return grad
end

function para_update!(qm::QDNNModel{T}, grad::Vector{Vector{Vector{T}}}, opt) where {T}
    lys = qm.layers
    nl = size(lys, 1)
    for i = 1:nl
        grad_w = grad[i][1]
        grad_b = grad[i][2]
        Flux.Optimise.update!(opt, lys[i].params, grad_w)
        Flux.Optimise.update!(opt, lys[i].bias, grad_b)
    end
end
