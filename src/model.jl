using Flux.Optimise

export forward, back_propagation
export QDNNModel

struct QDNNModel
    layers::Array{Layer, 1}
    loss::Function
end

function forward(model::QDNNModel, x::Array{T}) where {T, L<:Layer}
    y = x
    Z = [x]
    for layer in model.layers
        y = QDNN.forward(layer, y)
        push!(Z, y)
    end
    return Z
end

function back_propagation(model::QDNNModel, x::Array{T}, L_y::Array{T}) where {T, L<:Layer}
    Z = forward(model, x)
    L_x = L_y
    nl = size(model.layers, 1)
    grad = Tuple{Array{T,1},Array{T,1}}[]

    for l = nl:-1:1
        layer = model.layers[l]
        L_xx, L_ww, L_bb = QDNN.back_propagation(layer, Z[l])
        L_w = L_ww * L_x
        if L_bb != 0
            L_b = L_bb * L_x
        else
            L_b = T[]
        end
        pushfirst!(grad, (L_w, L_b))

        L_x = L_xx * L_x
    end

    return grad
end
