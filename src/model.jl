using Flux.Optimise

export QDNN

struct QDNN
    layers::Array{Layer, 1}
end

function forward(model::QDNN, x::Array{T}) where {T}
    y = x
    Z = Array{Array{T}, 1}[]
    Z = [x]
    for layer in model
        y = forward(layer, y)
        push!(Z, y)
    end
    return Z
end

function back_propagation(model::QDNN, x::Array{T}, L_y::Array{T}) where {T}
    Z = forward(model, x)
    L_x = L_y
    nl = size(model, 1)
    grad = []

    for l = nl:-1:1
        layer = model[l]
        L_xx, L_ww, L_bb = back_propagation(layer, Z[l])
        L_w = L_ww * L_x
        if L_bb
            L_b = L_bb * L_x
        else
            L_b = Array{T, 1}[]
        end
        pushfirst!(grad, (L_w, L_b))

        L_x = L_xx
    end
end
