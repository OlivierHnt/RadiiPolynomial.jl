function ivp_ODE(u₀::T; f::Function, p, order::Int) where {T}
    n = length(u₀)

    u = Sequence(Taylor(order), [u₀ ; zeros(T, order)])

    @inbounds for α ∈ 1:order
        space_ = Taylor(α-1)
        u_ = Sequence(space_, view(u, eachindex(space_)))
        v = f(u_, p, α-1)
        u[α] = f(u_, p, α-1) / α
    end

    return u
end

function ivp_ODE(u₀::Vector{T}; f::Function, p, order::Int) where {T}
    n = length(u₀)

    u = [Sequence(Taylor(order), [u₀⁽ʲ⁾ ; zeros(T, order)]) for u₀⁽ʲ⁾ ∈ u₀]

    @inbounds for α ∈ 1:order
        space_ = Taylor(α-1)
        u_ = [Sequence(space_, view(u⁽ʲ⁾, eachindex(space_))) for u⁽ʲ⁾ ∈ u]
        v = f(u_, p, α-1)
        ldiv!(α, v)
        @inbounds for j ∈ 1:n
            u[j][α] = v[j]
        end
    end

    return u
end
