function ivp_ODE(u₀::T; f, order::Int) where {T}
    space = Taylor(order)
    u = Sequence(space, zeros(T, dimension(space)))
    u[0] = u₀
    for α ∈ 1:order
        space_view = Taylor(α-1)
        u_view = Sequence(space_view, view(u, allindices(space_view)))
        u[α] = f(u_view, α-1) / α
    end
    return u
end

function ivp_ODE(u₀::AbstractVector{T}; f, order::Int) where {T}
    n = length(u₀)
    space = Taylor(order)^n
    u = Sequence(space, zeros(T, dimension(space)))
    @inbounds for j ∈ 1:n
        component(u, j)[0] = u₀[j]
    end
    order == 0 && return u
    coeffs_view = Vector{T}(undef, n)
    for α ∈ 1:order
        taylor_space = Taylor(α-1)
        space_view = taylor_space^n
        u_view = Sequence(space_view, resize!(coeffs_view, dimension(space_view)))
        indices = allindices(taylor_space)
        @inbounds for j ∈ 1:n
            component(u_view, j).coefficients .= view(component(u, j), indices)
        end
        v = rdiv!(f(u_view, α-1), α)
        for j ∈ 1:n
            component(u, j)[α] = v[j]
        end
    end
    return u
end
