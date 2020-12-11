function manifold_ODE_equilibrium(c::Vector{T}, ξ::Vector{T}, λ::T;
        f̂::Function, Df::Function, p, order::Int) where {T}

    @assert length(c) == length(ξ)
    n = length(c)

    a = [Sequence(Taylor(order), [cⱼ ; ξⱼ ; zeros(T, order-1)]) for (cⱼ,ξⱼ) ∈ zip(c,ξ)]

    Df_hessenberg = hessenberg!(Df(c, p))

    for α ∈ 2:order
        space_ = Taylor(α)
        a_ = [Sequence(space_, view(aⱼ, eachindex(space_))) for aⱼ ∈ a]
        αλ = α*λ
        v = f̂(a_, p, α)
        ldiv!(Df_hessenberg - αλ*I, v)
        @inbounds for j ∈ 1:n
            a[j][α] = -v[j]
        end
    end

    return a
end

function manifold_ODE_equilibrium(c::Vector{T}, ξ::Matrix{T}, λ::Vector{T};
        f̂::Function, Df::Function, p, order::NTuple{N,Int}) where {T,N}

    @assert length(c) == size(ξ, 1) && size(ξ, 2) == length(λ) == N
    n = length(c)

    space = TensorSpace(map(Taylor, orders))
    a = [Sequence(space, [cⱼ ; zeros(T, length(space)-1)]) for cⱼ ∈ c]
    for i ∈ 1:N
        idx = ntuple(l -> l == i ? 1 : 0, N)
        @inbounds for j ∈ 1:n
            a[j][idx] = ξ[j,i]
        end
    end

    Df_hessenberg = hessenberg!(Df(c, p))

    for α ∈ eachindex(space)
        if sum(α) ≥ 2
            space_ = TensorSpace(map(Taylor, α))
            a_ = [Sequence(space_, view(aⱼ, eachindex(space_))) for aⱼ ∈ a]
            αλ = mapreduce(*, +, α, λ)
            v = f̂(a_, p, α)
            ldiv!(Df_hessenberg - αλ*I, v)
            @inbounds for j ∈ 1:n
                a[j][α] = -v[j]
            end
        end
    end

    return a
end

##

function manifold_DDE_equilibrium(c::T, ξ::T, λ::T;
        f̂::Function, D₁f::Function, D₂f::Function, τ::Real, p, order::Int) where {T}

    a = Sequence(Taylor(order), [c ; ξ ; zeros(T, order-1)])
    b = Sequence(Taylor(order), [c ; ξ*exp(-λ*τ) ; zeros(T, order-1)])

    D₁f_ = D₁f(c, c, p)
    D₂f_ = D₂f(c, c, p)

    @inbounds for α ∈ 2:order
        space_ = Taylor(α)
        a_ = Sequence(space_, view(a, eachindex(space_)))
        b_ = Sequence(space_, view(b, eachindex(space_)))
        αλ = α*λ
        a[α] = (αλ - D₁f_ - D₂f_*exp(-αλ*τ))\f̂(a_, b_, p, α)
        b[α] = a[α]*exp(-αλ*τ)
    end

    return a, b
end

function manifold_DDE_equilibrium(c::T, ξ::Vector{T}, λ::Vector{T};
        f̂::Function, D₁f::Function, D₂f::Function, τ::Real, p, order::NTuple{N,Int}) where {T,N}

    @assert length(ξ) == length(λ) == N

    space = TensorSpace(map(Taylor, orders))
    a = Sequence(space, [c ; zeros(T, length(space)-1)])
    b = Sequence(space, [c ; zeros(T, length(space)-1)])
    @inbounds for i ∈ 1:N
        a[ntuple(l -> l == i ? 1 : 0, N)] = ξ[i]
        b[ntuple(l -> l == i ? 1 : 0, N)] = ξ[i]*exp(-λ[i]*τ)
    end

    D₁f_ = D₁f(c, c, p)
    D₂f_ = D₂f(c, c, p)

    @inbounds for α ∈ eachindex(space)
        if sum(α) ≥ 2
            space_ = TensorSpace(map(Taylor, α))
            a_ = Sequence(space_, view(a, eachindex(space_)))
            b_ = Sequence(space_, view(b, eachindex(space_)))
            αλ = mapreduce(*, +, α, λ)
            a[α] = (αλ - D₁f_ - D₂f_*exp(-αλ*τ))\f̂(a_, b_, p, α)
            b[α] = a[α]*exp(-αλ*τ)
        end
    end

    return a, b
end

function manifold_DDE_equilibrium(c::Vector{T}, ξ::Vector{T}, λ::T;
        f̂::Function, D₁f::Function, D₂f::Function, τ::Real, p, order::Int) where {T}

    @assert length(c) == length(ξ)
    n = length(c)

    a = [Sequence(Taylor(order), [cⱼ ; ξⱼ ; zeros(T, order-1)]) for (cⱼ,ξⱼ) ∈ zip(c,ξ)]
    b = [Sequence(Taylor(order), [cⱼ ; ξⱼ*exp(-λ*τ) ; zeros(T, order-1)]) for (cⱼ,ξⱼ) ∈ zip(c,ξ)]

    D₁f_ = D₁f(c, c, p)
    D₂f_ = D₂f(c, c, p)

    for α ∈ 2:order
        space_ = Taylor(α)
        a_ = [Sequence(space_, view(aⱼ, eachindex(space_))) for aⱼ ∈ a]
        b_ = [Sequence(space_, view(bⱼ, eachindex(space_))) for bⱼ ∈ b]
        αλ = α*λ
        v = f̂(a_, b_, p, α)
        ldiv!(lu!(D₁f_ + D₂f_*exp(-αλ*τ) - αλ*I), v)
        @inbounds for j ∈ 1:n
            a[j][α] = -v[j]
            b[j][α] = -v[j]*exp(-αλ*τ)
        end
    end

    return a, b
end

function manifold_DDE_equilibrium(c::Vector{T}, ξ::Matrix{T}, λ::Vector{T};
        f̂::Function, D₁f::Function, D₂f::Function, τ::Real, p, order::NTuple{N,Int}) where {T,N}

    @assert length(c) == size(ξ, 1) && size(ξ, 2) == length(λ) == N
    n = length(c)

    space = TensorSpace(map(Taylor, orders))
    a = [Sequence(space, [cⱼ ; zeros(T, length(space)-1)]) for cⱼ ∈ c]
    b = [Sequence(space, [cⱼ ; zeros(T, length(space)-1)]) for cⱼ ∈ c]
    for i ∈ 1:N
        idx = ntuple(l -> l == i ? 1 : 0, N)
        eλᵢτ = exp(-λ[i]*τ)
        @inbounds for j ∈ 1:n
            a[j][idx] = ξ[j,i]
            b[j][idx] = ξ[j,i]*eλᵢτ
        end
    end

    D₁f_ = D₁f(c, c, p)
    D₂f_ = D₂f(c, c, p)

    for α ∈ eachindex(space)
        if sum(α) ≥ 2
            space_ = TensorSpace(map(Taylor, α))
            a_ = [Sequence(space_, view(aⱼ, eachindex(space_))) for aⱼ ∈ a]
            b_ = [Sequence(space_, view(bⱼ, eachindex(space_))) for bⱼ ∈ b]
            αλ = mapreduce(*, +, α, λ)
            v = f̂(a_, b_, p, α)
            ldiv!(lu!(D₁f_ + D₂f_*exp(-αλ*τ) - αλ*I), v)
            @inbounds for j ∈ 1:n
                a[j][α] = -v[j]
                b[j][α] = -v[j]*exp(-αλ*τ)
            end
        end
    end

    return a, b
end
