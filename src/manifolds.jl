function manifold_ODE_equilibrium(c::AbstractVector{T}, ξ::AbstractVector{T}, λ::T;
        Df, f̂, order::Int) where {T}

    @assert length(c) == length(ξ)
    n = length(c)

    a = [Sequence(Taylor(order), [cⱼ ; ξⱼ ; zeros(T, order-1)]) for (cⱼ,ξⱼ) ∈ zip(c,ξ)]

    Df_hessenberg = hessenberg(Df)

    @inbounds for α ∈ 2:order
        space_ = Taylor(α)
        a_ = [Sequence(space_, view(aⱼ, allindices(space_))) for aⱼ ∈ a]
        αλ = α*λ
        v = f̂(a_, α)
        ldiv!(Df_hessenberg - αλ*I, v)
        for j ∈ 1:n
            a[j][α] = -v[j]
        end
    end

    return a
end

function manifold_ODE_equilibrium(c::AbstractVector{T}, ξ::AbstractMatrix{T}, λ::AbstractVector{T};
        Df, f̂, order::NTuple{N,Int}) where {T,N}

    @assert length(c) == size(ξ, 1) && size(ξ, 2) == length(λ) == N
    n = length(c)

    space = TensorSpace(map(Taylor, order))
    a = [Sequence(space, [cⱼ ; zeros(T, dimension(space)-1)]) for cⱼ ∈ c]
    for i ∈ 1:N
        idx = ntuple(l -> l == i ? 1 : 0, N)
        @inbounds for j ∈ 1:n
            a[j][idx] = ξ[j,i]
        end
    end

    Df_hessenberg = hessenberg(Df)

    @inbounds for α ∈ allindices(space)
        if sum(α) ≥ 2
            space_ = TensorSpace(map(Taylor, α))
            a_ = [Sequence(space_, view(aⱼ, allindices(space_))) for aⱼ ∈ a]
            αλ = mapreduce(*, +, α, λ)
            v = f̂(a_, α)
            ldiv!(Df_hessenberg - αλ*I, v)
            for j ∈ 1:n
                a[j][α] = -v[j]
            end
        end
    end

    return a
end

##

function manifold_DDE_equilibrium(c::T, ξ::T, λ::T;
        Ψ, f̂, order::Int) where {T}

    a = Sequence(Taylor(order), [c ; ξ ; zeros(T, order-1)])

    @inbounds for α ∈ 2:order
        space_ = Taylor(α)
        a_ = Sequence(space_, view(a, allindices(space_)))
        αλ = α*λ
        a[α] = -Ψ(αλ)\f̂(a_, αλ, α)
    end

    return a
end

function manifold_DDE_equilibrium(c::T, ξ::AbstractVector{T}, λ::AbstractVector{T};
        Ψ, f̂, order::NTuple{N,Int}) where {T,N}

    @assert length(ξ) == length(λ) == N

    space = TensorSpace(map(Taylor, order))
    a = Sequence(space, [c ; zeros(T, dimension(space)-1)])
    @inbounds for i ∈ 1:N
        a[ntuple(l -> l == i ? 1 : 0, N)] = ξ[i]
    end

    @inbounds for α ∈ allindices(space)
        if sum(α) ≥ 2
            space_ = TensorSpace(map(Taylor, α))
            a_ = Sequence(space_, view(a, allindices(space_)))
            αλ = mapreduce(*, +, α, λ)
            a[α] = -Ψ(αλ)\f̂(a_, αλ, α)
        end
    end

    return a
end

function manifold_DDE_equilibrium(c::AbstractVector{T}, ξ::AbstractVector{T}, λ::T;
        Ψ, f̂, order::Int) where {T}

    @assert length(c) == length(ξ)
    indices = eachindex(c)

    @inbounds a = [Sequence(Taylor(order), [c[j] ; ξ[j] ; zeros(T, order-1)]) for j ∈ indices]

    @inbounds for α ∈ 2:order
        space_ = Taylor(α)
        a_ = [Sequence(space_, view(a[j], allindices(space_))) for j ∈ indices]
        αλ = α*λ
        v = f̂(a_, αλ, α)
        ldiv!(lu!(Ψ(αλ)), v)
        for j ∈ indices
            a[j][α] = -v[j]
        end
    end

    return a
end

function manifold_DDE_equilibrium(c::AbstractVector{T}, ξ::AbstractMatrix{T}, λ::AbstractVector{T};
        Ψ, f̂, order::NTuple{N,Int}) where {T,N}

    @assert length(c) == size(ξ, 1) && size(ξ, 2) == length(λ) == N
    indices = eachindex(c)

    space = TensorSpace(map(Taylor, order))
    a = [Sequence(space, [c[j] ; zeros(T, dimension(space)-1)]) for j ∈ indices]
    for i ∈ 1:N
        idx = ntuple(l -> l == i ? 1 : 0, N)
        @inbounds for j ∈ indices
            a[j][idx] = ξ[j,i]
        end
    end

    @inbounds for α ∈ allindices(space)
        if sum(α) ≥ 2
            space_ = TensorSpace(map(Taylor, α))
            a_ = [Sequence(space_, view(a[j], allindices(space_))) for j ∈ indices]
            αλ = mapreduce(*, +, α, λ)
            v = f̂(a_, αλ, α)
            ldiv!(lu!(Ψ(αλ)), v)
            for j ∈ indices
                a[j][α] = -v[j]
            end
        end
    end

    return a
end
