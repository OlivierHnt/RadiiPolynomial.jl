##

function initialize_manifold_parameterization(c, ξ, order::Int)
    CoefType = promote_type(typeof(c), typeof(ξ))
    space = Taylor(order)
    P = Sequence(space, zeros(CoefType, dimension(space)))
    P[0] = c
    P[1] = ξ
    return P
end

function initialize_manifold_parameterization(c, ξ::AbstractVector, order::NTuple{N,Int}) where {N}
    CoefType = promote_type(typeof(c), eltype(ξ))
    space = TensorSpace(map(Taylor, order))
    P = Sequence(space, zeros(CoefType, dimension(space)))
    P.coefficients[1] = c
    for (i, ξᵢ) ∈ enumerate(ξ)
        P[ntuple(j -> ifelse(j == i, 1, 0) , Val(N))] = ξᵢ
    end
    return P
end

function initialize_manifold_parameterization(c::AbstractVector, ξ::AbstractVector, order::Int)
    CoefType = promote_type(eltype(c), eltype(ξ))
    n = length(c)
    space = Taylor(order)^n
    P = Sequence(space, zeros(CoefType, dimension(space)))
    for i ∈ 1:n
        Pᵢ = component(P, i)
        Pᵢ[0] = c[i]
        Pᵢ[1] = ξ[i]
    end
    return P
end

function initialize_manifold_parameterization(c::AbstractVector, ξ::AbstractMatrix, order::NTuple{N,Int}) where {N}
    CoefType = promote_type(eltype(c), eltype(ξ))
    n = length(c)
    space = TensorSpace(map(Taylor, order))^n
    P = Sequence(space, zeros(CoefType, dimension(space)))
    r = Vector{NTuple{N,Int}}(undef, N)
    @inbounds for j ∈ 1:N
        r[j] = ntuple(k -> ifelse(k == j, 1, 0) , Val(N))
    end
    for i ∈ 1:n
        Pᵢ = component(P, i)
        Pᵢ.coefficients[1] = c[i]
        for j ∈ 1:N
            Pᵢ[r[j]] = ξ[i,j]
        end
    end
    return P
end

##

function _view_manifold_parameterization!(v, P::Sequence{CartesianPowerSpace{Taylor}}, order::Int)
    taylor_space = Taylor(order)
    space_view = taylor_space^P.space.dim
    P_view = Sequence(space_view, resize!(v, dimension(space_view)))
    indices = allindices(taylor_space)
    @inbounds for j ∈ 1:P.space.dim
        component(P_view, j).coefficients .= view(component(P, j), indices)
    end
    return P_view
end

function _view_manifold_parameterization!(v, P::Sequence{CartesianPowerSpace{T}}, order::NTuple{N,Int}) where {N,T<:TensorSpace{NTuple{N,Taylor}}}
    taylor_space = TensorSpace(map(Taylor, order))
    space_view = taylor_space^P.space.dim
    P_view = Sequence(space_view, resize!(v, dimension(space_view)))
    indices = allindices(taylor_space)
    @inbounds for j ∈ 1:P.space.dim
        component(P_view, j).coefficients .= view(component(P, j), indices)
    end
    return P_view
end

##

function manifold_DDS_equilibrium(c::AbstractVector, ξ::AbstractVector, λ; Df, f̂, order::Int)
    length(c) == length(ξ) || return throw(DimensionMismatch)
    n = length(c)
    P = initialize_manifold_parameterization(c, ξ, order)
    order == 1 && return P
    Df_hessenberg = hessenberg(Df)
    coeffs_view = Vector{eltype(P)}(undef, dimension(Taylor(2)))
    for α ∈ 2:order
        P_view = _view_manifold_parameterization!(coeffs_view, P, α)
        λᵅ = λ^α
        v = ldiv!(Df_hessenberg - λᵅ*I, f̂(P_view, α))
        for j ∈ 1:n
            component(P, j)[α] = -v[j]
        end
    end
    return P
end

function manifold_DDS_equilibrium(c::AbstractVector, ξ::AbstractMatrix, λ::AbstractVector; Df, f̂, order::NTuple{N,Int}) where {N}
    length(c) == size(ξ, 1) && size(ξ, 2) == length(λ) == N || return throw(DimensionMismatch)
    n = length(c)
    P = initialize_manifold_parameterization(c, ξ, order)
    maximum(order) == 1 && return P
    Df_hessenberg = hessenberg(Df)
    coeffs_view = Vector{eltype(P)}(undef, dimension(TensorSpace(ntuple(i -> Taylor(2), Val(N)))))
    for α ∈ allindices(P.space.space)
        if sum(α) ≥ 2
            P_view = _view_manifold_parameterization!(coeffs_view, P, α)
            λᵅ = mapreduce(^, *, λ, α)
            v = ldiv!(Df_hessenberg - λᵅ*I, f̂(P_view, α))
            for j ∈ 1:n
                component(P, j)[α] = -v[j]
            end
        end
    end
    return P
end

##

function manifold_ODE_equilibrium(c::AbstractVector, ξ::AbstractVector, λ; Df, f̂, order::Int)
    length(c) == length(ξ) || return throw(DimensionMismatch)
    n = length(c)
    P = initialize_manifold_parameterization(c, ξ, order)
    order == 1 && return P
    Df_hessenberg = hessenberg(Df)
    coeffs_view = Vector{eltype(P)}(undef, dimension(Taylor(2)))
    for α ∈ 2:order
        P_view = _view_manifold_parameterization!(coeffs_view, P, α)
        λα = λ*α
        v = ldiv!(Df_hessenberg - λα*I, f̂(P_view, α))
        for j ∈ 1:n
            component(P, j)[α] = -v[j]
        end
    end
    return P
end

function manifold_ODE_equilibrium(c::AbstractVector, ξ::AbstractMatrix, λ::AbstractVector; Df, f̂, order::NTuple{N,Int}) where {N}
    length(c) == size(ξ, 1) && size(ξ, 2) == length(λ) == N || return throw(DimensionMismatch)
    n = length(c)
    P = initialize_manifold_parameterization(c, ξ, order)
    maximum(order) == 1 && return P
    Df_hessenberg = hessenberg(Df)
    coeffs_view = Vector{eltype(P)}(undef, dimension(TensorSpace(ntuple(i -> Taylor(2), Val(N)))))
    for α ∈ allindices(P.space.space)
        if sum(α) ≥ 2
            P_view = _view_manifold_parameterization!(coeffs_view, P, α)
            λα = mapreduce(*, +, λ, α)
            v = ldiv!(Df_hessenberg - λα*I, f̂(P_view, α))
            for j ∈ 1:n
                component(P, j)[α] = -v[j]
            end
        end
    end
    return P
end

##

function manifold_DDE_equilibrium(c, ξ, λ; Ψ, f̂, order::Int)
    P = initialize_manifold_parameterization(c, ξ, order)
    for α ∈ 2:order
        space_view = Taylor(α)
        P_view = Sequence(space_view, view(P, allindices(space_view)))
        λα = λ*α
        P[α] = - Ψ(λα) \ f̂(P_view, λα, α)
    end
    return P
end

function manifold_DDE_equilibrium(c, ξ::AbstractVector, λ::AbstractVector; Ψ, f̂, order::NTuple{N,Int}) where {N}
    length(ξ) == length(λ) == N || return throw(DimensionMismatch)
    P = initialize_manifold_parameterization(c, ξ, order)
    for α ∈ allindices(P.space)
        if sum(α) ≥ 2
            space_view = TensorSpace(map(Taylor, α))
            P_view = Sequence(space_view, view(P, allindices(space_view)))
            λα = mapreduce(*, +, λ, α)
            P[α] = - Ψ(λα) \ f̂(P_view, λα, α)
        end
    end
    return P
end

function manifold_DDE_equilibrium(c::AbstractVector, ξ::AbstractVector, λ; Ψ, f̂, order::Int)
    length(c) == length(ξ) || return throw(DimensionMismatch)
    n = length(c)
    P = initialize_manifold_parameterization(c, ξ, order)
    order == 1 && return P
    coeffs_view = Vector{eltype(P)}(undef, dimension(Taylor(2)))
    for α ∈ 2:order
        P_view = _view_manifold_parameterization!(coeffs_view, P, α)
        λα = λ*α
        v = ldiv!(lu!(Ψ(λα)), f̂(P_view, λα, α))
        for j ∈ 1:n
            component(P, j)[α] = -v[j]
        end
    end
    return P
end

function manifold_DDE_equilibrium(c::AbstractVector, ξ::AbstractMatrix, λ::AbstractVector; Ψ, f̂, order::NTuple{N,Int}) where {N}
    length(c) == size(ξ, 1) && size(ξ, 2) == length(λ) == N || return throw(DimensionMismatch)
    n = length(c)
    P = initialize_manifold_parameterization(c, ξ, order)
    maximum(order) == 1 && return P
    coeffs_view = Vector{eltype(P)}(undef, dimension(TensorSpace(ntuple(i -> Taylor(2), Val(N)))))
    for α ∈ allindices(P.space.space)
        if sum(α) ≥ 2
            P_view = _view_manifold_parameterization!(coeffs_view, P, α)
            λα = mapreduce(*, +, λ, α)
            v = ldiv!(lu!(Ψ(λα)), f̂(P_view, λα, α))
            for j ∈ 1:n
                component(P, j)[α] = -v[j]
            end
        end
    end
    return P
end
