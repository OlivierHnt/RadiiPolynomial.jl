## utilities

eachcomponent(A::Operator{CartesianSpace{T},CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}} =
    (view(A, i, j) for i ∈ Base.OneTo(N₂), j ∈ Base.OneTo(N₁))

eachcomponent(A::Operator{CartesianSpace{T},<:SequenceSpace}) where {N,T<:NTuple{N,SequenceSpace}} =
    (view(A, j) for i ∈ Base.OneTo(1), j ∈ Base.OneTo(N))

eachcomponent(A::Operator{<:SequenceSpace,CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} =
    (view(A, i) for i ∈ Base.OneTo(N), j ∈ Base.OneTo(1))

Base.eachcol(A::Operator{CartesianSpace{T},CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}} =
    (view(A, :, j) for j ∈ Base.OneTo(N₁))

Base.eachrow(A::Operator{CartesianSpace{T},CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}} =
    (view(A, i, :) for i ∈ Base.OneTo(N₂))

## getindex, view, setindex! - TODO: add methods, e.g. for AbstractUnitRange

for f ∈ (:getindex, :view)
    @eval begin
        Base.@propagate_inbounds function Base.$f(A::Operator{CartesianSpace{T},CartesianSpace{S}}, i::Int, j::Int) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
            @boundscheck(i < 1 || N₂ < i || j < 1 || N₁ < j && throw(BoundsError((A.codomain, A.domain), (i, j))))
            domain, codomain = A.domain[j], A.codomain[i]
            skip₁ = i == 1 ? 0 : mapreduce(k -> length(A.codomain[k]), +, 1:i-1)
            skip₂ = j == 1 ? 0 : mapreduce(k -> length(A.domain[k]), +, 1:j-1)
            return Operator(domain, codomain, view(A.coefficients, 1+skip₁:length(codomain)+skip₁, 1+skip₂:length(domain)+skip₂))
        end

        Base.@propagate_inbounds function Base.$f(A::Operator{CartesianSpace{T},<:CartesianSpace}, ::Colon, j::Int) where {N,T<:NTuple{N,SequenceSpace}}
            @boundscheck(j < 1 || N < j && throw(BoundsError(A.domain, j)))
            domain = A.domain[j]
            skip = j == 1 ? 0 : mapreduce(k -> length(A.domain[k]), +, 1:j-1)
            return Operator(domain, A.codomain, view(A.coefficients, :, 1+skip:length(domain)+skip))
        end

        Base.@propagate_inbounds function Base.$f(A::Operator{<:CartesianSpace,CartesianSpace{T}}, i::Int, ::Colon) where {N,T<:NTuple{N,SequenceSpace}}
            @boundscheck(i < 1 || N < i && throw(BoundsError(A.codomain, i)))
            codomain = A.codomain[i]
            skip = i == 1 ? 0 : mapreduce(k -> length(A.codomain[k]), +, 1:i-1)
            return Operator(A.domain, codomain, view(A.coefficients, 1+skip:length(codomain)+skip, :))
        end

        Base.@propagate_inbounds Base.$f(A::Operator{<:CartesianSpace,<:CartesianSpace}, ::Colon, ::Colon) =
            Operator(A.domain, A.codomain, view(A.coefficients, :, :))

        Base.@propagate_inbounds function Base.$f(A::Operator{CartesianSpace{T},<:SequenceSpace}, j::Int) where {N,T<:NTuple{N,SequenceSpace}}
            @boundscheck(j < 1 || N < j && throw(BoundsError(A.domain, j)))
            domain = A.domain[j]
            skip = j == 1 ? 0 : mapreduce(k -> length(A.domain[k]), +, 1:j-1)
            return Operator(domain, A.codomain, view(A.coefficients, :, 1+skip:length(domain)+skip))
        end

        Base.@propagate_inbounds Base.$f(A::Operator{<:CartesianSpace,<:SequenceSpace}, ::Colon) =
            Operator(A.domain, A.codomain, view(A.coefficients, :, :))

        Base.@propagate_inbounds function Base.$f(A::Operator{<:SequenceSpace,CartesianSpace{T}}, i::Int) where {N,T<:NTuple{N,SequenceSpace}}
            @boundscheck(i < 1 || N < i && throw(BoundsError(A.codomain, i)))
            codomain = A.codomain[i]
            skip = i == 1 ? 0 : mapreduce(k -> length(A.codomain[k]), +, 1:i-1)
            return Operator(A.domain, codomain, view(A.coefficients, 1+skip:length(codomain)+skip, :))
        end

        Base.@propagate_inbounds Base.$f(A::Operator{<:SequenceSpace,<:CartesianSpace}, ::Colon) =
            Operator(A.domain, A.codomain, view(A.coefficients, :, :))
    end
end

Base.@propagate_inbounds function Base.setindex!(A::Operator{CartesianSpace{T},CartesianSpace{S}}, x::Operator, i::Int, j::Int) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @boundscheck(i < 1 || N₂ < i || j < 1 || N₁ < j && throw(BoundsError((A.codomain, A.domain), (i, j))))
    domain, codomain = A.domain[j], A.codomain[i]
    domain == x.domain || codomain == x.codomain || return throw(ArgumentError)
    skip₁ = i == 1 ? 0 : mapreduce(k -> length(A.codomain[k]), +, 1:i-1)
    skip₂ = j == 1 ? 0 : mapreduce(k -> length(A.domain[k]), +, 1:j-1)
    return setindex!(A.coefficients, x.coefficients, 1+skip₁:length(codomain)+skip₁, 1+skip₂:length(domain)+skip₂)
end

Base.@propagate_inbounds function Base.setindex!(A::Operator{CartesianSpace{T},<:SequenceSpace}, x::Operator, j::Int) where {N,T<:NTuple{N,SequenceSpace}}
    @boundscheck(j < 1 || N < j && throw(BoundsError(A.domain, j)))
    domain = A.domain[j]
    domain == x.domain || A.codomain == x.codomain || return throw(ArgumentError)
    skip = j == 1 ? 0 : mapreduce(k -> length(A.domain[k]), +, 1:j-1)
    return setindex!(A.coefficients, x.coefficients, :, 1+skip:length(domain)+skip)
end

Base.@propagate_inbounds function Base.setindex!(A::Operator{<:SequenceSpace,CartesianSpace{T}}, x::Operator, j::Int) where {N,T<:NTuple{N,SequenceSpace}}
    @boundscheck(i < 1 || N < i && throw(BoundsError(A.codomain, i)))
    codomain = A.codomain[i]
    A.domain == x.domain || codomain == x.codomain || return throw(ArgumentError)
    skip = i == 1 ? 0 : mapreduce(k -> length(A.codomain[k]), +, 1:i-1)
    return setindex!(A.coefficients, x.coefficients, 1+skip:length(codomain)+skip, :)
end

## opnorm

LinearAlgebra.opnorm(A::Operator{<:CartesianSpace,<:CartesianSpace}) =
    opnorm(map(opnorm, eachcomponent(A)), Inf)
LinearAlgebra.opnorm(A::Operator{<:CartesianSpace,<:SequenceSpace}) =
    opnorm(map(opnorm, eachcomponent(A)), Inf)
LinearAlgebra.opnorm(A::Operator{<:SequenceSpace,<:CartesianSpace}) =
    opnorm(map(opnorm, eachcomponent(A)), Inf)

function LinearAlgebra.opnorm(A::Operator{CartesianSpace{T},CartesianSpace{S}}, ν, μ, p::Real=Inf) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == length(ν) && N₂ == length(μ)
    return opnorm(map((Aᵢ, tᵢ) -> opnorm(Aᵢ, tᵢ[1], tᵢ[2]), eachcomponent(A), Iterators.product(ν, μ)), p)
end
function LinearAlgebra.opnorm(A::Operator{CartesianSpace{T},<:SequenceSpace}, ν, μ, p::Real=Inf) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(ν)
    return opnorm(map((Aᵢ, νᵢ) -> opnorm(Aᵢ, νᵢ, μ), eachcomponent(A), ν), p)
end
function LinearAlgebra.opnorm(A::Operator{<:SequenceSpace,CartesianSpace{T}}, ν, μ, p::Real=Inf) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(μ)
    return opnorm(map((Aᵢ, μᵢ) -> opnorm(Aᵢ, ν, μᵢ), eachcomponent(A), μ), p)
end

## action

function Base.:*(A::Operator{<:CartesianSpace,<:CartesianSpace}, b::Sequence{<:CartesianSpace})
    @assert length(A.domain.spaces) == length(b.space.spaces)
    return mapreduce((Aⱼ, bⱼ) -> Aⱼ * bⱼ, +, eachcol(A), eachcomponent(b))
end

function Base.:*(A::Operator{<:CartesianSpace,<:SequenceSpace}, b::Sequence{<:CartesianSpace})
    @assert length(A.domain.spaces) == length(b.space.spaces)
    return mapreduce((Aⱼ, bⱼ) -> Aⱼ * bⱼ, +, eachcomponent(A), eachcomponent(b))
end

function Base.:*(A::Operator{<:SequenceSpace,<:CartesianSpace}, b::Sequence{<:SequenceSpace})
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(A.codomain, Vector{CoefType}(undef, length(A.codomain)))
    foreach((Cᵢ, Aᵢ) -> Cᵢ.coefficients .= (Aᵢ * b).coefficients, eachcomponent(c), eachcomponent(A))
    return c
end

## arithmetic

function Base.:+(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
    domain, codomain = A.domain ∪ B.domain, A.codomain ∪ B.codomain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, length(codomain), length(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ + Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

function Base.:-(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
    domain, codomain = A.domain ∪ B.domain, A.codomain ∪ B.codomain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, length(codomain), length(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ - Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

#

function +̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
    domain, codomain = A.domain ∪̄ B.domain, A.codomain ∪̄ B.codomain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, length(codomain), length(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ +̄ Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

function -̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
    domain, codomain = A.domain ∪̄ B.domain, A.codomain ∪̄ B.codomain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, length(codomain), length(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ -̄ Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

#

function +̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, b)
    @assert length(A.domain.spaces) == length(A.codomain.spaces)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, length(A.codomain), length(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for i ∈ eachindex(A.domain.spaces)
        Cᵢ = view(C, i, i)
        Cᵢ.coefficients .= (Cᵢ +̄ b).coefficients
    end
    return C
end

function +̄(b, A::Operator{<:CartesianSpace,<:CartesianSpace})
    @assert length(A.domain.spaces) == length(A.codomain.spaces)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, length(A.codomain), length(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for i ∈ eachindex(A.domain.spaces)
        Cᵢ = view(C, i, i)
        Cᵢ.coefficients .= (b +̄ Cᵢ).coefficients
    end
    return C
end

function -̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, b)
    @assert length(A.domain.spaces) == length(A.codomain.spaces)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, length(A.codomain), length(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for i ∈ eachindex(A.domain.spaces)
        Cᵢ = view(C, i, i)
        Cᵢ.coefficients .= (Cᵢ -̄ b).coefficients
    end
    return C
end

function -̄(b, A::Operator{<:CartesianSpace,<:CartesianSpace})
    @assert length(A.domain.spaces) == length(A.codomain.spaces)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, length(A.codomain), length(A.domain)))
    @. C.coefficients = -A.coefficients
    @inbounds for i ∈ eachindex(A.domain.spaces)
        Cᵢ = view(C, i, i)
        Cᵢ.coefficients .= (b +̄ Cᵢ).coefficients
    end
    return C
end

#

+̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling) =
    +̄(A, J.λ)
+̄(J::UniformScaling, A::Operator{<:CartesianSpace,<:CartesianSpace}) =
    +̄(J.λ, A)

-̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling) =
    -̄(A, J.λ)
-̄(J::UniformScaling, A::Operator{<:CartesianSpace,<:CartesianSpace}) =
    -̄(J.λ, A)

## eigen

function LinearAlgebra.eigen(A::Operator{<:CartesianSpace,<:CartesianSpace})
    Λ, Ξ = eigen(A.coefficients)
    @inbounds Ξ_ = map(i -> Sequence(A.domain, Ξ[:,i]), axes(Ξ, 2))
    return Λ, Ξ_
end

##

function Operator(domain::CartesianSpace, codomain::CartesianSpace, A::AbstractMatrix{T}) where {T<:Sequence}
    @assert length(codomain.spaces) == size(A, 1) && length(domain.spaces) == size(A, 2)
    C = Operator(domain, codomain, Matrix{eltype(T)}(undef, length(codomain), length(domain)))
    foreach((Cᵢ, Aᵢ) -> Cᵢ.coefficients .= Operator(Cᵢ.domain, Cᵢ.codomain, Aᵢ).coefficients, eachcomponent(C), A)
    return C
end

## calculus

Operator(domain::CartesianSpace, codomain::CartesianSpace, 𝒟::Derivative) =
    Operator(domain, codomain, mapreduce((s₁, s₂) -> Operator(s₁, s₂, 𝒟).coefficients, (x, y) -> cat(x, y; dims=(1,2)),  domain.spaces, codomain.spaces))

Operator(domain::CartesianSpace, codomain::CartesianSpace, ℐ::Integral) =
    Operator(domain, codomain, mapreduce((s₁, s₂) -> Operator(s₁, s₂, ℐ).coefficients, (x, y) -> cat(x, y; dims=(1,2)),  domain.spaces, codomain.spaces))
