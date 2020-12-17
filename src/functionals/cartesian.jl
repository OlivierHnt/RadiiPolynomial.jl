## utilities

eachcomponent(A::Functional{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} =
    (view(A, i) for i ∈ Base.OneTo(N))

## getindex, view, setindex!

for f ∈ (:getindex, :view)
    @eval begin
        Base.@propagate_inbounds function Base.$f(A::Functional{CartesianSpace{T}}, i::Int) where {N,T<:NTuple{N,SequenceSpace}}
            @boundscheck(i < 1 || N < i && throw(BoundsError(A.domain, i)))
            domain = A.domain[i]
            skip = i == 1 ? 0 : mapreduce(j -> length(A.domain[j]), +, 1:i-1)
            return Functional(domain, view(A.coefficients, 1+skip:length(domain)+skip))
        end

        Base.@propagate_inbounds function Base.$f(A::Functional{CartesianSpace{T}}, u::AbstractUnitRange) where {N,T<:NTuple{N,SequenceSpace}}
            @boundscheck(first(u) < 1 || N < last(u) && throw(BoundsError(A.domain, u)))
            domain = A.domain[u]
            premier = first(u)
            skip = premier == 1 ? 0 : mapreduce(j -> length(A.domain[j]), +, 1:premier-1)
            return Functional(domain, view(A.coefficients, 1+skip:length(domain)+skip))
        end

        Base.@propagate_inbounds Base.$f(A::Functional{<:CartesianSpace}, ::Colon) =
            Functional(A.domain, view(A.coefficients, :))
    end
end

Base.@propagate_inbounds function Base.setindex!(A::Functional{CartesianSpace{T}}, x::Functional, i::Int) where {N,T<:NTuple{N,SequenceSpace}}
    @boundscheck(i < 1 || N < i && throw(BoundsError(A.domain, i)))
    domain = A.domain[i]
    domain == x.domain || return throw(ArgumentError)
    skip = i == 1 ? 0 : mapreduce(j -> length(A.domain[j]), +, 1:i-1)
    return setindex!(A.coefficients, x.coefficients, 1+skip:length(domain)+skip)
end

## opnorm

LinearAlgebra.opnorm(A::Functional{<:CartesianSpace}) = norm(map(opnorm, eachcomponent(A)), 1)

function LinearAlgebra.opnorm(A::Functional{CartesianSpace{T}}, ν::NTuple{N₂,Any}, p::Real) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂}
    @assert N₁ == N₂
    return opnorm(transpose(map(opnorm, eachcomponent(A), ν)), p)
end

## action

function Base.:*(A::Functional{CartesianSpace{T}}, b::Sequence{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    if A.domain == b.space
        s = zero(promote_type(eltype(A), eltype(b)))
        for (Aᵢ,bᵢ) ∈ zip(A.coefficients, b.coefficients)
            s += Aᵢ*bᵢ
        end
        return s
    else
        return mapreduce((Aᵢ, bᵢ) -> Aᵢ * bᵢ, +, eachcomponent(A), eachcomponent(b))
    end
end

## arithmetic

function Base.:+(A::Functional{<:CartesianSpace}, B::Functional{<:CartesianSpace})
    domain = A.domain ∪ B.domain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Sequence(domain, Vector{CoefType}(undef, length(domain)))
    if A.domain == B.domain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ + Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

function Base.:-(A::Functional{<:CartesianSpace}, B::Functional{<:CartesianSpace})
    domain = A.domain ∪ B.domain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Sequence(domain, Vector{CoefType}(undef, length(domain)))
    if A.domain == B.domain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ - Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

#

function +̄(A::Functional{<:CartesianSpace}, B::Functional{<:CartesianSpace})
    domain = A.domain ∪̄ B.domain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Sequence(domain, Vector{CoefType}(undef, length(domain)))
    if A.domain == B.domain
        @. C.coefficients = A.coefficients +̄ B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ +̄ Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

function -̄(A::Functional{<:CartesianSpace}, B::Functional{<:CartesianSpace})
    domain = A.domain ∪̄ B.domain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Sequence(domain, Vector{CoefType}(undef, length(domain)))
    if A.domain == B.domain
        @. C.coefficients = A.coefficients -̄ B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ -̄ Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end
