## utilities

Base.length(A::Functional{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} = N # needed for map(f, A)

Base.iterate(A::Functional{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} = (view(A, 1), 2)
Base.iterate(A::Functional{CartesianSpace{T}}, i::Int) where {N,T<:NTuple{N,SequenceSpace}} = 1 ≤ i ≤ N ? (view(A, i), i+1) : nothing

## getindex, view, setindex!

for f ∈ (:getindex, :view)
    @eval Base.@propagate_inbounds function Base.$f(A::Functional{<:CartesianSpace}, i::Int)
        i == 1 && return Functional(A.domain[1], $f(A.coefficients, 1:length(A.domain[1])))
        len = mapreduce(j -> length(A.domain[j]), +, 1:i-1)
        indices = len+1:len+length(A.domain[i])
        return Functional(A.domain[i], $f(A.coefficients, indices))
    end
end

Base.@propagate_inbounds function Base.setindex!(A::Functional{<:CartesianSpace}, x::Functional, i::Int)
    @assert A.domain[i] == x.domain
    i == 1 && return setindex!(A.coefficients, x.coefficients, 1:length(A.domain[1]))
    len = mapreduce(j -> length(A.domain[j]), +, 1:i-1)
    indices = len+1:len+length(A.domain[i])
    return setindex!(A.coefficients, x.coefficients, indices)
end

## opnorm

LinearAlgebra.opnorm(A::Functional{<:CartesianSpace}) = opnorm(reshape(map(opnorm, A), 1, :), Inf)

function LinearAlgebra.opnorm(A::Functional{CartesianSpace{T}}, ν::NTuple{N₂,Any}, p::Real) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂}
    @assert N₁ == N₂
    return norm(map(opnorm, A, ν), p)
end

## action

function (A::Functional{CartesianSpace{T}})(b::Sequence{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    return mapreduce((Aᵢ, bᵢ) -> Aᵢ(bᵢ), +, A, b)
end

## arithmetic

function Base.:+(A::Functional{CartesianSpace{T}}, B::Functional{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    domain = A.domain ∪ B.domain
    if A.domain == B.domain
        return Functional(domain, A.coefficients + B.coefficients)
    else
        return Functional(domain, mapreduce((Aᵢ, Bᵢ) -> (Aᵢ + Bᵢ).coefficients, vcat, A, B))
    end
end

function Base.:-(A::Functional{CartesianSpace{T}}, B::Functional{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    domain = A.domain ∪ B.domain
    if a.space == b.space
        return Functional(domain, A.coefficients - B.coefficients)
    else
        return Functional(domain, mapreduce((Aᵢ, Bᵢ) -> (Aᵢ - Bᵢ).coefficients, vcat, A, B))
    end
end

#

function +̄(A::Functional{CartesianSpace{T}}, B::Functional{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    domain = A.domain ∪̄ B.domain
    if A.domain == B.domain
        return Functional(domain, A.coefficients + B.coefficients)
    else
        return Functional(domain, mapreduce((Aᵢ, Bᵢ) -> (Aᵢ +̄ Bᵢ).coefficients, vcat, A, B))
    end
end

function -̄(A::Functional{CartesianSpace{T}}, B::Functional{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    domain = A.domain ∪̄ B.domain
    if a.space == b.space
        return Functional(domain, A.coefficients - B.coefficients)
    else
        return Functional(domain, mapreduce((Aᵢ, Bᵢ) -> (Aᵢ -̄ Bᵢ).coefficients, vcat, A, B))
    end
end
