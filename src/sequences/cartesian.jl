## utilities

Base.length(a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} = N # needed for map(f, a)

Base.iterate(a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} = (view(a, 1), 2)
Base.iterate(a::Sequence{CartesianSpace{T}}, i::Int) where {N,T<:NTuple{N,SequenceSpace}} = 1 ≤ i ≤ N ? (view(a, i), i+1) : nothing

## getindex, view, setindex!

for f ∈ (:getindex, :view)
    @eval Base.@propagate_inbounds function Base.$f(a::Sequence{<:CartesianSpace}, i::Int)
        i == 1 && return Sequence(a.space[1], $f(a.coefficients, 1:length(a.space[1])))
        len = mapreduce(j -> length(a.space[j]), +, 1:i-1)
        indices = len+1:len+length(a.space[i])
        return Sequence(a.space[i], $f(a.coefficients, indices))
    end
end

Base.@propagate_inbounds function Base.setindex!(a::Sequence{<:CartesianSpace}, x::Sequence, i::Int)
    @assert a.space[i] == x.space
    i == 1 && return setindex!(a.coefficients, x.coefficients, 1:length(a.space[1]))
    len = mapreduce(j -> length(a.space[j]), +, 1:i-1)
    indices = len+1:len+length(a.space[i])
    return setindex!(a.coefficients, x.coefficients, indices)
end

## norm

LinearAlgebra.norm(a::Sequence{<:CartesianSpace}) = norm(map(norm, a), Inf)

function LinearAlgebra.norm(a::Sequence{CartesianSpace{T}}, ν::NTuple{N₂,Any}, p::Real) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂}
    @assert N₁ == N₂
    return norm(map(norm, a, ν), p)
end

## show

Base.show(io::IO, a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} =
    print(io, pretty_string(a.space) * ":" * mapreduce(aᵢ -> "\n\n" * pretty_string(aᵢ), * , a))

## arithmetic

function Base.:+(a::Sequence{CartesianSpace{T}}, b::Sequence{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    space = a.space ∪ b.space
    if a.space == b.space
        return Sequence(space, a.coefficients + b.coefficients)
    else
        return Sequence(space, mapreduce((aᵢ, bᵢ) -> (aᵢ + bᵢ).coefficients, vcat, a, b))
    end
end

function Base.:-(a::Sequence{CartesianSpace{T}}, b::Sequence{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    space = a.space ∪ b.space
    if a.space == b.space
        return Sequence(space, a.coefficients - b.coefficients)
    else
        return Sequence(space, mapreduce((aᵢ, bᵢ) -> (aᵢ - bᵢ).coefficients, vcat, a, b))
    end
end

#

function Base.:+(a::Sequence{CartesianSpace{T}}, b::Vector{<:Sequence}) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    space = CartesianSpace(ntuple(i -> a.space[i] ∪ b[i].space, Val(N)))
    return Sequence(space, mapreduce((aᵢ, bᵢ) -> (aᵢ + bᵢ).coefficients, vcat, a, b))
end

function Base.:+(b::Vector{<:Sequence}, a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    space = CartesianSpace(ntuple(i -> a.space[i] ∪ b[i].space, Val(N)))
    return Sequence(space, mapreduce((aᵢ, bᵢ) -> (bᵢ + aᵢ).coefficients, vcat, a, b))
end

function Base.:-(a::Sequence{CartesianSpace{T}}, b::Vector{<:Sequence}) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    space = CartesianSpace(ntuple(i -> a.space[i] ∪ b[i].space, Val(N)))
    return Sequence(space, mapreduce((aᵢ, bᵢ) -> (aᵢ - bᵢ).coefficients, vcat, a, b))
end

function Base.:-(b::Vector{<:Sequence}, a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    space = CartesianSpace(ntuple(i -> a.space[i] ∪ b[i].space, Val(N)))
    return Sequence(space, mapreduce((aᵢ, bᵢ) -> (bᵢ - aᵢ).coefficients, vcat, a, b))
end

#

function Base.:+(a::Sequence{CartesianSpace{T}}, b::Vector) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    return Sequence(a.space, mapreduce((aᵢ, bᵢ) -> (aᵢ + bᵢ).coefficients, vcat, a, b))
end

function Base.:+(b::Vector, a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    return Sequence(a.space, mapreduce((aᵢ, bᵢ) -> (bᵢ + aᵢ).coefficients, vcat, a, b))
end

function Base.:-(a::Sequence{CartesianSpace{T}}, b::Vector) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    return Sequence(a.space, mapreduce((aᵢ, bᵢ) -> (aᵢ - bᵢ).coefficients, vcat, a, b))
end

function Base.:-(b::Vector, a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    return Sequence(a.space, mapreduce((aᵢ, bᵢ) -> (bᵢ - aᵢ).coefficients, vcat, a, b))
end

#

function +̄(a::Sequence{CartesianSpace{T}}, b::Sequence{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    space = a.space ∪̄ b.space
    if a.space == b.space
        return Sequence(space, a.coefficients + b.coefficients)
    else
        return Sequence(space, mapreduce((aᵢ, bᵢ) -> (aᵢ +̄ bᵢ).coefficients, vcat, a, b))
    end
end

function -̄(a::Sequence{CartesianSpace{T}}, b::Sequence{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    space = a.space ∪̄ b.space
    if a.space == b.space
        return Sequence(space, a.coefficients - b.coefficients)
    else
        return Sequence(space, mapreduce((aᵢ, bᵢ) -> (aᵢ -̄ bᵢ).coefficients, vcat, a, b))
    end
end

#

function +̄(a::Sequence{CartesianSpace{T}}, b::Vector{<:Sequence}) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    space = CartesianSpace(ntuple(i -> a.space[i] ∪̄ b[i].space, Val(N)))
    return Sequence(space, mapreduce((aᵢ, bᵢ) -> (aᵢ +̄ bᵢ).coefficients, vcat, a, b))
end

function +̄(b::Vector{<:Sequence}, a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    space = CartesianSpace(ntuple(i -> a.space[i] ∪̄ b[i].space, Val(N)))
    return Sequence(space, mapreduce((aᵢ, bᵢ) -> (bᵢ +̄ aᵢ).coefficients, vcat, a, b))
end

function -̄(a::Sequence{CartesianSpace{T}}, b::Vector{<:Sequence}) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    space = CartesianSpace(ntuple(i -> a.space[i] ∪̄ b[i].space, Val(N)))
    return Sequence(space, mapreduce((aᵢ, bᵢ) -> (aᵢ -̄ bᵢ).coefficients, vcat, a, b))
end

function -̄(b::Vector{<:Sequence}, a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(b)
    space = CartesianSpace(ntuple(i -> a.space[i] ∪̄ b[i].space, Val(N)))
    return Sequence(space, mapreduce((aᵢ, bᵢ) -> (bᵢ -̄ aᵢ).coefficients, vcat, a, b))
end

## calculus

function differentiate(a::Sequence{T}) where {T<:CartesianSpace}
    ∂a = map(differentiate, a)
    space::T = mapreduce(∂aᵢ -> ∂aᵢ.space, ×, ∂a) # WARNING: not type stable
    return Sequence(space, mapreduce(∂aᵢ -> ∂aᵢ.coefficients, vcat, ∂a))
end
