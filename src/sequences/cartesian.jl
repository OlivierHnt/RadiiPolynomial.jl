## utilities

eachcomponent(a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} =
    (view(a, i) for i ∈ Base.OneTo(N))

## getindex, view, setindex!

for f ∈ (:getindex, :view)
    @eval begin
        Base.@propagate_inbounds function Base.$f(a::Sequence{CartesianSpace{T}}, i::Int) where {N,T<:NTuple{N,SequenceSpace}}
            @boundscheck(i < 1 || N < i && throw(BoundsError(a.space, i)))
            space = a.space[i]
            skip = i == 1 ? 0 : mapreduce(j -> length(a.space[j]), +, 1:i-1)
            return Sequence(space, view(a.coefficients, 1+skip:length(space)+skip))
        end

        Base.@propagate_inbounds function Base.$f(a::Sequence{CartesianSpace{T}}, u::AbstractUnitRange) where {N,T<:NTuple{N,SequenceSpace}}
            @boundscheck(first(u) < 1 || N < last(u) && throw(BoundsError(a.space, u)))
            space = a.space[u]
            premier = first(u)
            skip = premier == 1 ? 0 : mapreduce(j -> length(a.space[j]), +, 1:premier-1)
            return Sequence(space, view(a.coefficients, 1+skip:length(space)+skip))
        end

        Base.@propagate_inbounds Base.$f(a::Sequence{<:CartesianSpace}, ::Colon) =
            Sequence(a.space, view(a.coefficients, :))
    end
end

Base.@propagate_inbounds function Base.setindex!(a::Sequence{CartesianSpace{T}}, x::Sequence, i::Int) where {N,T<:NTuple{N,SequenceSpace}}
    @boundscheck(i < 1 || N < i && throw(BoundsError(a.space, i)))
    space = a.space[i]
    space == x.space || return throw(ArgumentError)
    skip = i == 1 ? 0 : mapreduce(j -> length(a.space[j]), +, 1:i-1)
    return setindex!(a.coefficients, x.coefficients, 1+skip:length(space)+skip)
end

## norm

LinearAlgebra.norm(a::Sequence{<:CartesianSpace}) = norm(map(norm, eachcomponent(a)), Inf)

function LinearAlgebra.norm(a::Sequence{CartesianSpace{T}}, ν::NTuple{N₂,Any}, p::Real) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂}
    @assert N₁ == N₂
    return norm(map(norm, eachcomponent(a), ν), p)
end

## show

Base.show(io::IO, a::Sequence{CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} =
    print(io, pretty_string(a.space) * ":" * mapreduce(aᵢ -> "\n\n" * pretty_string(aᵢ), * , eachcomponent(a)))

## arithmetic

function Base.:+(a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
    space = a.space ∪ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients + b.coefficients
        return c
    else
        foreach((cᵢ, aᵢ, bᵢ) -> cᵢ.coefficients .= (aᵢ + bᵢ).coefficients, eachcomponent(c), eachcomponent(a), eachcomponent(b))
        return c
    end
end

function Base.:-(a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
    space = a.space ∪ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients - b.coefficients
        return c
    else
        foreach((cᵢ, aᵢ, bᵢ) -> cᵢ.coefficients .= (aᵢ - bᵢ).coefficients, eachcomponent(c), eachcomponent(a), eachcomponent(b))
        return c
    end
end

#

function +̄(a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
    space = a.space ∪̄ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients + b.coefficients
        return c
    else
        foreach((cᵢ, aᵢ, bᵢ) -> cᵢ.coefficients .= (aᵢ +̄ bᵢ).coefficients, eachcomponent(c), eachcomponent(a), eachcomponent(b))
        return c
    end
end

function -̄(a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
    space = a.space ∪ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients -̄ b.coefficients
        return c
    else
        foreach((cᵢ, aᵢ, bᵢ) -> cᵢ.coefficients .= (aᵢ -̄ bᵢ).coefficients, eachcomponent(c), eachcomponent(a), eachcomponent(b))
        return c
    end
end

## calculus

function differentiate(a::Sequence{T}) where {T<:CartesianSpace}
    space = derivative_range(a.space)
    return Sequence(space, mapreduce(aᵢ -> differentiate(aᵢ).coefficients, vcat, eachcomponent(a)))
end

function integrate(a::Sequence{T}) where {T<:CartesianSpace}
    space = integral_range(a.space)
    return Sequence(space, mapreduce(aᵢ -> integrate(aᵢ).coefficients, vcat, eachcomponent(a)))
end

## evaluate

(a::Sequence{<:CartesianSpace})(x) = evaluate(a, x)
(a::Sequence{<:CartesianSpace})(x...) = evaluate(a, x)

evaluate(a::Sequence{<:CartesianSpace}, x) = map(aᵢ -> evaluate(aᵢ, x), eachcomponent(a))
