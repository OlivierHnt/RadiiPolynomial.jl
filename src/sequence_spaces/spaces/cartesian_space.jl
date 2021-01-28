"""
    CartesianSpace{T<:NTuple{N,SingleSpace} where {N}} <: VectorSpace

Space resulting from the cartesian product of some [`SingleSpace`](@ref).

Fields:
- `spaces :: T`
"""
struct CartesianSpace{T<:NTuple{N,SingleSpace} where {N}} <: VectorSpace
    spaces :: T
end

CartesianSpace(spaces::SingleSpace...) = CartesianSpace(spaces)

LinearAlgebra.:×(s₁::SingleSpace, s₂::SingleSpace) = CartesianSpace((s₁, s₂))
LinearAlgebra.:×(s₁::SingleSpace, s₂::CartesianSpace) = CartesianSpace((s₁, s₂.spaces...))
LinearAlgebra.:×(s₁::CartesianSpace, s₂::SingleSpace) = CartesianSpace((s₁.spaces..., s₂))
LinearAlgebra.:×(s₁::CartesianSpace, s₂::CartesianSpace) = CartesianSpace((s₁.spaces..., s₂.spaces...))

##

dimension(s::CartesianSpace) = mapreduce(dimension, +, s.spaces)
dimensions(s::CartesianSpace) = map(dimension, s.spaces)
dimensions(s::CartesianSpace, i::Int) = dimension(s.spaces[i])
startindex(s::CartesianSpace) = 1
endindex(s::CartesianSpace) = dimension(s)
allindices(s::CartesianSpace) = Base.OneTo(endindex(s))
isindexof(::Colon, space::CartesianSpace) = true
isindexof(i, space::CartesianSpace) = i ∈ allindices(space)
_findindex(i, space::CartesianSpace) = i

##

Base.issubset(s₁::CartesianSpace{<:NTuple{N,SingleSpace}}, s₂::CartesianSpace{<:NTuple{N,SingleSpace}}) where {N} =
    all(s -> issubset(s[1], s[2]), zip(s₁.spaces, s₂.spaces))

Base.:(==)(s₁::CartesianSpace{<:NTuple{N,SingleSpace}}, s₂::CartesianSpace{<:NTuple{N,SingleSpace}}) where {N} =
    all(s -> ==(s[1], s[2]), zip(s₁.spaces, s₂.spaces))

## getindex

Base.@propagate_inbounds Base.getindex(s::CartesianSpace, c::Colon) = CartesianSpace(getindex(s.spaces, c))
Base.@propagate_inbounds Base.getindex(s::CartesianSpace, i::Int) = getindex(s.spaces, i)
Base.@propagate_inbounds Base.getindex(s::CartesianSpace, u::AbstractRange) = CartesianSpace(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::CartesianSpace, u::Vector{Int}) = CartesianSpace(getindex(s.spaces, u))

Base.front(s::CartesianSpace) = CartesianSpace(Base.front(s.spaces))
Base.tail(s::CartesianSpace) = CartesianSpace(Base.tail(s.spaces))

## promotion

Base.convert(::Type{CartesianSpace{T}}, space::CartesianSpace{<:NTuple{N,SingleSpace}}) where {N,T<:Tuple{Vararg{SingleSpace,N}}} =
    CartesianSpace{T}(ntuple(i -> convert(T.parameters[i], space.spaces[i]), N))

Base.promote_rule(::Type{CartesianSpace{T}}, ::Type{CartesianSpace{S}}) where {N,T<:Tuple{Vararg{SingleSpace,N}},S<:Tuple{Vararg{SingleSpace,N}}} =
    CartesianSpace{Tuple{map(promote_type, T.parameters, S.parameters)...}}

## order

order(s::CartesianSpace) = map(order, s.spaces)
order(s::CartesianSpace, i::Int) = order(s.spaces[i])

## frequency

frequency(s::CartesianSpace) = map(frequency, s.spaces)
frequency(s::CartesianSpace, i::Int) = frequency(s.spaces[i])

## show

Base.show(io::IO, space::CartesianSpace) = print(io, pretty_string(space))

pretty_string(space::CartesianSpace) =
    mapreduce(sᵢ -> string(" ⨉ ", "(", pretty_string(sᵢ), ")"), *, Base.tail(space.spaces); init = "(" * pretty_string(space.spaces[1]) * ")")
pretty_string(space::CartesianSpace{Tuple{}}) = string(Tuple{}())
