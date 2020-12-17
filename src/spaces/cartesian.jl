struct CartesianSpace{T<:NTuple{N,SequenceSpace} where {N}} <: SequenceSpace
    spaces :: T
end

CartesianSpace(spaces::SequenceSpace...) = CartesianSpace(spaces)

LinearAlgebra.:×(s₁::SequenceSpace, s₂::SequenceSpace) = CartesianSpace((s₁, s₂))
LinearAlgebra.:×(s₁::SequenceSpace, s₂::CartesianSpace) = CartesianSpace((s₁, s₂.spaces...))
LinearAlgebra.:×(s₁::CartesianSpace, s₂::SequenceSpace) = CartesianSpace((s₁.spaces..., s₂))
LinearAlgebra.:×(s₁::CartesianSpace, s₂::CartesianSpace) = CartesianSpace((s₁.spaces..., s₂.spaces...))
LinearAlgebra.:×(s₁::CartesianSpace{Tuple{T}}, s₂::CartesianSpace{Tuple{}}) where {T<:SequenceSpace} = s₁.spaces[1]
LinearAlgebra.:×(s₁::CartesianSpace{Tuple{}}, s₂::CartesianSpace{Tuple{T}}) where {T<:SequenceSpace} = s₂.spaces[1]

## getindex

Base.@propagate_inbounds Base.getindex(s::CartesianSpace, c::Colon) = CartesianSpace(getindex(s.spaces, c))
Base.@propagate_inbounds Base.getindex(s::CartesianSpace, i::Int) = getindex(s.spaces, i)
Base.@propagate_inbounds Base.getindex(s::CartesianSpace, u::AbstractRange) = CartesianSpace(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::CartesianSpace, u::Vector{Int}) = CartesianSpace(getindex(s.spaces, u))

Base.front(s::CartesianSpace) = CartesianSpace(Base.front(s.spaces))
Base.tail(s::CartesianSpace) = CartesianSpace(Base.tail(s.spaces))

## order

order(s::CartesianSpace) = map(order, s.spaces)

## frequency

frequency(s::CartesianSpace) = map(frequency, s.spaces)

##

derivative_range(s::CartesianSpace) = CartesianSpace(map(derivative_range, s.spaces))

integral_range(s::CartesianSpace) = CartesianSpace(map(derivative_range, s.spaces))

##

Base.:∩(s₁::CartesianSpace{<:NTuple{N,SequenceSpace}}, s₂::CartesianSpace{<:NTuple{N,SequenceSpace}}) where {N} =
    CartesianSpace(map(∩, s₁.spaces, s₂.spaces))

Base.:∪(s₁::CartesianSpace{<:NTuple{N,SequenceSpace}}, s₂::CartesianSpace{<:NTuple{N,SequenceSpace}}) where {N} =
    CartesianSpace(map(∪, s₁.spaces, s₂.spaces))

∪̄(s₁::CartesianSpace{<:NTuple{N,SequenceSpace}}, s₂::CartesianSpace{<:NTuple{N,SequenceSpace}}) where {N} =
    CartesianSpace(map(∪̄, s₁.spaces, s₂.spaces))

Base.issubset(s₁::CartesianSpace{<:NTuple{N,SequenceSpace}}, s₂::CartesianSpace{<:NTuple{N,SequenceSpace}}) where {N} =
    all(s -> issubset(s[1], s[2]), zip(s₁.spaces, s₂.spaces))

Base.:(==)(s₁::CartesianSpace{<:NTuple{N,SequenceSpace}}, s₂::CartesianSpace{<:NTuple{N,SequenceSpace}}) where {N} =
    all(s -> ==(s[1], s[2]), zip(s₁.spaces, s₂.spaces))

## characterization of spaces

Base.length(s::CartesianSpace) = mapreduce(length, +, s.spaces)

Base.size(s::CartesianSpace) = tuple(length(s))

# Base.firstindex(s::CartesianSpace) = 1
#
# Base.lastindex(s::CartesianSpace) = length(s)
#
# Base.eachindex(s::CartesianSpace) = Base.OneTo(lastindex(s))
#
# Base.axes(s::CartesianSpace) = tuple(eachindex(s))

## promotion

Base.convert(::Type{CartesianSpace{T}}, space::CartesianSpace{<:NTuple{N,SequenceSpace}}) where {N,T<:Tuple{Vararg{SequenceSpace,N}}} =
    CartesianSpace{T}(ntuple(i -> convert(T.parameters[i], space.spaces[i]), N))

Base.promote_rule(::Type{CartesianSpace{T}}, ::Type{CartesianSpace{S}}) where {N,T<:Tuple{Vararg{SequenceSpace,N}},S<:Tuple{Vararg{SequenceSpace,N}}} =
    CartesianSpace{Tuple{map(promote_type, T.parameters, S.parameters)...}}

## show

Base.show(io::IO, space::CartesianSpace) = print(io, pretty_string(space))

pretty_string(space::CartesianSpace{Tuple{}}) = string(Tuple{}())

pretty_string(space::CartesianSpace) =
    mapreduce(sᵢ -> string(" ⨉ ", pretty_string(sᵢ)), *, Base.tail(space.spaces); init = pretty_string(space.spaces[1]))
