struct SequenceStyle <: Base.Broadcast.BroadcastStyle end

Base.Broadcast.broadcastable(a::Sequence) = a

Base.Broadcast.BroadcastStyle(::Type{<:Sequence}) = SequenceStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle{0}, ::SequenceStyle) = SequenceStyle()
Base.Broadcast.BroadcastStyle(::SequenceStyle, ::Base.Broadcast.AbstractArrayStyle{0}) = SequenceStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle{1}, ::SequenceStyle) = SequenceStyle()
Base.Broadcast.BroadcastStyle(::SequenceStyle, ::Base.Broadcast.AbstractArrayStyle{1}) = SequenceStyle()

function Base.similar(bc::Base.Broadcast.Broadcasted{SequenceStyle}, ::Type{ElType}) where {ElType}
    space = _find_space(bc)
    return Sequence(space, similar(Vector{ElType}, length(space)))
end

_find_space(bc::Base.Broadcast.Broadcasted) = _find_space(bc.args)
_find_space(args::Tuple) = _find_space(_find_space(args[1]), _find_space(Base.tail(args)))
_find_space(a::Sequence) = a.space
_find_space(::Tuple{}) = nothing
_find_space(::Any) = nothing
_find_space(space::SequenceSpace, ::Nothing) = space
_find_space(::Nothing, space::SequenceSpace) = space
_find_space(::Nothing, ::Nothing) = nothing
function _find_space(s₁::T, s₂::S) where {T<:SequenceSpace,S<:SequenceSpace}
    @assert s₁ == s₂
    return convert(promote_type(T, S), s₁)
end

@inline function Base.copyto!(dest::Sequence, src::Sequence)
    dest.space == src.space || return throw(ArgumentError())
    copyto!(dest.coefficients, src.coefficients)
    return dest
end

@inline function Base.copyto!(dest::Sequence, bc::Base.Broadcast.Broadcasted)
    bc.f === identity && bc.args isa Tuple{Sequence} && return copyto!(dest, bc.args[1])
    bc′ = Base.Broadcast.preprocess(dest, bc)
    @inbounds @simd for i ∈ eachindex(bc′)
        dest.coefficients[i] = bc′[i]
    end
    return dest
end

Base.Broadcast._broadcast_getindex(a::Sequence, i::Int) =
    Base.Broadcast._broadcast_getindex(a.coefficients, i)
