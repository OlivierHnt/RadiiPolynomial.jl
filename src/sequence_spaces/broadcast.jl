# Extend broadcasting for Sequence

struct SequenceStyle <: Base.Broadcast.BroadcastStyle end

Base.Broadcast.broadcastable(a::Sequence) = a

Base.Broadcast.BroadcastStyle(::Type{<:Sequence}) = SequenceStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle, ::SequenceStyle) = SequenceStyle()
Base.Broadcast.BroadcastStyle(::SequenceStyle, ::Base.Broadcast.AbstractArrayStyle) = SequenceStyle()

function Base.similar(bc::Base.Broadcast.Broadcasted{SequenceStyle}, ::Type{ElType}) where {ElType}
    space = _find_space(bc)
    return Sequence(space, similar(Vector{ElType}, dimension(space)))
end

_find_space(bc::Base.Broadcast.Broadcasted) = _find_space(bc.args)
_find_space(args::Tuple) = _find_space(_find_space(args[1]), _find_space(Base.tail(args)))
_find_space(a::Sequence) = space(a)
_find_space(::Tuple{}) = nothing
_find_space(::Any) = nothing
_find_space(space::VectorSpace, ::Nothing) = space
_find_space(::Nothing, space::VectorSpace) = space
_find_space(::Nothing, ::Nothing) = nothing
function _find_space(s₁::T, s₂::S) where {T<:VectorSpace,S<:VectorSpace}
    s₁ == s₂ || return throw(ArgumentError("spaces must be equal: s₁ is $s₁, s₂ is $s₂"))
    return convert(promote_type(T, S), s₁)
end

function Base.copyto!(dest::Sequence, bc::Base.Broadcast.Broadcasted)
    axes(dest) == axes(bc) || return throw(DimensionMismatch)
    bc′ = Base.Broadcast.preprocess(dest, bc)
    @inbounds @simd for i ∈ eachindex(bc′)
        coefficients(dest)[i] = bc′[i]
    end
    return dest
end

Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(a::Sequence, i::Int) =
    Base.Broadcast._broadcast_getindex(coefficients(a), i)

Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(a::Sequence, I::CartesianIndex) =
    Base.Broadcast._broadcast_getindex(coefficients(a), I)

# to allow a[...] .= f.(...)
Base.@propagate_inbounds Base.Broadcast.dotview(a::Sequence, α) = view(a, α)

# Extend broadcasting for LinearOperator

struct LinearOperatorStyle <: Base.Broadcast.BroadcastStyle end

Base.Broadcast.broadcastable(A::LinearOperator) = A

Base.Broadcast.BroadcastStyle(::Type{<:LinearOperator}) = LinearOperatorStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle, ::LinearOperatorStyle) = LinearOperatorStyle()
Base.Broadcast.BroadcastStyle(::LinearOperatorStyle, ::Base.Broadcast.AbstractArrayStyle) = LinearOperatorStyle()

function Base.similar(bc::Base.Broadcast.Broadcasted{LinearOperatorStyle}, ::Type{ElType}) where {ElType}
    domain, codomain = _find_domain_codomain(bc)
    return LinearOperator(domain, codomain, similar(Matrix{ElType}, dimension(codomain), dimension(domain)))
end

_find_domain_codomain(bc::Base.Broadcast.Broadcasted) = _find_domain_codomain(bc.args)
_find_domain_codomain(args::Tuple) = _find_domain_codomain(_find_domain_codomain(args[1]), _find_domain_codomain(Base.tail(args)))
_find_domain_codomain(A::LinearOperator) = (A.domain, A.codomain)
_find_domain_codomain(::Tuple{}) = nothing
_find_domain_codomain(::Any) = nothing
_find_domain_codomain(domain_codomain::NTuple{2,VectorSpace}, ::Nothing) = domain_codomain
_find_domain_codomain(::Nothing, domain_codomain::NTuple{2,VectorSpace}) = domain_codomain
_find_domain_codomain(::Nothing, ::Nothing) = nothing
function _find_domain_codomain(s₁::Tuple{T₁,S₁}, s₂::Tuple{T₂,S₂}) where {T₁<:VectorSpace,S₁<:VectorSpace,T₂<:VectorSpace,S₂<:VectorSpace}
    s₁ == s₂ || return throw(ArgumentError("spaces must be equal: s₁ is $s₁, s₂ is $s₂"))
    return @inbounds (convert(promote_type(T₁, T₂), s₁[1]), convert(promote_type(S₁, S₂), s₁[2]))
end

function Base.copyto!(dest::LinearOperator, bc::Base.Broadcast.Broadcasted)
    axes(dest) == axes(bc) || return throw(DimensionMismatch)
    bc′ = Base.Broadcast.preprocess(dest, bc)
    @inbounds @simd for i ∈ eachindex(bc′)
        dest.coefficients[i] = bc′[i]
    end
    return dest
end

Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(A::LinearOperator, i::Int) =
    Base.Broadcast._broadcast_getindex(A.coefficients, i)

Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(A::LinearOperator, I::CartesianIndex) =
    Base.Broadcast._broadcast_getindex(A.coefficients, I)

# to allow A[...] .= f.(...)
Base.@propagate_inbounds Base.Broadcast.dotview(A::LinearOperator, α, β) = view(A, α, β)
