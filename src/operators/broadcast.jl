struct OperatorStyle <: Base.Broadcast.BroadcastStyle end

Base.Broadcast.broadcastable(A::Operator) = A

Base.Broadcast.BroadcastStyle(::Type{<:Operator}) = OperatorStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle{0}, ::OperatorStyle) = OperatorStyle()
Base.Broadcast.BroadcastStyle(::OperatorStyle, ::Base.Broadcast.AbstractArrayStyle{0}) = OperatorStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle{1}, ::OperatorStyle) = OperatorStyle()
Base.Broadcast.BroadcastStyle(::OperatorStyle, ::Base.Broadcast.AbstractArrayStyle{1}) = OperatorStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle{2}, ::OperatorStyle) = OperatorStyle()
Base.Broadcast.BroadcastStyle(::OperatorStyle, ::Base.Broadcast.AbstractArrayStyle{2}) = OperatorStyle()

function Base.similar(bc::Base.Broadcast.Broadcasted{OperatorStyle}, ::Type{ElType}) where {ElType}
    domain, codomain = _find_domain_codomain(bc)
    return Operator(domain, codomain, similar(Matrix{ElType}, length(codomain), length(domain)))
end

_find_domain_codomain(bc::Base.Broadcast.Broadcasted) = _find_domain_codomain(bc.args)
_find_domain_codomain(args::Tuple) = _find_domain_codomain(_find_domain_codomain(args[1]), _find_domain_codomain(Base.tail(args)))
_find_domain_codomain(A::Operator) = (A.domain, A.codomain)
_find_domain_codomain(::Tuple{}) = nothing
_find_domain_codomain(::Any) = nothing
_find_domain_codomain(domain_codomain::NTuple{2,SequenceSpace}, ::Nothing) = domain_codomain
_find_domain_codomain(::Nothing, domain_codomain::NTuple{2,SequenceSpace}) = domain_codomain
_find_domain_codomain(::Nothing, ::Nothing) = nothing
function _find_domain_codomain(s₁::T, s₂::S) where {T<:NTuple{2,SequenceSpace},S<:NTuple{2,SequenceSpace}}
    @assert s₁ == s₂
    return convert(promote_type(T, S), s₁)
end

@inline function Base.copyto!(dest::Operator, src::Operator)
    (dest.domain == src.domain && dest.codomain == src.codomain) || return throw(ArgumentError())
    copyto!(dest.coefficients, src.coefficients)
    return dest
end

@inline function Base.copyto!(dest::Operator, bc::Base.Broadcast.Broadcasted)
    bc.f === identity && bc.args isa Tuple{Operator} && return copyto!(dest, bc.args[1])
    bc′ = Base.Broadcast.preprocess(dest, bc)
    @inbounds @simd for i ∈ eachindex(bc′)
        dest.coefficients[i] = bc′[i]
    end
    return dest
end

Base.Broadcast._broadcast_getindex(A::Operator, I::CartesianIndex{2}) =
    Base.Broadcast._broadcast_getindex(A.coefficients, I)

# to allow A[...] .= f.(...)
Base.@propagate_inbounds Base.Broadcast.dotview(A::Operator, α, β) = view(A, α, β)
