struct FunctionalStyle <: Base.Broadcast.BroadcastStyle end

Base.Broadcast.broadcastable(A::Functional) = A

Base.Broadcast.BroadcastStyle(::Type{<:Functional}) = FunctionalStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle{0}, ::FunctionalStyle) = FunctionalStyle()
Base.Broadcast.BroadcastStyle(::FunctionalStyle, ::Base.Broadcast.AbstractArrayStyle{0}) = FunctionalStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle{1}, ::FunctionalStyle) = FunctionalStyle()
Base.Broadcast.BroadcastStyle(::FunctionalStyle, ::Base.Broadcast.AbstractArrayStyle{1}) = FunctionalStyle()

function Base.similar(bc::Base.Broadcast.Broadcasted{FunctionalStyle}, ::Type{ElType}) where {ElType}
    domain = _find_domain(bc)
    return Functional(domain, similar(Vector{ElType}, length(domain)))
end

_find_domain(bc::Base.Broadcast.Broadcasted) = _find_domain(bc.args)
_find_domain(args::Tuple) = _find_domain(_find_domain(args[1]), _find_domain(Base.tail(args)))
_find_domain(A::Functional) = A.domain
_find_domain(::Tuple{}) = nothing
_find_domain(::Any) = nothing
_find_domain(domain::SequenceSpace, ::Nothing) = domain
_find_domain(::Nothing, domain::SequenceSpace) = domain
_find_domain(::Nothing, ::Nothing) = nothing
function _find_domain(s₁::T, s₂::S) where {T<:SequenceSpace,S<:SequenceSpace}
    @assert s₁ == s₂
    return convert(promote_type(T, S), s₁)
end

@inline function Base.copyto!(dest::Functional, src::Functional)
    dest.domain == src.domain || return throw(ArgumentError())
    copyto!(dest.coefficients, src.coefficients)
    return dest
end

@inline function Base.copyto!(dest::Functional, bc::Base.Broadcast.Broadcasted)
    bc.f === identity && bc.args isa Tuple{Functional} && return copyto!(dest, bc.args[1])
    bc′ = Base.Broadcast.preprocess(dest, bc)
    @inbounds @simd for i ∈ eachindex(bc′)
        dest.coefficients[i] = bc′[i]
    end
    return dest
end

Base.Broadcast._broadcast_getindex(A::Functional, i::Int) =
    Base.Broadcast._broadcast_getindex(A.coefficients, i)

# to allow A[...] .= f.(...)
Base.@propagate_inbounds Base.Broadcast.dotview(A::Functional, α) = view(A, α)
