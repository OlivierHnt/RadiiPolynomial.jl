"""
    Operator{T<:VectorSpace,S<:VectorSpace,R<:Union{AbstractMatrix,Factorization}}

Compactly supported operator with effective domain and codomain.

Fields:
- `domain :: T`
- `codomain :: S`
- `coefficients :: R`
"""
struct Operator{T<:VectorSpace,S<:VectorSpace,R<:Union{AbstractMatrix,Factorization}}
    domain :: T
    codomain :: S
    coefficients :: R
    function Operator{T,S,R}(domain::T, codomain::S, coefficients::R) where {T<:VectorSpace,S<:VectorSpace,R<:Union{AbstractMatrix,Factorization}}
        @assert dimension(codomain) == size(coefficients, 1) && dimension(domain) == size(coefficients, 2)
        return new{T,S,R}(domain, codomain, coefficients)
    end
end

Operator(domain::T, codomain::S, coefficients::R) where {T<:VectorSpace,S<:VectorSpace,R<:Union{AbstractMatrix,Factorization}} =
    Operator{T,S,R}(domain, codomain, coefficients)

## domain, codomain, coefficients

domain(A::Operator) = A.domain
domain(A::Operator, j::Int) = A.domain[j]
codomain(A::Operator) = A.codomain
codomain(A::Operator, i::Int) = A.codomain[i]
coefficients(A::Operator) = A.coefficients

## order, frequency

order(A::Operator) = (order(A.domain), order(A.codomain))
order(A::Operator, i::Int, j::Int) = (order(A.domain, j), order(A.codomain, i))

frequency(A::Operator) = (frequency(A.domain), frequency(A.codomain))
frequency(A::Operator, i::Int, j::Int) = (frequency(A.domain, j), frequency(A.codomain, i))

## utilities

function Base.firstindex(A::Operator, i::Int)
    i == 1 && return startindex(A.codomain)
    i == 2 && return startindex(A.domain)
    return 1
end

function Base.lastindex(A::Operator, i::Int)
    i == 1 && return endindex(A.codomain)
    i == 2 && return endindex(A.domain)
    return 1
end

Base.length(A::Operator) = length(A.coefficients)

Base.size(A::Operator) = size(A.coefficients)
Base.size(A::Operator, i::Int) = size(A.coefficients, i)

Base.iterate(A::Operator) = iterate(A.coefficients)
Base.iterate(A::Operator, i) = iterate(A.coefficients, i)

Base.eltype(A::Operator) = eltype(A.coefficients)
Base.eltype(::Type{Operator{T,S,R}}) where {T,S,R} = eltype(R)

## getindex, view, setindex!

for f ∈ (:getindex, :view)
    @eval Base.@propagate_inbounds function Base.$f(A::Operator, α, β)
        @boundscheck(!isindexof(α, A.codomain) && !isindexof(β, A.domain) && throw(BoundsError((A.codomain, A.domain), (α, β))))
        return $f(A.coefficients, _findindex(α, A.codomain), _findindex(β, A.domain))
    end
end

Base.@propagate_inbounds function Base.setindex!(A::Operator, x, α, β)
    @boundscheck(!isindexof(α, A.codomain) && !isindexof(β, A.domain) && throw(BoundsError((A.codomain, A.domain), (α, β))))
    return setindex!(A.coefficients, x, _findindex(α, A.codomain), _findindex(β, A.domain))
end

## ==, iszero, isapprox

Base.:(==)(A::Operator, B::Operator) = A.codomain == B.codomain && A.domain == B.domain && A.coefficients == B.coefficients

Base.iszero(A::Operator) = iszero(A.coefficients)

Base.isapprox(A::Operator, B::Operator; kwargs...) = A.codomain == B.codomain && A.domain == B.domain && isapprox(A.coefficients, B.coefficients; kwargs...)

## copy, similar

Base.copy(A::Operator) = Operator(A.domain, A.codomain, copy(A.coefficients))

Base.similar(A::Operator) = Operator(A.domain, A.codomain, similar(A.coefficients))

## zero, one

Base.zero(A::Operator) = Operator(A.domain, A.codomain, zero.(A.coefficients))

function Base.one(A::Operator)
    length(A.domain.spaces) == length(A.codomain.spaces) || return throw(DimensionMismatch)
    CoefType = eltype(A)
    B = zero(A)
    @inbounds for α ∈ ∩(allindices(A.domain), allindices(A.codomain))
        B[α,α] = one(CoefType)
    end
    return B
end

## float, complex, real, imag, conj, conj!

for f ∈ (:float, :complex, :real, :imag, :conj, :conj!)
    @eval Base.$f(A::Operator) = Operator(A.domain, A.codomain, $f(A.coefficients))
end

## promotion

Base.convert(::Type{Operator{T₁,S₁,R₁}}, A::Operator{T₂,S₂,R₂}) where {T₁,S₁,R₁,T₂,S₂,R₂} =
    Operator{T₁,S₁,R₁}(convert(T₁, A.domain), convert(S₁, A.codomain), convert(R₁, A.coefficients))

Base.promote_rule(::Type{Operator{T₁,S₁,R₁}}, ::Type{Operator{T₂,S₂,R₂}}) where {T₁,S₁,R₁,T₂,S₂,R₂} =
    Operator{promote_type(T₁,T₂),promote_type(S₁,S₂),promote_type(R₁,R₂)}

##

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
    return Operator(domain, codomain, similar(Matrix{ElType}, dimension(codomain), dimension(domain)))
end

_find_domain_codomain(bc::Base.Broadcast.Broadcasted) = _find_domain_codomain(bc.args)
_find_domain_codomain(args::Tuple) = _find_domain_codomain(_find_domain_codomain(args[1]), _find_domain_codomain(Base.tail(args)))
_find_domain_codomain(A::Operator) = (A.domain, A.codomain)
_find_domain_codomain(::Tuple{}) = nothing
_find_domain_codomain(::Any) = nothing
_find_domain_codomain(domain_codomain::NTuple{2,VectorSpace}, ::Nothing) = domain_codomain
_find_domain_codomain(::Nothing, domain_codomain::NTuple{2,VectorSpace}) = domain_codomain
_find_domain_codomain(::Nothing, ::Nothing) = nothing
function _find_domain_codomain(s₁::T, s₂::S) where {T<:NTuple{2,VectorSpace},S<:NTuple{2,VectorSpace}}
    @assert s₁ == s₂
    return convert(promote_type(T, S), s₁)
end

@inline function Base.copyto!(dest::Operator, src::Operator)
    (dest.domain == src.domain && dest.codomain == src.codomain) || return throw(ArgumentError)
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

## SEQUENCE SPACE

function Operator(domain::Taylor, codomain::Taylor, a::Sequence{Taylor})
    CoefType = eltype(a)
    A = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    A.coefficients .= zero(CoefType)
    @inbounds for j ∈ allindices(domain), i ∈ j:min(order(codomain),order(a.space)+j)
        A[i,j] = a[i-j]
    end
    return A
end

function Operator(domain::Fourier, codomain::Fourier, a::Sequence{<:Fourier})
    @assert domain.frequency == codomain.frequency == a.space.frequency
    CoefType = eltype(a)
    A = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    A.coefficients .= zero(CoefType)
    @inbounds for j ∈ allindices(domain), i ∈ max(-order(codomain),-order(a.space)+j):min(order(codomain),order(a.space)+j)
        A[i,j] = a[i-j]
    end
    return A
end

function Operator(domain::Chebyshev, codomain::Chebyshev, a::Sequence{Chebyshev})
    CoefType = float(eltype(a))
    A = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    A.coefficients .= zero(CoefType)
    for j ∈ allindices(domain), i ∈ max(-order(codomain),-order(a.space)+j):min(order(codomain),order(a.space)+j)
        if abs(i-j) == 0
            A[abs(i),j] += a[abs(i-j)]
        else
            A[abs(i),j] += a[abs(i-j)]
        end
    end
    @inbounds A[0,1:end] .*= 2
    @inbounds A[1:end,0] ./= 2
    return A
end

## CARTESIAN SPACE

function Operator(domain::CartesianSpace, codomain::CartesianSpace, A::AbstractMatrix{T}) where {T<:Sequence}
    length(codomain.spaces) == size(A, 1) && length(domain.spaces) == size(A, 2) || return throw(DimensionMismatch)
    C = Operator(domain, codomain, Matrix{eltype(T)}(undef, dimension(codomain), dimension(domain)))
    foreach((Cᵢ, Aᵢ) -> Cᵢ.coefficients .= Operator(Cᵢ.domain, Cᵢ.codomain, Aᵢ).coefficients, eachcomponent(C), A)
    return C
end

# components

eachcomponent(A::Operator{CartesianSpace{T},CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SingleSpace},N₂,S<:NTuple{N₂,SingleSpace}} =
    (@inbounds(component(A, i, j)) for i ∈ Base.OneTo(N₂), j ∈ Base.OneTo(N₁))

eachcomponent(A::Operator{CartesianSpace{T},<:SingleSpace}) where {N,T<:NTuple{N,SingleSpace}} =
    (@inbounds(component(A, j)) for i ∈ Base.OneTo(1), j ∈ Base.OneTo(N))

eachcomponent(A::Operator{<:SingleSpace,CartesianSpace{T}}) where {N,T<:NTuple{N,SingleSpace}} =
    (@inbounds(component(A, i)) for i ∈ Base.OneTo(N), j ∈ Base.OneTo(1))

Base.eachcol(A::Operator{CartesianSpace{T},CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SingleSpace},N₂,S<:NTuple{N₂,SingleSpace}} =
    (@inbounds(components(A, :, j)) for j ∈ Base.OneTo(N₁))

Base.eachrow(A::Operator{CartesianSpace{T},CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SingleSpace},N₂,S<:NTuple{N₂,SingleSpace}} =
    (@inbounds(components(A, i, :)) for i ∈ Base.OneTo(N₂))

Base.@propagate_inbounds function component(A::Operator{CartesianSpace{T},CartesianSpace{S}}, i::Int, j::Int) where {N₁,T<:NTuple{N₁,SingleSpace},N₂,S<:NTuple{N₂,SingleSpace}}
    @boundscheck(i < 1 || N₂ < i || j < 1 || N₁ < j && throw(BoundsError((A.codomain, A.domain), (i, j))))
    domain, codomain = A.domain[j], A.codomain[i]
    skip₁ = i == 1 ? 0 : mapreduce(k -> dimension(A.codomain[k]), +, 1:i-1)
    skip₂ = j == 1 ? 0 : mapreduce(k -> dimension(A.domain[k]), +, 1:j-1)
    return Operator(domain, codomain, view(A.coefficients, 1+skip₁:dimension(codomain)+skip₁, 1+skip₂:dimension(domain)+skip₂))
end

Base.@propagate_inbounds function components(A::Operator{CartesianSpace{T},<:CartesianSpace}, ::Colon, j::Int) where {N,T<:NTuple{N,SingleSpace}}
    @boundscheck(j < 1 || N < j && throw(BoundsError(A.domain, j)))
    domain = A.domain[j]
    skip = j == 1 ? 0 : mapreduce(k -> dimension(A.domain[k]), +, 1:j-1)
    return Operator(domain, A.codomain, view(A.coefficients, :, 1+skip:dimension(domain)+skip))
end

Base.@propagate_inbounds function components(A::Operator{<:CartesianSpace,CartesianSpace{T}}, i::Int, ::Colon) where {N,T<:NTuple{N,SingleSpace}}
    @boundscheck(i < 1 || N < i && throw(BoundsError(A.codomain, i)))
    codomain = A.codomain[i]
    skip = i == 1 ? 0 : mapreduce(k -> dimension(A.codomain[k]), +, 1:i-1)
    return Operator(A.domain, codomain, view(A.coefficients, 1+skip:dimension(codomain)+skip, :))
end

Base.@propagate_inbounds components(A::Operator{<:CartesianSpace,<:CartesianSpace}, ::Colon, ::Colon) =
    Operator(A.domain, A.codomain, view(A.coefficients, :, :))

Base.@propagate_inbounds function component(A::Operator{CartesianSpace{T},<:SingleSpace}, j::Int) where {N,T<:NTuple{N,SingleSpace}}
    @boundscheck(j < 1 || N < j && throw(BoundsError(A.domain, j)))
    domain = A.domain[j]
    skip = j == 1 ? 0 : mapreduce(k -> dimension(A.domain[k]), +, 1:j-1)
    return Operator(domain, A.codomain, view(A.coefficients, :, 1+skip:dimension(domain)+skip))
end

Base.@propagate_inbounds components(A::Operator{<:CartesianSpace,<:SingleSpace}, ::Colon) =
    Operator(A.domain, A.codomain, view(A.coefficients, :, :))

Base.@propagate_inbounds function component(A::Operator{<:SingleSpace,CartesianSpace{T}}, i::Int) where {N,T<:NTuple{N,SingleSpace}}
    @boundscheck(i < 1 || N < i && throw(BoundsError(A.codomain, i)))
    codomain = A.codomain[i]
    skip = i == 1 ? 0 : mapreduce(k -> dimension(A.codomain[k]), +, 1:i-1)
    return Operator(A.domain, codomain, view(A.coefficients, 1+skip:dimension(codomain)+skip, :))
end

Base.@propagate_inbounds components(A::Operator{<:SingleSpace,<:CartesianSpace}, ::Colon) =
    Operator(A.domain, A.codomain, view(A.coefficients, :, :))
