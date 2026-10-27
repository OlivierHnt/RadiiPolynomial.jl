"""
    Shift{T<:Union{Number,Tuple{Vararg{Number}}}} <: AbstractLinearOperator

Generic shift operator.

On a [`Fourier`](@ref) space, `Shift(τ)` shifts by ``\\tau`` periods and
multiplies the ``k``-th coefficient by ``e^{2i\\pi\\tau k}``.

Field:
- `value :: T`

Constructors:
- `Shift(::Number)`
- `Shift(::Tuple{Vararg{Number}})`
- `Shift(value::Number...)`: equivalent to `Shift(value)`

# Examples

```jldoctest
julia> Shift(1.0)
Shift{Float64}(1.0)

julia> Shift(1.0, 2.0)
Shift{Tuple{Float64, Float64}}((1.0, 2.0))
```
"""
struct Shift{T<:Union{Number,Tuple{Vararg{Number}}}} <: AbstractLinearOperator
    value :: T
    Shift{T}(value::T) where {T<:Union{Number,Tuple{Vararg{Number}}}} = new{T}(value)
    Shift{Tuple{}}(::Tuple{}) = throw(ArgumentError("Shift is only defined for at least one Number"))
end

Shift(value::T) where {T<:Number} = Shift{T}(value)
Shift(value::T) where {T<:Tuple{Vararg{Number}}} = Shift{T}(value)
Shift(value::Number...) = Shift(value)

value(𝒮::Shift) = 𝒮.value

Base.:*(𝒮₁::Shift{<:Number}, 𝒮₂::Shift{<:Number}) = Shift(value(𝒮₁) + value(𝒮₂))
Base.:*(𝒮₁::Shift{<:NTuple{N,Number}}, 𝒮₂::Shift{<:NTuple{N,Number}}) where {N} = Shift(map(+, value(𝒮₁), value(𝒮₂)))

Base.:^(𝒮::Shift{<:Number}, n::Integer) = Shift(value(𝒮) * exact(n))
Base.:^(𝒮::Shift{<:Tuple{Vararg{Number}}}, n::Integer) = Shift(map(τᵢ -> τᵢ * exact(n), value(𝒮)))
Base.:^(𝒮::Shift{<:NTuple{N,Number}}, n::NTuple{N,Integer}) where {N} = Shift(map((τᵢ, nᵢ) -> τᵢ * exact(nᵢ), value(𝒮), n))

# Tensor space

domain(S::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((τᵢ, sᵢ) -> domain(Shift(τᵢ), sᵢ), value(S), spaces(s)))

codomain(𝒮::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((τᵢ, sᵢ) -> codomain(Shift(τᵢ), sᵢ), value(𝒮), spaces(s)))

_coeftype(𝒮::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    @inbounds promote_type(_coeftype(Shift(value(𝒮)[1]), s[1], T), _coeftype(Shift(Base.tail(value(𝒮))), Base.tail(s), T))
_coeftype(𝒮::Shift{<:Tuple{Number}}, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Shift(value(𝒮)[1]), s[1], T)

getcoefficient(𝒮::Shift{<:NTuple{N,Number}}, (codom, i)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, (dom, j)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, ::Type{T}) where {N,T} =
    @inbounds getcoefficient(Shift(value(𝒮)[1]), (codom[1], i[1]), (dom[1], j[1]), T) * getcoefficient(Shift(Base.tail(value(𝒮))), (Base.tail(codom), Base.tail(i)), (Base.tail(dom), Base.tail(j)), T)
getcoefficient(𝒮::Shift{<:Tuple{Number}}, (codom, i)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, (dom, j)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, ::Type{T}) where {T} =
    @inbounds getcoefficient(Shift(value(𝒮)[1]), (codom[1], i[1]), (dom[1], j[1]), T)

# Taylor

domain(::Shift, s::Taylor) = s

codomain(::Shift, s::Taylor) = s

_coeftype(::Shift{T}, ::Taylor, ::Type{S}) where {T,S} = promote_type(T, S)

function getcoefficient(𝒮::Shift, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}, ::Type{T}) where {T}
    τ = value(𝒮)
    if iszero(τ)
        return ifelse(i == j, one(T), zero(T))
    else # TODO: lift restriction
        return throw(DomainError(𝒮, "Shift is only defined for τ = 0 for Taylor spaces"))
    end
end

# Fourier

# the value is measured in periods: the factor at `k` is e^{2iπτk}, an exact root of unity for `Integer` and `Rational` values

domain(::Shift, s::Fourier) = s

codomain(::Shift, s::Fourier) = s

_coeftype(::Shift{T}, ::Fourier, ::Type{S}) where {T,S} = promote_type(typeof(cispi(zero(T))), S)

function getcoefficient(𝒮::Shift, (codom, i)::Tuple{Fourier,Integer}, (dom, j)::Tuple{Fourier,Integer}, ::Type{T}) where {T}
    i == j || return zero(T)
    return convert(T, _period_factor(T, value(𝒮), i))
end
_period_factor(::Type{T}, τ::Union{Integer,Rational}, k::Integer) where {T} = _cispi(T, Rational{Int}(mod(2τ * k, 2)))
_period_factor(::Type{T}, τ::Number, k::Integer) where {T} = cispi(2τ * exact(k))

# Chebyshev

domain(::Shift, s::Chebyshev) = s

codomain(::Shift, s::Chebyshev) = s

_coeftype(::Shift{T}, ::Chebyshev, ::Type{S}) where {T,S} = promote_type(T, S)

function getcoefficient(𝒮::Shift, (codom, i)::Tuple{Chebyshev,Integer}, (dom, j)::Tuple{Chebyshev,Integer}, ::Type{T}) where {T}
    τ = value(𝒮)
    if iszero(τ)
        return ifelse(i == j, one(T), zero(T))
    else # TODO: lift restriction
        return throw(DomainError(𝒮, "Shift is only defined for τ = 0 for Chebyshev spaces"))
    end
end

# Symmetric space

# the shift by `r` periods turns the symmetry a_k = α(k) a_{βk} into c_k = α(k) e^{2iπ⟨r, k - βk⟩} c_{βk},
# which moves the phase of the cocycle by 2(I - βᵀ)r; a Taylor or Chebyshev axis only admits the zero
# shift, so it contributes nothing to `r`
function domain(𝒮::Shift, s::SymmetricSpace)
    r = _period_fraction(𝒮, desymmetrize(s))
    G = unsafe_group!(Set(_groupelem_shift(-r, g) for g ∈ elements(symmetry(s))))
    return SymmetricSpace(domain(𝒮, desymmetrize(s)), G)
end

function codomain(𝒮::Shift, s::SymmetricSpace)
    r = _period_fraction(𝒮, desymmetrize(s))
    G = unsafe_group!(Set(_groupelem_shift(r, g) for g ∈ elements(symmetry(s))))
    return SymmetricSpace(codomain(𝒮, desymmetrize(s)), G)
end

_period_fraction(𝒮::Shift{<:Number}, s::BaseSpace) = StaticArrays.SVector(_period_fraction(value(𝒮), s))
_period_fraction(𝒮::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    StaticArrays.SVector(map(_period_fraction, value(𝒮), spaces(s)))

_period_fraction(τ::Union{Integer,Rational}, ::Fourier) = τ
_period_fraction(τ::ExactReal{<:Union{Integer,Rational}}, ::Fourier) = τ.value
_period_fraction(τ::Interval{<:Rational}, ::Fourier) =
    isthin(τ) ? sup(τ) : throw(DomainError(τ, "Shift value must be a single number of periods for symmetric Fourier spaces"))
_period_fraction(τ::Number, ::Fourier) = throw(DomainError(τ, "Shift value must be an Integer or Rational number of periods for symmetric Fourier spaces"))
_period_fraction(τ::Number, s::Union{Taylor,Chebyshev}) =
    _safe_iszero(τ) ? zero(Rational{Int}) : throw(DomainError(τ, "Shift is only defined for τ = 0 for $(nameof(typeof(s))) spaces"))

function _groupelem_shift(r::StaticArrays.SVector{N,<:Union{Integer,Rational}}, g::GroupElement{N}) where {N}
    new_va = Cocycle(g.cocycle.angle,
                     g.cocycle.phase + 2 * (r - g.lattice_aut.matrix' * r))
    return GroupElement(g.lattice_aut, new_va)
end

_coeftype(𝒮::Shift, s::SymmetricSpace, ::Type{T}) where {T} =
    _coeftype(𝒮, desymmetrize(s), T)



# action

Base.:*(𝒮::Shift, a::AbstractSequence) = shift(a, value(𝒮))

function shift(a::Sequence, τ::Union{Number,Tuple{Vararg{Number}}})
    𝒮 = Shift(τ)
    space_a = space(a)
    new_space = codomain(𝒮, space_a)
    CoefType = _coeftype(𝒮, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, 𝒮, a)
    return c
end

function shift!(c::Sequence, a::Sequence, τ::Union{Number,Tuple{Vararg{Number}}})
    𝒮 = Shift(τ)
    space_c = space(c)
    new_space = codomain(𝒮, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $𝒮(a) has space $new_space"))
    _apply!(c, 𝒮, a)
    return c
end

# Tensor space

function _apply!(c::Sequence{<:TensorSpace}, 𝒮::Shift, a)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
    _apply!(C, 𝒮, space_a, A)
    return c
end

_apply!(C, 𝒮::Shift, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply!(C, Shift(value(𝒮)[1]), space[1], Val(N₂-N₁+1), _apply!(C, Shift(Base.tail(value(𝒮))), Base.tail(space), A))

_apply!(C, 𝒮::Shift, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray) =
    @inbounds _apply!(C, Shift(value(𝒮)[1]), space[1], A)

# Taylor

function _apply!(c::Sequence{Taylor}, 𝒮::Shift, a)
    τ = value(𝒮)
    if iszero(τ)
        coefficients(c) .= coefficients(a)
    else # TODO: lift restriction
        return throw(DomainError(𝒮, "Shift is only defined for τ = 0 for Taylor spaces"))
    end
    return c
end

function _apply!(C, 𝒮::Shift, ::Taylor, ::Val{D}, A) where {D}
    τ = value(𝒮)
    iszero(τ) || return throw(DomainError(𝒮, "Shift is only defined for τ = 0 for Taylor spaces")) # TODO: lift restriction
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Shift, ::Taylor, A) where {T,N}
    τ = value(𝒮)
    if iszero(τ)
        C .= A
    else # TODO: lift restriction
        return throw(DomainError(𝒮, "Shift is only defined for τ = 0 for Taylor spaces"))
    end
    return C
end

# Fourier

function _apply!(c::Sequence{<:Fourier}, 𝒮::Shift, a)
    τ = value(𝒮)
    @inbounds for k ∈ indices(space(a))
        c[k] = _period_factor(eltype(c), τ, k) * a[k]
    end
    return c
end

function _apply!(C, 𝒮::Shift, space::Fourier, ::Val{D}, A) where {D}
    τ = value(𝒮)
    ord = order(space)
    @inbounds for k ∈ -ord:ord
        selectdim(C, D, ord+1+k) .*= _period_factor(eltype(C), τ, k)
    end
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Shift, space::Fourier, A) where {T,N}
    τ = value(𝒮)
    ord = order(space)
    @inbounds for k ∈ -ord:ord
        selectdim(C, N, ord+1+k) .= _period_factor(T, τ, k) .* selectdim(A, N, ord+1+k)
    end
    return C
end

# Chebyshev

function _apply!(c::Sequence{Chebyshev}, 𝒮::Shift, a)
    τ = value(𝒮)
    if iszero(τ)
        coefficients(c) .= coefficients(a)
    else # TODO: lift restriction
        return throw(DomainError(𝒮, "Shift is only defined for τ = 0 for Chebyshev spaces"))
    end
    return c
end

function _apply!(C, 𝒮::Shift, ::Chebyshev, ::Val{D}, A) where {D}
    τ = value(𝒮)
    iszero(τ) || return throw(DomainError(𝒮, "Shift is only defined for τ = 0 for Chebyshev spaces")) # TODO: lift restriction
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Shift, ::Chebyshev, A) where {T,N}
    τ = value(𝒮)
    if iszero(τ)
        C .= A
    else # TODO: lift restriction
        return throw(DomainError(𝒮, "Shift is only defined for τ = 0 for Chebyshev spaces"))
    end
    return C
end
