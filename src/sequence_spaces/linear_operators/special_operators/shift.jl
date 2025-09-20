"""
    Shift{T<:Union{Number,Tuple{Vararg{Number}}}} <: AbstractLinearOperator

Generic shift operator.

Field:
- `value :: T`

Constructors:
- `Shift(::Number)`
- `Shift(::Tuple{Vararg{Number}})`
- `Shift(value::Number...)`: equivalent to `Shift(value)`

See also: [`shift`](@ref), [`shift!`](@ref),
[`project(::Shift, ::VectorSpace, ::VectorSpace)`](@ref) and
[`project!(::LinearOperator, ::Shift)`](@ref).

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

_infer_domain(S::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((τᵢ, sᵢ) -> _infer_domain(Shift(τᵢ), sᵢ), value(S), spaces(s)))
_infer_domain(::Shift, s::Taylor) = s
_infer_domain(::Shift, s::Fourier) = s
_infer_domain(::Shift, s::Chebyshev) = s
_infer_domain(S::Shift, s::CartesianPower) = CartesianPower(_infer_domain(S, space(s)), nspaces(s))
_infer_domain(S::Shift, s::CartesianSpace) = CartesianProduct(map(sᵢ -> _infer_domain(S, sᵢ), spaces(s)))

Base.:*(𝒮₁::Shift{<:Number}, 𝒮₂::Shift{<:Number}) = Shift(value(𝒮₁) + value(𝒮₂))
Base.:*(𝒮₁::Shift{<:NTuple{N,Number}}, 𝒮₂::Shift{<:NTuple{N,Number}}) where {N} = Shift(map(+, value(𝒮₁), value(𝒮₂)))

Base.:^(𝒮::Shift{<:Number}, n::Integer) = Shift(value(𝒮) * exact(n))
Base.:^(𝒮::Shift{<:Tuple{Vararg{Number}}}, n::Integer) = Shift(map(τᵢ -> τᵢ * exact(n), value(𝒮)))
Base.:^(𝒮::Shift{<:NTuple{N,Number}}, n::NTuple{N,Integer}) where {N} = Shift(map((τᵢ, nᵢ) -> τᵢ * exact(nᵢ), value(𝒮), n))

"""
    *(𝒮::Shift, a::AbstractSequence)

Shift `a` by `value(𝒮)`; equivalent to `shift(a, value(𝒮))`.

See also: [`Shift`](@ref), [`shift`](@ref) and [`shift!`](@ref).
"""
Base.:*(𝒮::Shift, a::AbstractSequence) = shift(a, value(𝒮))

"""
    shift(a::Sequence, τ)

Shift `a` by `τ`.

See also: [`shift!`](@ref), [`Shift`](@ref), [`*(::Shift, ::Sequence)`](@ref)
and [`(::Shift)(::Sequence)`](@ref).
"""
function shift(a::Sequence, τ)
    𝒮 = Shift(τ)
    space_a = space(a)
    new_space = codomain(𝒮, space_a)
    CoefType = _coeftype(𝒮, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, 𝒮, a)
    return c
end

"""
    shift!(c::Sequence, a::Sequence, τ)

Shift `a` by `τ`. The result is stored in `c` by overwriting it.

See also: [`shift`](@ref), [`Shift`](@ref), [`*(::Shift, ::Sequence)`](@ref)
and [`(::Shift)(::Sequence)`](@ref).
"""
function shift!(c::Sequence, a::Sequence, τ)
    𝒮 = Shift(τ)
    space_c = space(c)
    new_space = codomain(𝒮, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $𝒮(a) has space $new_space"))
    _apply!(c, 𝒮, a)
    return c
end

"""
    project!(C::LinearOperator, 𝒮::Shift)

Represent `𝒮` as a [`LinearOperator`](@ref) from `domain(C)` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project(::Shift, ::VectorSpace, ::VectorSpace)`](@ref) and
[`Shift`](@ref).
"""
function project!(C::LinearOperator, 𝒮::Shift)
    image_domain = codomain(𝒮, domain(C))
    codomain_C = codomain(C)
    _iscompatible(image_domain, codomain_C) || return throw(ArgumentError("spaces must be compatible: image of domain(C) under $𝒮 is $image_domain, C has codomain $codomain_C"))
    coefficients(C) .= zero(eltype(C))
    _project!(C, 𝒮)
    return C
end

_findposition_nzind_domain(𝒮::Shift, domain, codomain) =
    _findposition(_nzind_domain(𝒮, domain, codomain), domain)

_findposition_nzind_codomain(𝒮::Shift, domain, codomain) =
    _findposition(_nzind_codomain(𝒮, domain, codomain), codomain)

# Sequence spaces

codomain(𝒮::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((τᵢ, sᵢ) -> codomain(Shift(τᵢ), sᵢ), value(𝒮), spaces(s)))

_coeftype(𝒮::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    @inbounds promote_type(_coeftype(Shift(value(𝒮)[1]), s[1], T), _coeftype(Shift(Base.tail(value(𝒮))), Base.tail(s), T))
_coeftype(𝒮::Shift{<:Tuple{Number}}, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Shift(value(𝒮)[1]), s[1], T)

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

for (_f, __f) ∈ ((:_nzind_domain, :__nzind_domain), (:_nzind_codomain, :__nzind_codomain))
    @eval begin
        $_f(𝒮::Shift{<:NTuple{N,Number}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
            TensorIndices($__f(𝒮, domain, codomain))
        $__f(𝒮::Shift, domain::TensorSpace, codomain) =
            @inbounds ($_f(Shift(value(𝒮)[1]), domain[1], codomain[1]), $__f(Shift(Base.tail(value(𝒮))), Base.tail(domain), Base.tail(codomain))...)
        $__f(𝒮::Shift, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain) =
            @inbounds ($_f(Shift(value(𝒮)[1]), domain[1], codomain[1]),)
    end
end

function _project!(C::LinearOperator{<:SequenceSpace,<:SequenceSpace}, 𝒮::Shift)
    domain_C = domain(C)
    codomain_C = codomain(C)
    CoefType = eltype(C)
    @inbounds for (α, β) ∈ zip(_nzind_codomain(𝒮, domain_C, codomain_C), _nzind_domain(𝒮, domain_C, codomain_C))
        C[α,β] = _nzval(𝒮, domain_C, codomain_C, CoefType, α, β)
    end
    return C
end

_nzval(𝒮::Shift{<:NTuple{N,Number}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}, α, β) where {N,T} =
    @inbounds _nzval(Shift(value(𝒮)[1]), domain[1], codomain[1], T, α[1], β[1]) * _nzval(Shift(Base.tail(value(𝒮))), Base.tail(domain), Base.tail(codomain), T, Base.tail(α), Base.tail(β))
_nzval(𝒮::Shift{<:Tuple{Number}}, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}, α, β) where {T} =
    @inbounds _nzval(Shift(value(𝒮)[1]), domain[1], codomain[1], T, α[1], β[1])

# Taylor

codomain(::Shift, s::Taylor) = s

_coeftype(::Shift{T}, ::Taylor, ::Type{S}) where {T,S} = promote_type(T, S)

function _apply!(c::Sequence{Taylor}, 𝒮::Shift, a)
    τ = value(𝒮)
    if iszero(τ)
        coefficients(c) .= coefficients(a)
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C, 𝒮::Shift, space::Taylor, ::Val{D}, A) where {D}
    τ = value(𝒮)
    iszero(τ) || return throw(DomainError) # TODO: lift restriction
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Shift, space::Taylor, A) where {T,N}
    τ = value(𝒮)
    if iszero(τ)
        C .= A
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end

_nzind_domain(::Shift, domain::Taylor, codomain::Taylor) = 0:min(order(domain), order(codomain))
_nzind_codomain(::Shift, domain::Taylor, codomain::Taylor) = 0:min(order(domain), order(codomain))
function _nzval(𝒮::Shift, ::Taylor, ::Taylor, ::Type{T}, i, j) where {T}
    τ = value(𝒮)
    if iszero(τ)
        return one(T)
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

# Fourier

codomain(::Shift, s::Fourier) = s

_coeftype(::Shift{T}, s::Fourier, ::Type{S}) where {T,S} =
    promote_type(typeof(cis(frequency(s)*zero(T))), S)

function _apply!(c::Sequence{<:Fourier}, 𝒮::Shift, a)
    τ = value(𝒮)
    if iszero(τ)
        coefficients(c) .= coefficients(a)
    else
        @inbounds c[0] = a[0]
        eiωτ = cis(frequency(a)*τ)
        eiωτj = one(eiωτ)
        @inbounds for j ∈ 1:order(a)
            eiωτj *= eiωτ
            c[j] = eiωτj * a[j]
            c[-j] = a[-j] / eiωτj
        end
    end
    return c
end

function _apply!(C, 𝒮::Shift, space::Fourier, ::Val{D}, A) where {D}
    τ = value(𝒮)
    if !iszero(τ)
        ord = order(space)
        eiωτ = cis(frequency(space)*τ)
        eiωτj = one(eiωτ)
        @inbounds for j ∈ 1:ord
            eiωτj *= eiωτ
            selectdim(C, D, ord+1+j) .*= eiωτj
            selectdim(C, D, ord+1-j) ./= eiωτj
        end
    end
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Shift, space::Fourier, A) where {T,N}
    τ = value(𝒮)
    if iszero(τ)
        C .= A
    else
        ord = order(space)
        @inbounds selectdim(C, N, ord+1) .= selectdim(A, N, ord+1)
        eiωτ = cis(frequency(space)*τ)
        eiωτj = one(eiωτ)
        @inbounds for j ∈ 1:ord
            eiωτj *= eiωτ
            selectdim(C, N, ord+1+j) .= eiωτj .* selectdim(A, N, ord+1+j)
            selectdim(C, N, ord+1-j) .= selectdim(A, N, ord+1-j) ./ eiωτj
        end
    end
    return C
end

function _nzind_domain(::Shift, domain::Fourier, codomain::Fourier)
    ord = min(order(domain), order(codomain))
    return -ord:ord
end
function _nzind_codomain(::Shift, domain::Fourier, codomain::Fourier)
    ord = min(order(domain), order(codomain))
    return -ord:ord
end
function _nzval(𝒮::Shift, domain::Fourier, ::Fourier, ::Type{T}, i, j) where {T}
    τ = value(𝒮)
    if iszero(τ)
        return one(T)
    else
        return convert(T, cis(frequency(domain) * τ * exact(i)))
    end
end

# Chebyshev

codomain(::Shift, s::Chebyshev) = s

_coeftype(::Shift{T}, ::Chebyshev, ::Type{S}) where {T,S} = promote_type(T, S)

function _apply!(c::Sequence{Chebyshev}, 𝒮::Shift, a)
    τ = value(𝒮)
    if iszero(τ)
        coefficients(c) .= coefficients(a)
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C, 𝒮::Shift, space::Chebyshev, ::Val{D}, A) where {D}
    τ = value(𝒮)
    iszero(τ) || return throw(DomainError) # TODO: lift restriction
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Shift, space::Chebyshev, A) where {T,N}
    τ = value(𝒮)
    if iszero(τ)
        C .= A
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end

_nzind_domain(::Shift, domain::Chebyshev, codomain::Chebyshev) = 0:min(order(domain), order(codomain))
_nzind_codomain(::Shift, domain::Chebyshev, codomain::Chebyshev) = 0:min(order(domain), order(codomain))
function _nzval(𝒮::Shift, ::Chebyshev, ::Chebyshev, ::Type{T}, i, j) where {T}
    τ = value(𝒮)
    if iszero(τ)
        return one(T)
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

# Cartesian spaces

codomain(𝒮::Shift, s::CartesianPower) =
    CartesianPower(codomain(𝒮, space(s)), nspaces(s))

codomain(𝒮::Shift, s::CartesianProduct) =
    CartesianProduct(map(sᵢ -> codomain(𝒮, sᵢ), spaces(s)))

_coeftype(𝒮::Shift, s::CartesianPower, ::Type{T}) where {T} =
    _coeftype(𝒮, space(s), T)

_coeftype(𝒮::Shift, s::CartesianProduct, ::Type{T}) where {T} =
    @inbounds promote_type(_coeftype(𝒮, s[1], T), _coeftype(𝒮, Base.tail(s), T))
_coeftype(𝒮::Shift, s::CartesianProduct{<:Tuple{VectorSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(𝒮, s[1], T)

function _apply!(c::Sequence{<:CartesianPower}, 𝒮::Shift, a)
    @inbounds for i ∈ 1:nspaces(space(c))
        _apply!(component(c, i), 𝒮, component(a, i))
    end
    return c
end
function _apply!(c::Sequence{CartesianProduct{T}}, 𝒮::Shift, a) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds _apply!(component(c, 1), 𝒮, component(a, 1))
    @inbounds _apply!(component(c, 2:N), 𝒮, component(a, 2:N))
    return c
end
function _apply!(c::Sequence{CartesianProduct{T}}, 𝒮::Shift, a) where {T<:Tuple{VectorSpace}}
    @inbounds _apply!(component(c, 1), 𝒮, component(a, 1))
    return c
end

function _findposition_nzind_domain(𝒮::Shift, domain::CartesianSpace, codomain::CartesianSpace)
    u = map((dom, codom) -> _findposition_nzind_domain(𝒮, dom, codom), spaces(domain), spaces(codomain))
    len = sum(length, u)
    v = Vector{Int}(undef, len)
    δ = δδ = 0
    @inbounds for (i, uᵢ) in enumerate(u)
        δ_ = δ
        δ += length(uᵢ)
        view(v, 1+δ_:δ) .= δδ .+ uᵢ
        δδ += dimension(domain[i])
    end
    return v
end

function _findposition_nzind_codomain(𝒮::Shift, domain::CartesianSpace, codomain::CartesianSpace)
    u = map((dom, codom) -> _findposition_nzind_codomain(𝒮, dom, codom), spaces(domain), spaces(codomain))
    len = sum(length, u)
    v = Vector{Int}(undef, len)
    δ = δδ = 0
    @inbounds for (i, uᵢ) in enumerate(u)
        δ_ = δ
        δ += length(uᵢ)
        view(v, 1+δ_:δ) .= δδ .+ uᵢ
        δδ += dimension(codomain[i])
    end
    return v
end

function _project!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, 𝒮::Shift)
    @inbounds for i ∈ 1:nspaces(domain(C))
        _project!(component(C, i, i), 𝒮)
    end
    return C
end
