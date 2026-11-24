"""
    Integral{T<:Union{Nothing,Evaluation,Tuple{Vararg{Union{Nothing,Evaluation}}}}} <: AbstractLinearOperator

Generic integral operator.

Field:
- `condition :: T`

Constructors:
- `Integral()`: equivalent to `Integral(Evaluation(nothing))`
- `Integral(::Union{Nothing,Evaluation})`
- `Integral(::Tuple{Vararg{Union{Nothing,Evaluation}}})`
- `Integral(condition::Union{Nothing,Evaluation}...)`: equivalent to `Integral(condition)`

# Examples

```jldoctest
julia> Integral()
Integral{Evaluation{Nothing}}(Evaluation{Nothing}(nothing))

julia> Integral(Evaluation(-1.0))
Integral{Evaluation{Float64}}(Evaluation{Float64}(-1.0))

julia> Integral(Evaluation(-1.0), nothing)
Integral{Tuple{Evaluation{Float64}, Nothing}}((Evaluation{Float64}(-1.0), nothing))
```
"""
struct Integral{T<:Union{Nothing,Evaluation{<:Union{Nothing,Number}},Tuple{Vararg{Union{Nothing,Evaluation{<:Union{Nothing,Number}}}}}}} <: AbstractLinearOperator
    condition :: T
    Integral{T}(condition::T) where {T<:Union{Nothing,Evaluation{<:Union{Nothing,Number}},Tuple{Vararg{Union{Nothing,Evaluation{<:Union{Nothing,Number}}}}}}} = new{T}(condition)
    Integral{Tuple{}}(::Tuple{}) = throw(ArgumentError("Integral is only defined for at least one Evaluation or Nothing"))
end

Integral(condition::T) where {T<:Union{Nothing,Evaluation{<:Union{Nothing,Number}}}} = Integral{T}(condition)
Integral(condition::T) where {T<:Tuple{Vararg{Union{Nothing,Evaluation{<:Union{Nothing,Number}}}}}} = Integral{T}(condition)
Integral(condition::Union{Nothing,Evaluation{<:Union{Nothing,Number}}}...) = Integral(condition)
Integral() = Integral(Evaluation(nothing))

condition(ℐ::Integral) = ℐ.condition

order(ℐ::Integral) = _integral_order(condition(ℐ))

_integral_order(::Nothing) = 0
_integral_order(::Evaluation) = 1
_integral_order(condition::Tuple) = map(_integral_order, condition)

_integral_value(::Nothing) = nothing
_integral_value(condition::Evaluation) = value(condition)
_integral_value(condition::Tuple) = map(_integral_value, condition)

# Tensor space

function domain(I::Integral{<:NTuple{N,Union{Nothing,Evaluation{<:Union{Nothing,Number}}}}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    s_out = map((xᵢ, sᵢ) -> domain(Integral(xᵢ), sᵢ), condition(I), spaces(s))
    any(sᵢ -> sᵢ isa UndefSpace, s_out) && return UndefSpace()
    return TensorSpace(s_out)
end

codomain(ℱ::Integral{<:NTuple{N,Union{Nothing,Evaluation{<:Union{Nothing,Number}}}}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((xᵢ, sᵢ) -> codomain(Integral(xᵢ), sᵢ), condition(ℱ), spaces(s)))

_coeftype(ℱ::Integral{<:NTuple{N,Union{Nothing,Evaluation{<:Union{Nothing,Number}}}}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    @inbounds promote_type(_coeftype(Integral(condition(ℱ)[1]), s[1], T), _coeftype(Integral(Base.tail(condition(ℱ))), Base.tail(s), T))
_coeftype(ℱ::Integral{<:Tuple{Union{Nothing,Evaluation{<:Union{Nothing,Number}}}}}, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Integral(condition(ℱ)[1]), s[1], T)

getcoefficient(ℱ::Integral{<:NTuple{N,Union{Nothing,Evaluation{<:Union{Nothing,Number}}}}}, (codom, i)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, (dom, j)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, ::Type{T}) where {N,T} =
    @inbounds getcoefficient(Integral(condition(ℱ)[1]), (codom[1], i[1]), (dom[1], j[1]), T) * getcoefficient(Integral(Base.tail(condition(ℱ))), (Base.tail(codom), Base.tail(i)), (Base.tail(dom), Base.tail(j)), T)
getcoefficient(ℱ::Integral{<:Tuple{Union{Nothing,Evaluation{<:Union{Nothing,Number}}}}}, (codom, i)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, (dom, j)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, ::Type{T}) where {T} =
    @inbounds getcoefficient(Integral(condition(ℱ)[1]), (codom[1], i[1]), (dom[1], j[1]), T)

# No integration

domain(::Integral{Nothing}, s::BaseSpace) = s

codomain(::Integral{Nothing}, s::BaseSpace) = s

_coeftype(::Integral{Nothing}, ::BaseSpace, ::Type{T}) where {T} = T

getcoefficient(::Integral{Nothing}, (codom, i)::Tuple{BaseSpace,Integer}, (dom, j)::Tuple{BaseSpace,Integer}, ::Type{T}) where {T} =
    ifelse(i == j, one(T), zero(T))

# Taylor

domain(::Integral{Evaluation{Nothing}}, s::Taylor) = codomain(Derivative(1), s)

codomain(::Integral{<:Evaluation}, s::Taylor) = Taylor(order(s)+1)

_coeftype(::Integral{Evaluation{Nothing}}, ::Taylor, ::Type{T}) where {T} = float(T)

function getcoefficient(::Integral{Evaluation{Nothing}}, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}, ::Type{T}) where {T}
    i != j+1 && return zero(T)
    return convert(T, inv(exact(j+1) * one(real(T))))
end

# Fourier

domain(::Integral{Evaluation{Nothing}}, s::Fourier) = s

codomain(::Integral{<:Evaluation}, s::Fourier) = s

_coeftype(::Integral{Evaluation{Nothing}}, ::Fourier{T}, ::Type{S}) where {T,S} = complex(promote_type(float(T), float(S)))

function getcoefficient(::Integral{Evaluation{Nothing}}, (codom, i)::Tuple{Fourier,Integer}, (dom, j)::Tuple{Fourier,Integer}, ::Type{T}) where {T}
    (i != j) | (j == 0) && return zero(T)
    ω⁻¹j⁻¹ = inv(one(real(T)) * frequency(dom) * exact(j))
    return convert(T, complex(zero(ω⁻¹j⁻¹), -ω⁻¹j⁻¹))
end

# Chebyshev

domain(::Integral{Evaluation{Nothing}}, s::Chebyshev) = Chebyshev(order(s)+1) # the row i reads the columns i-1 and i+1

codomain(::Integral{<:Evaluation}, s::Chebyshev) = Chebyshev(order(s)+1)

_coeftype(::Integral{Evaluation{Nothing}}, ::Chebyshev, ::Type{T}) where {T} = float(T)

function getcoefficient(::Integral{Evaluation{Nothing}}, (codom, i)::Tuple{Chebyshev,Integer}, (dom, j)::Tuple{Chebyshev,Integer}, ::Type{T}) where {T}
    if i == 0
        return zero(T)
    elseif i+1 == j
        return convert(T, -one(T) / exact(2i))
    elseif i == j+1
        return convert(T, one(T) / exact(2i))
    else
        return zero(T)
    end
end

# Condition: the coefficient of the constants is such that the integral vanishes at the value of the condition

_constant_index(::Taylor) = 1
_constant_index(s::Fourier) = order(s) + 1
_constant_index(::Chebyshev) = 1

for (S, Seq) ∈ ((:Taylor, :(Sequence{Taylor})), (:Fourier, :(Sequence{<:Fourier})), (:Chebyshev, :(Sequence{Chebyshev})))
    @eval begin
        domain(::Integral{<:Evaluation{<:Number}}, ::$S) = UndefSpace()

        _coeftype(ℐ::Integral{<:Evaluation{<:Number}}, s::$S, ::Type{T}) where {T} =
            _coeftype(condition(ℐ), codomain(ℐ, s), _coeftype(Integral(), s, T))

        function getcoefficient(ℐ::Integral{<:Evaluation{<:Number}}, (codom, i)::Tuple{$S,Integer}, (dom, j)::Tuple{$S,Integer}, ::Type{T}) where {T}
            ℐ₀ = Integral()
            i == 0 || return getcoefficient(ℐ₀, (codom, i), (dom, j), T)
            ℰ = condition(ℐ)
            s = codomain(ℐ₀, dom)
            s₀ = codomain(ℰ, s)
            x = zero(T)
            for k ∈ indices(s)
                x -= getcoefficient(ℰ, (s₀, 0), (s, k), T) * getcoefficient(ℐ₀, (s, k), (dom, j), T)
            end
            return x
        end

        function _apply!(c::$Seq, ℐ::Integral{<:Evaluation{<:Number}}, a)
            _apply!(c, Integral(), a)
            @inbounds c[0] = -evaluate(c, value(condition(ℐ)))
            return c
        end

        function _apply!(C::AbstractArray, ℐ::Integral{<:Evaluation{<:Number}}, space::$S, d::Val{D}, A) where {D}
            _apply!(C, Integral(), space, d, A)
            codom = codomain(ℐ, space)
            @inbounds selectdim(C, D, _constant_index(codom)) .= .-_apply(condition(ℐ), codom, d, C)
            return C
        end
    end
end

# Symmetric space

function domain(I::Integral{<:Evaluation}, s::SymmetricSpace)
    V = domain(I, desymmetrize(s))
    V isa UndefSpace && return UndefSpace()
    G = unsafe_group!(Set(_groupelem_integral(g, desymmetrize(s)) for g ∈ elements(symmetry(s))))
    return SymmetricSpace(V, G)
end

function codomain(I::Integral{Evaluation{Nothing}}, s::SymmetricSpace)
    V = codomain(I, desymmetrize(s))
    G = unsafe_group!(Set(_groupelem_integral(g, desymmetrize(s)) for g ∈ elements(symmetry(s))))
    return SymmetricSpace(V, G)
end
# the condition adds a constant: only the symmetries keeping the constant coefficient remain
codomain(::Integral{<:Evaluation{<:Number}}, s::SymmetricSpace) =
    _compatible_space_with_constant_index(codomain(Integral(), s))

function _groupelem_integral(g::GroupElement, ::Fourier)
    c = g.lattice_aut.matrix[1]
    new_va = Cocycle(g.cocycle.angle + (c == -1), # the sign c = ±1 adds a half turn when negative
                     g.cocycle.phase)
    return GroupElement(g.lattice_aut, new_va)
end

_coeftype(I::Integral{<:Evaluation}, s::SymmetricSpace, ::Type{T}) where {T} =
    _symtype(symmetry(s), _coeftype(I, desymmetrize(s), T))

domain(::Integral{Nothing}, s::SymmetricSpace{<:BaseSpace}) = s

codomain(::Integral{Nothing}, s::SymmetricSpace{<:BaseSpace}) = s

_coeftype(::Integral{Nothing}, ::SymmetricSpace{<:BaseSpace}, ::Type{T}) where {T} = T

function _apply!(c::Sequence{<:SymmetricSpace{<:Fourier}}, ℐ::Integral, a::Sequence{<:SymmetricSpace{<:Fourier}})
    @inbounds for k ∈ indices(space(c))
        c[k] = zero(eltype(c))
        for l ∈ indices(space(a))
            c[k] += _sym_getcoefficient(ℐ, space(a), space(c), k, l, eltype(c)) * a[l]
        end
    end
    return c
end



# action

Base.:*(ℐ::Integral, a::AbstractSequence) = integrate(a, condition(ℐ))

function integrate(a::Sequence, ℰ::Union{Nothing,Evaluation,Tuple{Vararg{Union{Nothing,Evaluation}}}}=Evaluation(nothing))
    ℐ = Integral(ℰ)
    space_a = space(a)
    new_space = codomain(ℐ, space_a)
    CoefType = _coeftype(ℐ, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, ℐ, a)
    return c
end

function integrate!(c::Sequence, a::Sequence, ℰ::Union{Nothing,Evaluation,Tuple{Vararg{Union{Nothing,Evaluation}}}}=Evaluation(nothing))
    ℐ = Integral(ℰ)
    space_c = space(c)
    new_space = codomain(ℐ, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $ℐ(a) has space $new_space"))
    _apply!(c, ℐ, a)
    return c
end

# Tensor space

function _apply!(c::Sequence{<:TensorSpace}, ℱ::Integral, a)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
    _apply!(C, ℱ, space_a, A)
    return c
end

_apply!(C, ℱ::Integral, space::TensorSpace, A) =
    @inbounds _apply!(C, Integral(condition(ℱ)[1]), space[1], Val(1), _apply(Integral(Base.tail(condition(ℱ))), Base.tail(space), A))
_apply!(C, ℱ::Integral, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply!(C, Integral(condition(ℱ)[1]), space[1], Val(1), A)

_apply(ℱ::Integral, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply(Integral(condition(ℱ)[1]), space[1], Val(N₂-N₁+1), _apply(Integral(Base.tail(condition(ℱ))), Base.tail(space), A))
_apply(ℱ::Integral, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray{T,N}) where {T,N} =
    @inbounds _apply(Integral(condition(ℱ)[1]), space[1], Val(N), A)

# along a direction other than the first, the result is allocated
function _apply(ℐ::Integral, space::BaseSpace, d::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    C = Array{_coeftype(ℐ, space, T),N}(undef, ntuple(i -> ifelse(i == D, dimension(codomain(ℐ, space)), size(A, i)), Val(N)))
    return _apply!(C, ℐ, space, d, A)
end

# No integration

function _apply!(c::Sequence{<:BaseSpace}, ::Integral{Nothing}, a)
    coefficients(c) .= coefficients(a)
    return c
end

function _apply!(C::AbstractArray, ::Integral{Nothing}, ::BaseSpace, ::Val, A)
    C .= A
    return C
end

_apply(::Integral{Nothing}, ::BaseSpace, ::Val, A::AbstractArray) = A

# Taylor

function _apply!(c::Sequence{Taylor}, ::Integral{Evaluation{Nothing}}, a)
    @inbounds c[0] = zero(eltype(c))
    @inbounds for i ∈ 0:order(a)
        c[i+1] = a[i] / exact(i+1)
    end
    return c
end

function _apply!(C::AbstractArray{T}, ::Integral{Evaluation{Nothing}}, space::Taylor, ::Val{D}, A) where {T,D}
    ord = order(space)
    @inbounds selectdim(C, D, 1) .= zero(T)
    @inbounds for i ∈ 0:ord
        selectdim(C, D, i+2) .= selectdim(A, D, i+1) ./ exact(i+1)
    end
    return C
end

# Fourier

function _apply!(c::Sequence{<:Fourier}, ::Integral{Evaluation{Nothing}}, a)
    ω = one(real(eltype(a)))*frequency(a)
    @inbounds c[0] = zero(eltype(c))
    @inbounds for j ∈ 1:order(c)
        ω⁻¹j⁻¹ = inv(ω * exact(j))
        aⱼ = a[j]
        a₋ⱼ = a[-j]
        c[j] = complex(ω⁻¹j⁻¹ * imag(aⱼ), -ω⁻¹j⁻¹ * real(aⱼ))
        c[-j] = complex(-ω⁻¹j⁻¹ * imag(a₋ⱼ), ω⁻¹j⁻¹ * real(a₋ⱼ))
    end
    return c
end

function _apply!(C::AbstractArray{T}, ::Integral{Evaluation{Nothing}}, space::Fourier, ::Val{D}, A) where {T,D}
    ord = order(space)
    ω = one(real(eltype(A)))*frequency(space)
    @inbounds selectdim(C, D, ord+1) .= zero(T)
    @inbounds for j ∈ 1:ord
        ω⁻¹j⁻¹ = inv(ω * exact(j))
        Aⱼ = selectdim(A, D, ord+1+j)
        A₋ⱼ = selectdim(A, D, ord+1-j)
        selectdim(C, D, ord+1+j) .= complex.(ω⁻¹j⁻¹ .* imag.(Aⱼ), (-ω⁻¹j⁻¹) .* real.(Aⱼ))
        selectdim(C, D, ord+1-j) .= complex.((-ω⁻¹j⁻¹) .* imag.(A₋ⱼ), ω⁻¹j⁻¹ .* real.(A₋ⱼ))
    end
    return C
end

# Chebyshev

function _apply!(c::Sequence{Chebyshev}, ::Integral{Evaluation{Nothing}}, a)
    order_a = order(a)
    @inbounds c[0] = zero(eltype(c))
    if order_a == 0
        @inbounds c[1] = a[0] / exact(2)
    elseif order_a == 1
        @inbounds c[1] = a[0] / exact(2)
        @inbounds c[2] = a[1] / exact(4)
    else
        @inbounds c[1] = (a[0] - a[2]) / exact(2)
        @inbounds for i ∈ 2:order_a-1
            c[i] = (a[i-1] - a[i+1]) / exact(2i)
        end
        @inbounds c[order_a] = a[order_a-1] / exact(2order_a)
        @inbounds c[order_a+1] = a[order_a] / exact(2(order_a+1))
    end
    return c
end

function _apply!(C::AbstractArray{T}, ::Integral{Evaluation{Nothing}}, space::Chebyshev, ::Val{D}, A) where {T,D}
    ord = order(space)
    @inbounds C₀ = selectdim(C, D, 1)
    @inbounds C₁ = selectdim(C, D, 2)
    @inbounds A₀ = selectdim(A, D, 1)
    C₀ .= zero(T)
    if ord == 0
        C₁ .= A₀ ./ exact(2)
    elseif ord == 1
        C₁ .= A₀ ./ exact(2)
        @inbounds selectdim(C, D, 3) .= selectdim(A, D, 2) ./ exact(4)
    else
        @inbounds C₁ .= (A₀ .- selectdim(A, D, 3)) ./ exact(2)
        @inbounds for i ∈ 2:ord-1
            selectdim(C, D, i+1) .= (selectdim(A, D, i) .- selectdim(A, D, i+2)) ./ exact(2i)
        end
        @inbounds selectdim(C, D, ord+1) .= selectdim(A, D, ord) ./ exact(2ord)
        @inbounds selectdim(C, D, ord+2) .= selectdim(A, D, ord+1) ./ exact(2(ord+1))
    end
    return C
end

#

function integrate(a::InfiniteSequence, ℰ::Union{Nothing,Evaluation,Tuple{Vararg{Union{Nothing,Evaluation}}}}=Evaluation(nothing))
    c = integrate(sequence(a), ℰ)
    X = banachspace(a)
    s = space(a)
    fe = finite_error(a)
    te = tail_error(a)
    to = total_error(a)
    _safe_iszero(to) || _check_domain(X, s, _integral_value(ℰ))
    new_finite = (_safe_iszero(fe) ? zero(fe) : _integral_finite_error(X, s, ℰ) * fe) +
                 (_safe_iszero(te) ? zero(te) : _integral_cross_error(X, s, ℰ)  * te)
    new_tail   = _safe_iszero(te) ? te : _integral_tail_error(X, s, ℰ) * te
    new_total  = _safe_iszero(to) ? to : _integral_total_error(X, s, ℰ) * to
    # the constructor already stores min(new_finite + new_tail, new_total) as the total error
    return InfiniteSequence(c, new_finite, new_tail, new_total, X)
end

#-

#= A condition fills the coefficient of the constants with -(∫ a)(x): the column j has the entry
   -(∫ eⱼ)(x), of norm |(∫ eⱼ)(x)| / wⱼ relative to eⱼ. This row belongs to the finite part of the
   image: the head columns j ≤ N add to the finite factor, the tail columns to the cross factor. =#

# Taylor: (∫ eⱼ)(x) = xʲ⁺¹/(j+1) and wⱼ = νʲ, decreasing in j for |x| ≤ ν

_integral_condition_finite_error(X::Ell1{<:GeometricWeight}, ::Taylor, x::Number) = abs(x) * one(rate(X))

function _integral_condition_tail_error(X::Ell1{<:GeometricWeight}, s::Taylor, x::Number)
    N = order(s)
    return abs(x) * (abs(x) / rate(X)) ^ exact(N + 1) / exact(N + 2)
end

# Fourier: (∫ eⱼ)(x) = e^{iωjx}/(iωj) for j ≠ 0 and wⱼ = ν^|j|, decreasing in |j| for |ω Im(x)| ≤ log(ν)

function _integral_condition_finite_error(X::Ell1{<:GeometricWeight}, s::Fourier, x::Number)
    ω = abs(frequency(s)) * one(rate(X))
    order(s) == 0 && return zero(inv(ω))
    return exp(ω * abs(imag(x))) / (ω * rate(X))
end

function _integral_condition_tail_error(X::Ell1{<:GeometricWeight}, s::Fourier, x::Number)
    ω = abs(frequency(s)) * one(rate(X))
    N = order(s)
    return (exp(ω * abs(imag(x))) / rate(X)) ^ exact(N + 1) / (ω * exact(N + 1))
end

#= Chebyshev: ∫ e₀ = T₁, ∫ e₁ = T₂/2 and ∫ eⱼ = Tⱼ₊₁/(j+1) - Tⱼ₋₁/(j-1), with w₀ = 1 and wⱼ = 2νʲ.
   On [-1, 1], |Tₖ(x)| ≤ 1 and Tₖ(±1) = (±1)ᵏ, so that |(∫ eⱼ)(±1)| = 2/(j²-1); on the Bernstein ellipse,
   |Tₖ(x)| ≤ νᵏ. The norm of the column j decreases in j ≥ 2. =#

_integral_condition_finite_error(X::Ell1{<:GeometricWeight}, s::Chebyshev, x::Number) =
    maximum(j -> _integral_condition_column(X, x, j), 0:min(order(s), 2))

_integral_condition_tail_error(X::Ell1{<:GeometricWeight}, s::Chebyshev, x::Number) =
    maximum(j -> _integral_condition_column(X, x, j), order(s)+1:max(order(s)+1, 2))

function _integral_condition_column(X::Ell1{<:GeometricWeight}, x::Number, j::Int)
    ν = rate(X)
    isreal_x = _safe_iszero(imag(x)) && sup(abs(x)) ≤ 1
    if j == 0
        return isreal_x ? abs(x) * one(ν) : ν
    elseif j == 1
        return (isreal_x ? one(ν) : ν ^ exact(2)) / (exact(4) * ν)
    elseif isreal_x
        _safe_isone(abs(x)) && return inv(exact(j^2 - 1) * ν ^ exact(j))
        return (inv(exact(j + 1)) + inv(exact(j - 1))) / (exact(2) * ν ^ exact(j))
    else
        return ν / exact(2(j + 1)) + inv(exact(2(j - 1)) * ν)
    end
end

#-

function _integral_tail_error(X::Ell1{<:NTuple{N,Weight}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ℰ::NTuple{N,Union{Nothing,Evaluation}}) where {N}
    tails   = map((wᵢ, sᵢ, ℰᵢ) -> _integral_tail_error(Ell1(wᵢ), sᵢ, ℰᵢ), weight(X), spaces(s), ℰ)
    crosses = map((wᵢ, sᵢ, ℰᵢ) -> _integral_cross_error(Ell1(wᵢ), sᵢ, ℰᵢ), weight(X), spaces(s), ℰ)
    totals  = map((wᵢ, sᵢ, ℰᵢ) -> _integral_total_error(Ell1(wᵢ), sᵢ, ℰᵢ), weight(X), spaces(s), ℰ)
    return _box_complement_factor(tails .+ crosses, totals)
end

_integral_tail_error(X::Ell1{<:GeometricWeight}, ::BaseSpace, ::Nothing) = one(rate(X))

_integral_tail_error(X::Ell1{<:GeometricWeight}, s::BaseSpace, ::Evaluation{<:Number}) =
    _integral_tail_error(X, s, Evaluation(nothing)) # the condition fills a row of the finite part

_integral_tail_error(X::Ell1{<:GeometricWeight}, s::Taylor, ::Evaluation{Nothing}) =
    rate(X) / exact(order(s) + 2)

function _integral_tail_error(X::Ell1{<:GeometricWeight}, s::Fourier, ::Evaluation{Nothing})
    ω = abs(frequency(s)) * one(rate(X))
    return inv(ω * exact(order(s) + 1))
end

function _integral_tail_error(X::Ell1{<:GeometricWeight}, s::Chebyshev, ::Evaluation{Nothing})
    ν = rate(X)
    N = exact(order(s))
    return max(ν / (exact(2) * (N + exact(2))),
               ν / (exact(2) * (N + exact(4))) + inv(exact(2) * ν * (N + exact(2))))
end

#-

_integral_finite_error(X::Ell1{<:NTuple{N,Weight}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ℰ::NTuple{N,Union{Nothing,Evaluation}}) where {N} =
    mapreduce((wᵢ, sᵢ, ℰᵢ) -> _integral_finite_error(Ell1(wᵢ), sᵢ, ℰᵢ), *, weight(X), spaces(s), ℰ)

_integral_finite_error(X::Ell1{<:GeometricWeight}, ::BaseSpace, ::Nothing) = one(rate(X))

_integral_finite_error(X::Ell1{<:GeometricWeight}, s::BaseSpace, ℰ::Evaluation{<:Number}) =
    _integral_finite_error(X, s, Evaluation(nothing)) + _integral_condition_finite_error(X, s, value(ℰ))

_integral_finite_error(X::Ell1{<:GeometricWeight}, ::Taylor, ::Evaluation{Nothing}) = rate(X)

function _integral_finite_error(X::Ell1{<:GeometricWeight}, s::Fourier, ::Evaluation{Nothing})
    ω = abs(frequency(s)) * one(rate(X))
    order(s) == 0 && return zero(inv(ω))
    return inv(ω)
end

_integral_finite_error(X::Ell1{<:GeometricWeight}, ::Chebyshev, ::Evaluation{Nothing}) = rate(X) # attained at j = 0

#-

_integral_cross_error(X::Ell1{<:NTuple{N,Weight}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ℰ::NTuple{N,Union{Nothing,Evaluation}}) where {N} =
    _box_complement_factor(map((wᵢ, sᵢ, ℰᵢ) -> _integral_cross_error(Ell1(wᵢ), sᵢ, ℰᵢ), weight(X), spaces(s), ℰ),
                           map((wᵢ, sᵢ, ℰᵢ) -> max(_integral_finite_error(Ell1(wᵢ), sᵢ, ℰᵢ),
                                                   _integral_cross_error(Ell1(wᵢ), sᵢ, ℰᵢ)), weight(X), spaces(s), ℰ))

_integral_cross_error(X::Ell1{<:GeometricWeight}, ::BaseSpace, ::Nothing) = zero(rate(X))

_integral_cross_error(X::Ell1{<:GeometricWeight}, s::BaseSpace, ℰ::Evaluation{<:Number}) =
    _integral_cross_error(X, s, Evaluation(nothing)) + _integral_condition_tail_error(X, s, value(ℰ))

_integral_cross_error(X::Ell1{<:GeometricWeight}, ::Taylor, ::Evaluation{Nothing}) = zero(rate(X))
_integral_cross_error(X::Ell1{<:GeometricWeight}, ::Fourier, ::Evaluation{Nothing}) = zero(rate(X))

function _integral_cross_error(X::Ell1{<:GeometricWeight}, s::Chebyshev, ::Evaluation{Nothing})
    ν = rate(X)
    #= A tail column j > N of ∫ hits the rows j-1 and j+1, of which the head Chebyshev(N+1) keeps
       row j-1 when 2 ≤ j ≤ N+2; the resulting norm 1/(2ν(j-1)) decreases in j, so it is attained
       at the first tail column with a head row. For N = 0 that is j = 2, the column j = 1 having
       no head row. =#
    j = exact(max(order(s) + 1, 2))
    return inv(exact(2) * ν * (j - exact(1)))
end

#-

_integral_total_error(X::Ell1{<:NTuple{N,Weight}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ℰ::NTuple{N,Union{Nothing,Evaluation}}) where {N} =
    mapreduce((wᵢ, sᵢ, ℰᵢ) -> _integral_total_error(Ell1(wᵢ), sᵢ, ℰᵢ), *, weight(X), spaces(s), ℰ)

_integral_total_error(X::Ell1{<:GeometricWeight}, ::BaseSpace, ::Nothing) = one(rate(X))

_integral_total_error(X::Ell1{<:GeometricWeight}, s::BaseSpace, ℰ::Evaluation{<:Number}) =
    _integral_total_error(X, s, Evaluation(nothing)) +
        max(_integral_condition_finite_error(X, s, value(ℰ)), _integral_condition_tail_error(X, s, value(ℰ)))

_integral_total_error(X::Ell1{<:GeometricWeight}, ::Taylor, ::Evaluation{Nothing}) = rate(X)

function _integral_total_error(X::Ell1{<:GeometricWeight}, s::Fourier, ::Evaluation{Nothing})
    ω = abs(frequency(s)) * one(rate(X))
    return inv(ω)
end

_integral_total_error(X::Ell1{<:GeometricWeight}, ::Chebyshev, ::Evaluation{Nothing}) = rate(X)
