"""
    Multiplication{T<:Sequence{<:SequenceSpace}} <: AbstractLinearOperator

Multiplication operator associated with a given [`Sequence`](@ref).

Field:
- `sequence :: T`

Constructor:
- `Multiplication(::Sequence{<:SequenceSpace})`

See also: [`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref),
[`project(::Multiplication, ::SequenceSpace, ::SequenceSpace)`](@ref),
[`project!(::LinearOperator{<:SequenceSpace,<:SequenceSpace}, ::Multiplication)`](@ref)
and [`Multiplication`](@ref).
"""
struct Multiplication{T<:Sequence{<:SequenceSpace}} <: AbstractLinearOperator
    sequence :: T
end

sequence(ℳ::Multiplication) = ℳ.sequence

IntervalArithmetic.interval(::Type{T}, ℳ::Multiplication, d::IntervalArithmetic.Decoration = com; format::Symbol = :infsup) where {T} =
    Multiplication(interval(T, sequence(ℳ), d; format = format))
IntervalArithmetic.interval(ℳ::Multiplication, d::IntervalArithmetic.Decoration = com; format::Symbol = :infsup) =
    Multiplication(interval(sequence(ℳ), d; format = format))
IntervalArithmetic.interval(::Type{T}, ℳ::Multiplication, d::AbstractVector{IntervalArithmetic.Decoration}; format::Symbol = :infsup) where {T} =
    Multiplication(interval(T, sequence(ℳ), d; format = format))
IntervalArithmetic.interval(ℳ::Multiplication, d::AbstractVector{IntervalArithmetic.Decoration}; format::Symbol = :infsup) =
    Multiplication(interval(sequence(ℳ), d; format = format))

_infer_domain(M::Multiplication, s::SequenceSpace) = _infer_domain(*, space(sequence(M)), s)
_infer_domain(::typeof(*), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((s₁ᵢ, s₂ᵢ) -> _infer_domain(*, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))
_infer_domain(::typeof(*), ::Taylor, s::Taylor) = s
_infer_domain(::typeof(*), s₁::Fourier, s₂::Fourier) = codomain(*, s₁, s₂)
_infer_domain(::typeof(*), s₁::Chebyshev, s₂::Chebyshev) = codomain(*, s₁, s₂)
_infer_domain(::typeof(*), s₁::CosFourier, s₂::CosFourier) = codomain(*, s₁, s₂)
_infer_domain(::typeof(*), s₁::SinFourier, s₂::SinFourier) = codomain(*, s₁, s₂)
_infer_domain(::typeof(*), s₁::SinFourier, s₂::CosFourier) = codomain(*, s₁, s₂)
_infer_domain(::typeof(*), s₁::CosFourier, s₂::SinFourier) = codomain(*, s₁, s₂)

"""
    *(ℳ::Multiplication, a::Sequence)

Compute the discrete convolution (associated with `space(sequence(ℳ))` and
`space(a)`) of `sequence(ℳ)` and `a`; equivalent to `sequence(ℳ) * a`.

See also: [`Multiplication`](@ref),
[`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref)
and [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref).
"""
Base.:*(ℳ::Multiplication, a::Sequence) = *(sequence(ℳ), a)

Base.:+(ℳ::Multiplication) = Multiplication(+(sequence(ℳ)))
Base.:-(ℳ::Multiplication) = Multiplication(-(sequence(ℳ)))
Base.:^(ℳ::Multiplication, n::Integer) = Multiplication(sequence(ℳ) ^ n)

for f ∈ (:+, :-, :*)
    @eval begin
        Base.$f(ℳ₁::Multiplication, ℳ₂::Multiplication) = Multiplication($f(sequence(ℳ₁), sequence(ℳ₂)))
        Base.$f(a::Number, ℳ::Multiplication) = Multiplication($f(a, sequence(ℳ)))
        Base.$f(ℳ::Multiplication, a::Number) = Multiplication($f(sequence(ℳ), a))
    end
end

Base.:/(ℳ::Multiplication, a::Number) = Multiplication(/(sequence(ℳ), a))
Base.:\(a::Number, ℳ::Multiplication) = Multiplication(\(a, sequence(ℳ)))

codomain(ℳ::Multiplication, s::SequenceSpace) = codomain(*, space(sequence(ℳ)), s)
_coeftype(ℳ::Multiplication, ::SequenceSpace, ::Type{T}) where {T} =
    promote_type(eltype(sequence(ℳ)), T)

#

function _project!(C::LinearOperator{<:SequenceSpace,<:SequenceSpace}, ℳ::Multiplication)
    dom = domain(C)
    codom = codomain(C)
    space_ℳ = space(sequence(ℳ))
    @inbounds for β ∈ _mult_domain_indices(dom), α ∈ indices(codom)
        if _isvalid(dom, space_ℳ, α, β)
            x = _inverse_symmetry_action(codom, α) * _symmetry_action(space_ℳ, α, β) * _symmetry_action(dom, β)
            C[α,_extract_valid_index(dom, β)] += exact(x) * sequence(ℳ)[_extract_valid_index(space_ℳ, α, β)]
        end
    end
    return C
end

_mult_domain_indices(s::TensorSpace) = TensorIndices(map(_mult_domain_indices, spaces(s)))

_isvalid(dom::TensorSpace{<:NTuple{N,BaseSpace}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}, β::NTuple{N,Int}) where {N} =
    @inbounds _isvalid(dom[1], s[1], α[1], β[1]) & _isvalid(Base.tail(dom), Base.tail(s), Base.tail(α), Base.tail(β))
_isvalid(dom::TensorSpace{<:Tuple{BaseSpace}}, s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}, β::Tuple{Int}) =
    @inbounds _isvalid(dom[1], s[1], α[1], β[1])

_symmetry_action(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}, β::NTuple{N,Int}) where {N} =
    @inbounds _symmetry_action(s[1], α[1], β[1]) * _symmetry_action(Base.tail(s), Base.tail(α), Base.tail(β))
_symmetry_action(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}, β::Tuple{Int}) =
    @inbounds _symmetry_action(s[1], α[1], β[1])
_symmetry_action(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _symmetry_action(s[1], α[1]) * _symmetry_action(Base.tail(s), Base.tail(α))
_symmetry_action(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) = @inbounds _symmetry_action(s[1], α[1])
_inverse_symmetry_action(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _inverse_symmetry_action(s[1], α[1]) * _inverse_symmetry_action(Base.tail(s), Base.tail(α))
_inverse_symmetry_action(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) = @inbounds _inverse_symmetry_action(s[1], α[1])

_extract_valid_index(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}, β::NTuple{N,Int}) where {N} =
    @inbounds (_extract_valid_index(s[1], α[1], β[1]), _extract_valid_index(Base.tail(s), Base.tail(α), Base.tail(β))...)
_extract_valid_index(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}, β::Tuple{Int}) =
    @inbounds (_extract_valid_index(s[1], α[1], β[1]),)
_extract_valid_index(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds (_extract_valid_index(s[1], α[1]), _extract_valid_index(Base.tail(s), Base.tail(α))...)
_extract_valid_index(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds (_extract_valid_index(s[1], α[1]),)

# Taylor

function _project!(C::LinearOperator{Taylor,Taylor}, ℳ::Multiplication)
    order_codomain = order(codomain(C))
    ord = order(sequence(ℳ))
    @inbounds for j ∈ indices(domain(C)), i ∈ j:min(order_codomain, ord+j)
        C[i,j] = sequence(ℳ)[i-j]
    end
    return C
end

_mult_domain_indices(s::Taylor) = indices(s)
_isvalid(::Taylor, s::Taylor, i::Int, j::Int) = _checkbounds_indices(i-j, s)

_symmetry_action(::Taylor, ::Int, ::Int) = 1
_symmetry_action(::Taylor, ::Int) = 1
_inverse_symmetry_action(::Taylor, ::Int) = 1

_extract_valid_index(::Taylor, i::Int, j::Int) = i-j
_extract_valid_index(::Taylor, i::Int) = i

# Fourier

_mult_domain_indices(s::Fourier) = indices(s)
_isvalid(::Fourier, s::Fourier, i::Int, j::Int) = _checkbounds_indices(abs(i-j), s)

_symmetry_action(::Fourier, ::Int, ::Int) = 1
_symmetry_action(::Fourier, ::Int) = 1
_inverse_symmetry_action(::Fourier, ::Int) = 1

_extract_valid_index(::Fourier, i::Int, j::Int) = i-j
_extract_valid_index(::Fourier, i::Int) = i

# Chebyshev

function _project!(C::LinearOperator{Chebyshev,Chebyshev}, ℳ::Multiplication)
    ord = order(sequence(ℳ))
    @inbounds for j ∈ indices(domain(C)), i ∈ indices(codomain(C))
        if abs(i-j) ≤ ord
            if j == 0
                C[i,j] = sequence(ℳ)[i]
            else
                C[i,j] = sequence(ℳ)[abs(i-j)]
                idx2 = i+j
                if idx2 ≤ ord
                    C[i,j] += sequence(ℳ)[idx2]
                end
            end
        end
    end
    return C
end

_mult_domain_indices(s::Chebyshev) = -order(s):order(s)
_isvalid(::Chebyshev, s::Chebyshev, i::Int, j::Int) = _checkbounds_indices(abs(i-j), s)

_symmetry_action(::Chebyshev, ::Int, ::Int) = 1
_symmetry_action(::Chebyshev, ::Int) = 1
_inverse_symmetry_action(::Chebyshev, ::Int) = 1

_extract_valid_index(::Chebyshev, i::Int, j::Int) = abs(i-j)
_extract_valid_index(::Chebyshev, i::Int) = abs(i)

# CosFourier

_mult_domain_indices(s::CosFourier) = _mult_domain_indices(desymmetrize(s))
_isvalid(::CosFourier, s::CosFourier, i::Int, j::Int) = _checkbounds_indices(abs(i-j), s)
_isvalid(::SinFourier, s::CosFourier, i::Int, j::Int) = (0 < abs(j)) & _checkbounds_indices(abs(i-j), s)

_symmetry_action(::CosFourier, ::Int, ::Int) = 1
_symmetry_action(::CosFourier, ::Int) = 1
_inverse_symmetry_action(::CosFourier, ::Int) = 1

_extract_valid_index(::CosFourier, i::Int, j::Int) = abs(i-j)
_extract_valid_index(::CosFourier, i::Int) = abs(i)

# SinFourier

_mult_domain_indices(s::SinFourier) = _mult_domain_indices(desymmetrize(s))
_isvalid(::SinFourier, s::SinFourier, i::Int, j::Int) = (0 < abs(j)) & _checkbounds_indices(abs(i-j), s)
_isvalid(::CosFourier, s::SinFourier, i::Int, j::Int) = _checkbounds_indices(abs(i-j), s)

function _symmetry_action(::SinFourier, i::Int, j::Int)
    x = j-i
    y = ifelse(x == 0, 0, flipsign(1, x))
    return complex(0, y)
end
function _symmetry_action(::SinFourier, i::Int)
    y = ifelse(i == 0, 0, flipsign(1, -i))
    return complex(0, y)
end
_inverse_symmetry_action(::SinFourier, ::Int) = complex(0, 1)

_extract_valid_index(::SinFourier, i::Int, j::Int) = abs(i-j)
_extract_valid_index(::SinFourier, i::Int) = abs(i)
