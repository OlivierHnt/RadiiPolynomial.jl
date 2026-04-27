"""
    Multiplication{T<:Sequence{<:SequenceSpace}} <: AbstractLinearOperator

Multiplication operator associated with a given [`Sequence`](@ref).

Field:
- `sequence :: T`

Constructor:
- `Multiplication(::Sequence{<:SequenceSpace})`
"""
struct Multiplication{T<:Sequence{<:SequenceSpace}} <: AbstractLinearOperator
    sequence :: T
end

sequence(ℳ::Multiplication) = ℳ.sequence

IntervalArithmetic._infer_numtype(ℳ::Multiplication) = IntervalArithmetic._infer_numtype(sequence(ℳ))
IntervalArithmetic._interval_infsup(::Type{T}, ℳ₁::Multiplication, ℳ₂::Multiplication, d::IntervalArithmetic.Decoration) where {T<:IntervalArithmetic.NumTypes} =
    Multiplication(IntervalArithmetic._interval_infsup(T, sequence(ℳ₁), sequence(ℳ₂), d))

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

#

domain(M::Multiplication, s::SequenceSpace) = _domain(*, space(sequence(M)), s)
_domain(::typeof(*), s::TensorSpace{<:NTuple{N,BaseSpace}}, s_prod::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((sᵢ, s_prodᵢ) -> _domain(*, sᵢ, s_prodᵢ), spaces(s), spaces(s_prod)))
_domain(::typeof(*), ::Taylor, s_prod::Taylor) = s_prod
_domain(::typeof(*), s::Fourier, s_prod::Fourier) = codomain(*, s, s_prod)
_domain(::typeof(*), s::Chebyshev, s_prod::Chebyshev) = codomain(*, s, s_prod)
function _domain(::typeof(*), s::SymmetricSpace, s_prod::SymmetricSpace)
    V = _domain(*, desymmetrize(s), desymmetrize(s_prod))
    G = _domain_convolution_symmetry(symmetry(s), symmetry(s_prod))
    return SymmetricSpace(V, G)
end
function _domain_convolution_symmetry(G::Group{N,T}, G_prod::Group{N,T}) where {N,T<:Number}
    idx = _by_idx_action(G)
    idx_prod = _by_idx_action(G_prod)

    elems = Set{GroupElement{N,T}}()
    for (key, vals) ∈ idx
        haskey(idx_prod, key) || continue
        vals_prod = idx_prod[key]
        for v ∈ vals, v_prod ∈ vals_prod
            if v.phase == v_prod.phase
                new_coeff = CoefAction{N,T}(v_prod.amplitude / v.amplitude, -v.phase)
                push!(elems, GroupElement{N,T}(key, new_coeff))
            end
        end
    end

    return unsafe_group!(elems)
end

codomain(ℳ::Multiplication, s::SequenceSpace) = codomain(*, space(sequence(ℳ)), s)

_coeftype(ℳ::Multiplication, ::SequenceSpace, ::Type{T}) where {T} =
    promote_type(eltype(sequence(ℳ)), T)

#

function _project!(C::LinearOperator{<:SequenceSpace,<:SequenceSpace}, ℳ::Multiplication)
    dom = domain(C)
    codom = codomain(C)
    space_ℳ = space(sequence(ℳ))
    ds = desymmetrize(space_ℳ)
    for β ∈ _mult_domain_indices(desymmetrize(dom))
        β_valid = _extract_valid_index(desymmetrize(dom), β)
        t, _ = _unsafe_get_representative_and_action(dom, β_valid)
        if _checkbounds_indices(t, dom)
            for α ∈ indices(codom)
                l = _extract_valid_index(ds, α, β)
                if _checkbounds_indices(l, ds)
                    @inbounds C[α,t] += getcoefficient(sequence(ℳ), (ds, l))
                end
            end
        end
    end
    return C
end

# Tensor space

_mult_domain_indices(s::TensorSpace) = TensorIndices(map(_mult_domain_indices, spaces(s)))

_isvalid(dom::TensorSpace{<:NTuple{N,BaseSpace}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}, β::NTuple{N,Int}) where {N} =
    @inbounds _isvalid(dom[1], s[1], α[1], β[1]) & _isvalid(Base.tail(dom), Base.tail(s), Base.tail(α), Base.tail(β))
_isvalid(dom::TensorSpace{<:Tuple{BaseSpace}}, s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}, β::Tuple{Int}) =
    @inbounds _isvalid(dom[1], s[1], α[1], β[1])

_extract_valid_index(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}, β::NTuple{N,Int}) where {N} =
    @inbounds (_extract_valid_index(s[1], α[1], β[1]), _extract_valid_index(Base.tail(s), Base.tail(α), Base.tail(β))...)
_extract_valid_index(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}, β::Tuple{Int}) =
    @inbounds (_extract_valid_index(s[1], α[1], β[1]),)
_extract_valid_index(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds (_extract_valid_index(s[1], α[1]), _extract_valid_index(Base.tail(s), Base.tail(α))...)
_extract_valid_index(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds (_extract_valid_index(s[1], α[1]),)

# Taylor

_mult_domain_indices(s::Taylor) = indices(s)
_isvalid(::Taylor, s::Taylor, i::Int, j::Int) = _checkbounds_indices(i-j, s)

_extract_valid_index(::Taylor, i::Int, j::Int) = i-j
_extract_valid_index(::Taylor, i::Int) = i

# Fourier

_mult_domain_indices(s::Fourier) = indices(s)
_isvalid(::Fourier, s::Fourier, i::Int, j::Int) = _checkbounds_indices(abs(i-j), s)

_extract_valid_index(::Fourier, i::Int, j::Int) = i-j
_extract_valid_index(::Fourier, i::Int) = i

# Chebyshev

_mult_domain_indices(s::Chebyshev) = -order(s):order(s)
_isvalid(::Chebyshev, s::Chebyshev, i::Int, j::Int) = _checkbounds_indices(abs(i-j), s)

_extract_valid_index(::Chebyshev, i::Int, j::Int) = abs(i-j)
_extract_valid_index(::Chebyshev, i::Int) = abs(i)



#

Base.:*(ℳ::Multiplication, a::Sequence) = *(sequence(ℳ), a)
