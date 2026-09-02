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

IntervalArithmetic.interval(::Type{T}, ℳ::Multiplication) where {T<:IntervalArithmetic.NumTypes} = Multiplication(IntervalArithmetic.interval(T, sequence(ℳ)))
IntervalArithmetic.interval(ℳ::Multiplication) = Multiplication(interval(sequence(ℳ)))
# IntervalArithmetic._infer_numtype(ℳ::Multiplication) = IntervalArithmetic._infer_numtype(sequence(ℳ))
# IntervalArithmetic._interval_infsup(::Type{T}, ℳ₁::Multiplication, ℳ₂::Multiplication, d::IntervalArithmetic.Decoration) where {T<:IntervalArithmetic.NumTypes} =
#     Multiplication(IntervalArithmetic._interval_infsup(T, sequence(ℳ₁), sequence(ℳ₂), d))

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
function _domain_convolution_symmetry(G::Group{N,T,L}, G_prod::Group{N,T,L}) where {N,T<:Number,L}
    idx = _by_idx_action(G)
    idx_prod = _by_idx_action(G_prod)

    elems = Set{GroupElement{N,T,L}}()
    for (key, vals) ∈ idx
        haskey(idx_prod, key) || continue
        vals_prod = idx_prod[key]
        for v ∈ vals, v_prod ∈ vals_prod
            # convolution identity e^{iπ⟨φ,j⟩} e^{iπ⟨φ,k−j⟩} = e^{iπ⟨φ,k⟩}
            if v.phase == v_prod.phase
                new_coeff = Cocycle{N,T}(v_prod.amplitude / v.amplitude, v.phase)
                push!(elems, GroupElement(key, new_coeff))
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
        # `x_β = factor * x_rep`: the contribution of `β` goes to the column `q` of its representative
        # weighted by the cocycle factor, so that (a*x)_α = Σ_rep x_rep Σ_{β ∈ Orb(rep)} factor(β) a_{α-β}
        q, factor = _unsafe_rep_pos_cocycle(dom, β_valid)
        if q != 0
            one_factor = _safe_isone(factor) # skip the product by 1 (keeps interval guarantees when the domain is not intervalized)
            for (i, α) ∈ enumerate(indices(codom))
                l = _extract_valid_index(ds, α, β)
                if _checkbounds_indices(l, ds)
                    v = getcoefficient(sequence(ℳ), (ds, l))
                    @inbounds coefficients(C)[i,q] += one_factor ? v : factor * v
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
