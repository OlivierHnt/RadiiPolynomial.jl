struct Multiplication{T<:Sequence{<:SequenceSpace}}
    sequence :: T
end

# fallback arithmetic methods

Base.:+(A::LinearOperator, ℳ::Multiplication) = A + project(ℳ, domain(A), codomain(A))
Base.:+(ℳ::Multiplication, A::LinearOperator) = project(ℳ, domain(A), codomain(A)) + A
Base.:-(A::LinearOperator, ℳ::Multiplication) = A - project(ℳ, domain(A), codomain(A))
Base.:-(ℳ::Multiplication, A::LinearOperator) = project(ℳ, domain(A), codomain(A)) - A

add!(C::LinearOperator, A::LinearOperator, ℳ::Multiplication) = add!(C, A, project(ℳ, domain(A), codomain(A), eltype(C)))
add!(C::LinearOperator, ℳ::Multiplication, A::LinearOperator) = add!(C, project(ℳ, domain(A), codomain(A), eltype(C)), A)
sub!(C::LinearOperator, A::LinearOperator, ℳ::Multiplication) = sub!(C, A, project(ℳ, domain(A), codomain(A), eltype(C)))
sub!(C::LinearOperator, ℳ::Multiplication, A::LinearOperator) = sub!(C, project(ℳ, domain(A), codomain(A), eltype(C)), A)

radd!(A::LinearOperator, ℳ::Multiplication) = radd!(A, project(ℳ, domain(A), codomain(A), eltype(A)))
rsub!(A::LinearOperator, ℳ::Multiplication) = rsub!(A, project(ℳ, domain(A), codomain(A), eltype(A)))

ladd!(ℳ::Multiplication, A::LinearOperator) = ladd!(project(ℳ, domain(A), codomain(A), eltype(A)), A)
lsub!(ℳ::Multiplication, A::LinearOperator) = lsub!(project(ℳ, domain(A), codomain(A), eltype(A)), A)

function Base.:*(ℳ::Multiplication, A::LinearOperator)
    codomain_A = codomain(A)
    return project(ℳ, codomain_A, image(*, space(ℳ.sequence), codomain_A)) * A
end

LinearAlgebra.mul!(c::Sequence, ℳ::Multiplication, a::Sequence, α::Number, β::Number) =
    mul!(c, ℳ.sequence, a, α, β)
LinearAlgebra.mul!(C::LinearOperator, ℳ::Multiplication, A::LinearOperator, α::Number, β::Number) =
    mul!(C, project(ℳ, codomain(A), codomain(C), eltype(C)), A, α, β)
LinearAlgebra.mul!(C::LinearOperator, A::LinearOperator, ℳ::Multiplication, α::Number, β::Number) =
    mul!(C, A, project(ℳ, domain(C), domain(A), eltype(C)), α, β)

#

(ℳ::Multiplication)(a::Sequence) = *(ℳ, a)
Base.:*(ℳ::Multiplication, a::Sequence) = *(ℳ.sequence, a)

Base.:+(ℳ::Multiplication) = Multiplication(+(ℳ.sequence))
Base.:-(ℳ::Multiplication) = Multiplication(-(ℳ.sequence))
Base.:^(ℳ::Multiplication, n::Int) = Multiplication(ℳ.sequence ^ n)

for f ∈ (:+, :-, :*)
    @eval begin
        Base.$f(ℳ₁::Multiplication, ℳ₂::Multiplication) = Multiplication($f(ℳ₁.sequence, ℳ₂.sequence))
        Base.$f(a::Number, ℳ::Multiplication) = Multiplication($f(a, ℳ.sequence))
        Base.$f(ℳ::Multiplication, a::Number) = Multiplication($f(ℳ.sequence, a))
    end
end

Base.:/(ℳ::Multiplication, a::Number) = Multiplication(/(ℳ.sequence, a))
Base.:\(a::Number, ℳ::Multiplication) = Multiplication(\(a, ℳ.sequence))

LinearAlgebra.opnorm(ℳ::Multiplication, X::BanachSpace) = norm(ℳ.sequence, X)

function project(ℳ::Multiplication, domain::SequenceSpace, codomain::SequenceSpace, ::Type{T}=eltype(ℳ.sequence)) where {T}
    _iscompatible(domain, codomain) & _iscompatible(space(ℳ.sequence), domain) || return throw(ArgumentError("spaces must be compatible"))
    C = LinearOperator(domain, codomain, zeros(T, dimension(codomain), dimension(domain)))
    _project!(C, ℳ)
    return C
end

function project!(C::LinearOperator{<:SequenceSpace,<:SequenceSpace}, ℳ::Multiplication)
    domain_C = domain(C)
    _iscompatible(domain_C, codomain(C)) & _iscompatible(space(ℳ.sequence), domain_C) || return throw(ArgumentError("spaces must be compatible"))
    coefficients(C) .= zero(eltype(C))
    _project!(C, ℳ)
    return C
end

#

function _project!(C::LinearOperator{<:TensorSpace,<:TensorSpace}, ℳ::Multiplication)
    space_ℳ = space(ℳ.sequence)
    @inbounds for β ∈ _mult_domain_indices(domain(C)), α ∈ indices(codomain(C))
        if _isvalid(space_ℳ, α, β)
            C[α,_extract_valid_index(space_ℳ, β, zero.(α))] += ℳ.sequence[_extract_valid_index(space_ℳ, α, β)]
        end
    end
    return C
end

_mult_domain_indices(s::TensorSpace) = TensorIndices(map(_mult_domain_indices, spaces(s)))

_isvalid(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}, β::NTuple{N,Int}) where {N} =
    @inbounds _isvalid(s[1], α[1], β[1]) & _isvalid(Base.tail(s), Base.tail(α), Base.tail(β))
_isvalid(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}, β::Tuple{Int}) =
    @inbounds _isvalid(s[1], α[1], β[1])

_extract_valid_index(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}, β::NTuple{N,Int}) where {N} =
    @inbounds (_extract_valid_index(s[1], α[1], β[1]), _extract_valid_index(Base.tail(s), Base.tail(α), Base.tail(β))...)
_extract_valid_index(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}, β::Tuple{Int}) =
    @inbounds (_extract_valid_index(s[1], α[1], β[1]),)

# Taylor

function _project!(C::LinearOperator{Taylor,Taylor}, ℳ::Multiplication)
    order_codomain = order(codomain(C))
    ord = order(ℳ.sequence)
    @inbounds for j ∈ indices(domain(C)), i ∈ j:min(order_codomain, ord+j)
        C[i,j] = ℳ.sequence[i-j]
    end
    return C
end

_mult_domain_indices(s::Taylor) = indices(s)
_isvalid(s::Taylor, i::Int, j::Int) = 0 ≤ i-j ≤ order(s)
_extract_valid_index(::Taylor, i::Int, j::Int) = i-j

# Fourier

function _project!(C::LinearOperator{<:Fourier,<:Fourier}, ℳ::Multiplication)
    order_codomain = order(codomain(C))
    ord = order(ℳ.sequence)
    @inbounds for j ∈ indices(domain(C)), i ∈ max(-order_codomain, -ord+j):min(order_codomain, ord+j)
        C[i,j] = ℳ.sequence[i-j]
    end
    return C
end

_mult_domain_indices(s::Fourier) = indices(s)
_isvalid(s::Fourier, i::Int, j::Int) = abs(i-j) ≤ order(s)
_extract_valid_index(::Fourier, i::Int, j::Int) = i-j

# Chebyshev

function _project!(C::LinearOperator{Chebyshev,Chebyshev}, ℳ::Multiplication)
    ord = order(ℳ.sequence)
    @inbounds for j ∈ indices(domain(C)), i ∈ indices(codomain(C))
        if abs(i-j) ≤ ord
            if j == 0
                C[i,j] = ℳ.sequence[i]
            else
                C[i,j] = ℳ.sequence[abs(i-j)]
                idx2 = i+j
                if idx2 ≤ ord
                    C[i,j] += ℳ.sequence[idx2]
                end
            end
        end
    end
    return C
end

_mult_domain_indices(s::Chebyshev) = -order(s):order(s)
_isvalid(s::Chebyshev, i::Int, j::Int) = abs(i-j) ≤ order(s)
_extract_valid_index(::Chebyshev, i::Int, j::Int) = abs(i-j)
