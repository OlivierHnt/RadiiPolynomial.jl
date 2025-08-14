"""
    SpecialOperator

Abstract type for all special operators.
"""
abstract type SpecialOperator end

add!(C::LinearOperator, S₁::SpecialOperator, S₂::SpecialOperator) = add!(C, project(S₁, domain(C), codomain(C), eltype(C)), S₂)
add!(C::LinearOperator, S::SpecialOperator, A::LinearOperator) = add!(C, project(S, domain(C), codomain(C), eltype(C)), A)
add!(C::LinearOperator, A::LinearOperator, S::SpecialOperator) = add!(C, A, project(S, domain(C), codomain(C), eltype(C)))

sub!(C::LinearOperator, S₁::SpecialOperator, S₂::SpecialOperator) = sub!(C, project(S₁, domain(C), codomain(C), eltype(C)), S₂)
sub!(C::LinearOperator, S::SpecialOperator, A::LinearOperator) = sub!(C, project(S, domain(C), codomain(C), eltype(C)), A)
sub!(C::LinearOperator, A::LinearOperator, S::SpecialOperator) = sub!(C, A, project(S, domain(C), codomain(C), eltype(C)))

radd!(A::LinearOperator, S::SpecialOperator) = radd!(A, project(S, domain(A), codomain(A), eltype(A)))
rsub!(A::LinearOperator, S::SpecialOperator) = rsub!(A, project(S, domain(A), codomain(A), eltype(A)))

ladd!(S::SpecialOperator, A::LinearOperator) = ladd!(project(S, domain(A), codomain(A), eltype(A)), A)
lsub!(S::SpecialOperator, A::LinearOperator) = lsub!(project(S, domain(A), codomain(A), eltype(A)), A)

Base.:*(S::SpecialOperator, A::LinearOperator) = (S * Projection(codomain(A))) * A
Base.:*(A::LinearOperator, S::SpecialOperator) = A * (Projection(domain(A)) * S)

function mul!(C::LinearOperator, S₁::SpecialOperator, S₂::SpecialOperator, α::Number, β::Number)
    domain_C = domain(C)
    return mul!(C, S₁, project(S₂, domain_C, image(S₂, domain_C), eltype(C)), α, β)
end
mul!(C::LinearOperator, S::SpecialOperator, A::LinearOperator, α::Number, β::Number) =
    mul!(C, project(S, codomain(A), codomain(C), eltype(C)), A, α, β)
mul!(C::LinearOperator, A::LinearOperator, S::SpecialOperator, α::Number, β::Number) =
    mul!(C, A, project(S, domain(C), domain(A), eltype(C)), α, β)

mul!(c::Sequence, S::SpecialOperator, a::Sequence, α::Number, β::Number) =
    mul!(c, project(S, space(a), space(c), eltype(c)), a, α, β)

#

function Base.:+(A::LinearOperator, S::SpecialOperator)
    domain_A = domain(A)
    new_codomain = image(+, codomain(A), image(S, domain_A))
    C = zeros(_coeftype(S, domain_A, eltype(A)), domain_A, new_codomain)
    ladd!(A, project!(C, S))
    return C
end
function Base.:+(S::SpecialOperator, A::LinearOperator)
    domain_A = domain(A)
    new_codomain = image(+, image(S, domain_A), codomain(A))
    C = zeros(_coeftype(S, domain_A, eltype(A)), domain_A, new_codomain)
    radd!(project!(C, S), A)
    return C
end
function Base.:-(A::LinearOperator, S::SpecialOperator)
    domain_A = domain(A)
    new_codomain = image(-, codomain(A), image(S, domain_A))
    C = zeros(_coeftype(S, domain_A, eltype(A)), domain_A, new_codomain)
    lsub!(A, project!(C, S))
    return C
end
function Base.:-(S::SpecialOperator, A::LinearOperator)
    domain_A = domain(A)
    new_codomain = image(-, image(S, domain_A), codomain(A))
    C = zeros(_coeftype(S, domain_A, eltype(A)), domain_A, new_codomain)
    rsub!(project!(C, S), A)
    return C
end

#

image(::UniformScaling, s::VectorSpace) = s

# infinite banded linear operator

struct BandedLinearOperator{T<:LinearOperator,S} <: SpecialOperator
    finite_part :: T
    banded_part :: S
end

const BLinearOperator = BandedLinearOperator

Base.:*(S::BandedLinearOperator, a::Sequence) = (S * Projection(space(a))) * a

image(A::BandedLinearOperator, s::VectorSpace) =
    image(A.finite_part, s) ∪ _image(A.banded_part, s)
_image(A, s) = image(A, s)
function _image(A::Union{Vector,Matrix}, s)
    v = CartesianProduct(ParameterSpace())
    for i ∈ 1:size(A, 1)
        w = _image(A[i,1], s[1])
        for j ∈ 2:size(A, 2)
            w = w ∪ _image(A[i,j], s[j])
        end
        v = CartesianProduct(spaces(v)..., w)
    end
    nspaces(v) == 2 && return v[2]
    return v[2:nspaces(v)]
end
function _image(A::Vector, s::ParameterSpace)
    v = ParameterSpace()
    for i ∈ 1:length(A)
        w = _image(A[i], s)
        v = v × w
    end
    nspaces(v) == 2 && return v[2]
    return v[2:nspaces(v)]
end


function project(A::BandedLinearOperator, dom::VectorSpace, codom::VectorSpace)
    B = zeros(eltype(A.finite_part), dom, codom)
    _band_project!(B, A.banded_part)
    view(B, codom ∩ codomain(A.finite_part), dom ∩ domain(A.finite_part)) .= view(A.finite_part, codom ∩ codomain(A.finite_part), dom ∩ domain(A.finite_part))
    return B
end

_infer_domain(A::BandedLinearOperator, s::VectorSpace) = domain(A.finite_part) ∪ _infer_domain(A.banded_part, s)


function _band_project!(B::LinearOperator, A::Matrix)
    for j ∈ axes(A, 2), i ∈ axes(A, 1)
       _band_project!(component(B, i, j), A[i,j])
    end
    return B
end
function _band_project!(B::LinearOperator{<:VectorSpace,ParameterSpace}, A::Matrix)
    for j ∈ axes(A, 2)
       _band_project!(component(B, j), A[j])
    end
    return B
end
function _band_project!(B::LinearOperator, A::Vector)
    for i ∈ axes(A, 1)
       _band_project!(component(B, i), A[i])
    end
    return B
end
_band_project!(B::LinearOperator, A::Union{LinearOperator,SpecialOperator,UniformScaling,ComposedFunction}) = project!(B, A)
