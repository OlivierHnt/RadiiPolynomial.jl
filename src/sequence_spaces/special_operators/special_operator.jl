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

function Base.:*(S::SpecialOperator, A::LinearOperator)
    codomain_A = codomain(A)
    image_S = image(S, codomain_A)
    return project(S, codomain_A, image_S, _coeftype(S, codomain_A, eltype(A))) * A
end

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
