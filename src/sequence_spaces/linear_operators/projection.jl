# composition of operators

codomain(A::ComposedOperator, s::VectorSpace) = codomain(A.outer, codomain(A.inner, s))

_infer_domain(A::ComposedOperator, s::VectorSpace) = _infer_domain(A.inner, _infer_domain(A.outer, s))

project(A::ComposedOperator, dom::VectorSpace, codom::VectorSpace) = Projection(codom) * (A.outer * (A.inner * Projection(dom)))
project(A::ComposedOperator, dom::VectorSpace, codom::VectorSpace, ::Type{T}) where {T} =
    Projection(codom, T) * (A.outer * (A.inner * Projection(dom, T))) # project(A.outer, _infer_domain(A.outer, codom), codom, T) * project(A.inner, dom, codomain(A.inner, dom), T)

project!(A::AbstractLinearOperator, B::ComposedOperator) = project!(A, project(B, domain(A), codomain(A)))

#

"""
    Projection{T<:VectorSpace,S<:Number} <: AbstractLinearOperator

Projection operator onto `space::T`. This is used to resize a sequence, a
linear operator, or materialize a finite projection of an abstract operator. The
type of the ouput entries is determined by promoting against `S`.

Field:
- `space :: T`
"""
struct Projection{T<:VectorSpace,S<:Number} <: AbstractLinearOperator
    space :: T
end

Projection(space::VectorSpace, ::Type{T}=Float64) where {T<:Number} = Projection{typeof(space),T}(space)

domain(P::Projection) = P.space # needed for general methods

codomain(P::Projection) = P.space # needed for general methods
function codomain(P::Projection, s::VectorSpace)
    dom = P.space
    _iscompatible(dom, s) || return throw(ArgumentError("spaces must be compatible: projection space is $dom, domain space is $s"))
    return dom
end

coefficients(P::Projection) = project(I, P.space, P.space) # needed for general methods

Base.eltype(::Projection{<:VectorSpace,S}) where {S<:Number} = S
Base.eltype(::Type{Projection{<:VectorSpace,S}}) where {S<:Number} = S

Base.:+(P::Projection, A::LinearOperator) = project(P, P.space, P.space) + A
Base.:+(P::Projection, A::LinearOperator) = A + project(P, P.space, P.space)
Base.:-(P::Projection, A::LinearOperator) = project(P, P.space, P.space) - A
Base.:-(P::Projection, A::LinearOperator) = A - project(P, P.space, P.space)

Base.:*(P₁::Projection, P₂::Projection) = Projection(intersect(P₁.space, P₂.space))

Base.:*(P::Projection, a::AbstractSequence) = project(a, P.space, promote_type(eltype(P), eltype(a)))

Base.:*(A::AbstractLinearOperator, P::Projection) = project(A, P.space, codomain(A, P.space), _coeftype(A, P.space, eltype(P)))
Base.:*(P::Projection, A::AbstractLinearOperator) = _lproj(A, _infer_domain(A, P.space), P)
_lproj(A::AbstractLinearOperator, domain::VectorSpace, P::Projection) = project(A, domain, P.space, _coeftype(A, domain, eltype(P)))
_lproj(A::AbstractLinearOperator, ::EmptySpace, P::Projection) = ComposedOperator(P, A)

Base.:*(A::ComposedOperator, P::Projection) = A.outer * (A.inner * P)
Base.:*(P::Projection, A::ComposedOperator) = (P * A.outer) * A.inner

Base.:*(P::Add{<:Projection,<:Projection}, A::AbstractLinearOperator) = P.A * A + P.B * A
Base.:*(A::AbstractLinearOperator, P::Add{<:Projection,<:Projection}) = A * P.A + A * P.B
Base.:*(P::Add{<:Projection,<:Negate{<:Projection}}, A::AbstractLinearOperator) = P.A * A + P.B * A
Base.:*(A::AbstractLinearOperator, P::Add{<:Projection,<:Negate{<:Projection}}) = A * P.A + A * P.B
Base.:*(P::Add{<:Negate{<:Projection},<:Projection}, A::AbstractLinearOperator) = P.A * A + P.B * A
Base.:*(A::AbstractLinearOperator, P::Add{<:Negate{<:Projection},<:Projection}) = A * P.A + A * P.B
Base.:*(P::Add{<:Negate{<:Projection},<:Negate{<:Projection}}, A::AbstractLinearOperator) = P.A * A + P.B * A
Base.:*(A::AbstractLinearOperator, P::Add{<:Negate{<:Projection},<:Negate{<:Projection}}) = A * P.A + A * P.B

Base.:*(P::Negate{<:Projection}, A::AbstractLinearOperator) = -(P.A * A)
Base.:*(A::AbstractLinearOperator, P::Negate{<:Projection}) = -(A * P.A)

_infer_domain(a, b) = throw(DomainError((a, b), "cannot infer a domain"))

function _infer_domain(A::LinearOperator, s::VectorSpace)
    _iscompatible(codomain(A), s) || return throw(ArgumentError("spaces must be compatible"))
    return domain(A)
end

_infer_domain(A::BandedLinearOperator, s::VectorSpace) = _infer_domain(linear_operator(A)) ∪ _infer_domain(banded_operator(A), s)

_infer_domain(S::Add, s::VectorSpace) = _infer_domain(S.A, s) ∪ _infer_domain(S.B, s)
_infer_domain(S::Negate, s::VectorSpace) = _infer_domain(S.A, s)

_infer_domain(::UniformScalingOperator, s::VectorSpace) = s

#

"""
    project(a::AbstractSequence, space_dest::VectorSpace, ::Type{T}=eltype(a))

Represent `a` as a [`Sequence`](@ref) in `space_dest`.

See also: [`project!`](@ref).
"""
function project(a::AbstractSequence, space_dest::VectorSpace, ::Type{T}=eltype(a)) where {T}
    space_a = space(a)
    _iscompatible(space_a, space_dest) || return throw(ArgumentError("spaces must be compatible: a has space $space_a, destination space is $space_dest"))
    c = Sequence(space_dest, zeros(T, dimension(space_dest)))
    _project!(c, a)
    return c
end

"""
    project!(c::Sequence, a::AbstractSequence)

Represent `a` as a [`Sequence`](@ref) in `space(c)`. The result is stored in `c`
by overwriting it.

See also: [`project`](@ref).
"""
function project!(c::Sequence, a::AbstractSequence)
    space_a = space(a)
    space_c = space(c)
    _iscompatible(space_a, space_c) || return throw(ArgumentError("spaces must be compatible: c has space $space_c, a has space $space_a"))
    _project!(c, a)
    return c
end

"""
    project(a::Sequence, ::ParameterSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(a))

Represent `a` as a [`LinearOperator`](@ref) from `ParameterSpace` to `codomain_dest`.

See also: [`project!`](@ref).
"""
project(a::Sequence, domain_dest::ParameterSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(a)) where {T} =
    project!(zeros(T, domain_dest, codomain_dest), a)

"""
    project!(C::LinearOperator{ParameterSpace,<:VectorSpace}, a::Sequence)

Represent `a` as a [`LinearOperator`](@ref) from `ParameterSpace` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project`](@ref).
"""
function project!(C::LinearOperator{ParameterSpace,<:VectorSpace}, a::Sequence)
    project!(Sequence(codomain(C), vec(coefficients(C))), a)
    return C
end

function _project!(c::Sequence, a::Sequence)
    space_a = space(a)
    space_c = space(c)
    if space_a == space_c
        coefficients(c) .= coefficients(a)
    elseif space_c ⊆ space_a
        @inbounds for α ∈ indices(space_c)
            c[α] = a[α]
        end
    else
        @inbounds for α ∈ indices(space_a ∩ space_c)
            c[α] = a[α]
        end
    end
    return c
end

function _project!(c::Sequence{<:CartesianSpace}, a::Sequence{<:CartesianSpace})
    space_c = space(c)
    if space(a) == space_c
        coefficients(c) .= coefficients(a)
    else
        @inbounds for i ∈ 1:nspaces(space_c)
            _project!(component(c, i), component(a, i))
        end
    end
    return c
end

#

"""
    project(A::AbstractLinearOperator, domain_dest::VectorSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(A))

Represent `A` as a [`LinearOperator`](@ref) from `domain_dest` to `codomain_dest`.

See also: [`project!`](@ref).
"""
function project(A::AbstractLinearOperator, domain_dest::VectorSpace, codomain_dest::VectorSpace, ::Type{T}=_coeftype(A, domain_dest, Float64)) where {T}
    _iscompatible(A, codomain(A, domain_dest), codomain_dest) || return throw(ArgumentError("spaces must be compatible"))
    C = zeros(T, domain_dest, codomain_dest)
    _project!(C, A)
    return C
end

project(J::UniformScaling, domain_dest::VectorSpace, codomain_dest::VectorSpace) = project(UniformScalingOperator(J), domain_dest, codomain_dest)
project(J::UniformScaling, domain_dest::VectorSpace, codomain_dest::VectorSpace, ::Type{T}) where {T} = project(UniformScalingOperator(J), domain_dest, codomain_dest, T)

_iscompatible(::AbstractLinearOperator, image_domain_dest, codomain_dest) =
    _iscompatible(image_domain_dest, codomain_dest)

_coeftype(A::LinearOperator, ::VectorSpace, ::Type{T}) where {T} = promote_type(eltype(A), T)
_coeftype(A::BandedLinearOperator, ::VectorSpace, ::Type{T}) where {T} = promote_type(eltype(A), T)

_coeftype(S::Add, s::VectorSpace, ::Type{T}) where {T} = promote_type(_coeftype(S.A, s, T), _coeftype(S.B, s, T))
_coeftype(S::Negate, s::VectorSpace, ::Type{T}) where {T} = _coeftype(S.A, s, T)

_coeftype(P::Projection, ::VectorSpace, ::Type{T}) where {T} = promote_type(eltype(P), T)
_coeftype(J::UniformScalingOperator, ::VectorSpace, ::Type{T}) where {T} = promote_type(eltype(J), T)

"""
    project!(C::LinearOperator, A::AbstractLinearOperator)

Represent `A` as a [`LinearOperator`](@ref) from `domain(C)` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project`](@ref).
"""
function project!(C::LinearOperator, A::AbstractLinearOperator)
    _iscompatible(A, codomain(A, domain(C)), codomain(C)) || return throw(ArgumentError("spaces must be compatible"))
    _project!(C, A)
    return C
end

project!(C::LinearOperator, J::UniformScaling) = project!(C, UniformScalingOperator(J))

"""
    project(A::LinearOperator{ParameterSpace,<:VectorSpace}, space_dest::VectorSpace, ::Type{T}=eltype(A))

Represent `A` as a [`Sequence`](@ref) in `space_dest`.

See also: [`project!`](@ref).
"""
project(A::LinearOperator{ParameterSpace,<:VectorSpace}, space_dest::VectorSpace, ::Type{T}=eltype(A)) where {T} =
    project!(zeros(T, space_dest), A)

"""
    project!(c::Sequence, A::LinearOperator{ParameterSpace,<:VectorSpace})

Represent `A` as a [`Sequence`](@ref) in `space(c)`. The result is stored in `c`
by overwriting it.

See also: [`project`](@ref).
"""
function project!(c::Sequence, A::LinearOperator{ParameterSpace,<:VectorSpace})
    project!(LinearOperator(domain(A), space(c), reshape(coefficients(c), :, 1)), A)
    return c
end

function _project!(C::LinearOperator, A::LinearOperator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == domain_C && codomain_A == codomain_C
        coefficients(C) .= coefficients(A)
    elseif domain_C ⊆ domain_A && codomain_C ⊆ codomain_A
        @inbounds for β ∈ indices(domain_C), α ∈ indices(codomain_C)
            C[α,β] = A[α,β]
        end
    else
        @inbounds for β ∈ indices(domain_A ∩ domain_C), α ∈ indices(codomain_A ∩ codomain_C)
            C[α,β] = A[α,β]
        end
    end
    return C
end

function _project!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace})
    domain_C = domain(C)
    codomain_C = codomain(C)
    if domain(A) == domain_C && codomain(A) == codomain_C
        coefficients(C) .= coefficients(A)
    else
        @inbounds for j ∈ 1:nspaces(domain_C), i ∈ 1:nspaces(codomain_C)
            _project!(component(C, i, j), component(A, i, j))
        end
    end
    return C
end

function _project!(C::LinearOperator{<:CartesianSpace,<:VectorSpace}, A::LinearOperator{<:CartesianSpace,<:VectorSpace})
    domain_C = domain(C)
    if domain(A) == domain_C && codomain(A) == codomain(C)
        coefficients(C) .= coefficients(A)
    else
        @inbounds for j ∈ 1:nspaces(domain_C)
            _project!(component(C, j), component(A, j))
        end
    end
    return C
end

function _project!(C::LinearOperator{<:VectorSpace,<:CartesianSpace}, A::LinearOperator{<:VectorSpace,<:CartesianSpace})
    codomain_C = codomain(C)
    if domain(A) == domain(C) && codomain(A) == codomain_C
        coefficients(C) .= coefficients(A)
    else
        @inbounds for i ∈ 1:nspaces(codomain_C)
            _project!(component(C, i), component(A, i))
        end
    end
    return C
end

#

function _project!(A::LinearOperator, P::Projection)
    @inbounds for α ∈ indices(P.space)
        A[α,α] = one(eltype(A))
    end
    return A
end

_project!(A::LinearOperator, B::Add) = add!(A, B.A, B.B)
_project!(A::LinearOperator, B::Negate) = rsub!(A, B.A)
_project!(A::LinearOperator, J::UniformScalingOperator) = _radd!(A, J)





# tail

tail(a::Sequence, order) = tail!(copy(a), order)

_tail_order_getindex(order::Number, i) = order

_tail_order_getindex(order, i) = order[i]

function tail!(a::Sequence{<:CartesianSpace}, order)
    for i ∈ 1:nspaces(space(a))
        tail!(component(a, i), _tail_order_getindex(order, i))
    end
    return a
end

function tail!(a::Sequence{ParameterSpace}, order)
    if order ≥ 0
        a[1] = zero(eltype(a))
    end
    return a
end

function tail!(a::Sequence{<:TensorSpace}, order)
    for α ∈ indices(space(a))
        if all(abs.(α) .≤ order)
            a[α] = zero(eltype(a))
        end
    end
    return a
end

function tail!(a::Sequence{<:BaseSpace}, order)
    for α ∈ indices(space(a))
        if abs(α) ≤ order
            a[α] = zero(eltype(a))
        end
    end
    return a
end
