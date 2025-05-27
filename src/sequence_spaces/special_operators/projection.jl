struct Projection{T<:VectorSpace} <: SpecialOperator
    space :: T
end

Base.:*(𝒫₁::Projection, 𝒫₂::Projection) = Projection(intersect(𝒫₁.space, 𝒫₂.space))

Base.:*(𝒫::Projection, a::Sequence) = project(a, 𝒫.space)

Base.:*(A::LinearOperator, 𝒫::Projection) = project(A, 𝒫.space, codomain(A))
Base.:*(𝒫::Projection, A::LinearOperator) = project(A, domain(A), 𝒫.space)

Base.:*(S::SpecialOperator, 𝒫::Projection) = project(S, 𝒫.space, image(S, 𝒫.space))
Base.:*(𝒫::Projection, S::SpecialOperator) = project(S, _infer_domain(S, 𝒫.space), 𝒫.space)

_infer_domain(a, b) = throw(DomainError((a, b), "cannot project into a finite linear operator"))

_infer_domain(ℳ::Multiplication, s::SequenceSpace) = image(ℳ, s)

_infer_domain(𝒟::Derivative{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((αᵢ, sᵢ) -> _infer_domain(Derivative(αᵢ), sᵢ), order(𝒟), spaces(s)))
_infer_domain(𝒟::Derivative, s::Taylor) = image(Integral(order(𝒟)), s)
_infer_domain(::Derivative, s::Fourier) = s
_infer_domain(𝒟::Derivative, s::CosFourier) = image(Derivative(order(𝒟)), s)
_infer_domain(𝒟::Derivative, s::SinFourier) = image(Derivative(order(𝒟)), s)
# error for Chebyshev

_infer_domain(ℐ::Integral{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((αᵢ, sᵢ) -> _infer_domain(Integral(αᵢ), sᵢ), order(ℐ), spaces(s)))
_infer_domain(ℐ::Integral, s::Taylor) = image(Derivative(order(ℐ)), s)
_infer_domain(::Integral, s::Fourier) = s
_infer_domain(ℐ::Integral, s::CosFourier) = image(Derivative(order(ℐ)), s)
_infer_domain(ℐ::Integral, s::SinFourier) = image(Derivative(order(ℐ)), s)
# error for Chebyshev

_infer_domain(𝒮::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((τᵢ, sᵢ) -> _infer_domain(Shift(τᵢ), sᵢ), value(𝒮), spaces(s)))
_infer_domain(::Shift, s::Taylor) = s
_infer_domain(::Shift, s::Fourier) = s
_infer_domain(::Shift, s::Chebyshev) = s

_infer_domain(𝒮::Scale{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((γᵢ, sᵢ) -> _infer_domain(Scale(γᵢ), sᵢ), value(𝒮), spaces(s)))
_infer_domain(::Scale, s::Taylor) = s
_infer_domain(::Scale, s::Fourier) = s
_infer_domain(::Scale, s::Chebyshev) = s

_infer_domain(S::SpecialOperator, s::CartesianPower) = CartesianPower(_infer_domain(S, space(s)), nspaces(s))
_infer_domain(S::SpecialOperator, s::CartesianSpace) = CartesianProduct(map(sᵢ -> _infer_domain(S, sᵢ), spaces(s)))

#

"""
    project(a::Sequence, space_dest::VectorSpace, ::Type{T}=eltype(a))

Represent `a` as a [`Sequence`](@ref) in `space_dest`.

See also: [`project!`](@ref).
"""
function project(a::Sequence, space_dest::VectorSpace, ::Type{T}=eltype(a)) where {T}
    space_a = space(a)
    _iscompatible(space_a, space_dest) || return throw(ArgumentError("spaces must be compatible: a has space $space_a, destination space is $space_dest"))
    c = Sequence(space_dest, Vector{T}(undef, dimension(space_dest)))
    _project!(c, a)
    return c
end

"""
    project!(c::Sequence, a::Sequence)

Represent `a` as a [`Sequence`](@ref) in `space(c)`. The result is stored in `c`
by overwriting it.

See also: [`project`](@ref).
"""
function project!(c::Sequence, a::Sequence)
    space_a = space(a)
    space_c = space(c)
    _iscompatible(space_a, space_c) || return throw(ArgumentError("spaces must be compatible: c has space $space_c, a has space $space_a"))
    _project!(c, a)
    return c
end

"""
    project(A::LinearOperator{ParameterSpace,<:VectorSpace}, space_dest::VectorSpace, ::Type{T}=eltype(A))

Represent `A` as a [`Sequence`](@ref) in `space_dest`.

See also: [`project!`](@ref).
"""
project(A::LinearOperator{ParameterSpace,<:VectorSpace}, space_dest::VectorSpace, ::Type{T}=eltype(A)) where {T} =
    project!(Sequence(space_dest, Vector{T}(undef, dimension(space_dest))), A)

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
        coefficients(c) .= zero(eltype(c))
        @inbounds for α ∈ indices(space_a ∩ space_c)
            c[α] = a[α]
        end
    end
    return c
end

# UniformScaling

"""
    project(J::UniformScaling, domain_dest::VectorSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(J))

Represent `J` as a [`LinearOperator`](@ref) from `domain_dest` to `codomain_dest`.

See also: [`project!`](@ref).
"""
function project(J::UniformScaling, domain_dest::VectorSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(J)) where {T}
    _iscompatible(domain_dest, codomain_dest) || return throw(ArgumentError("spaces must be compatible: destination domain is $domain_dest, destination codomain is $codomain_dest"))
    C = LinearOperator(domain_dest, codomain_dest, zeros(T, dimension(codomain_dest), dimension(domain_dest)))
    _radd!(C, J)
    return C
end

"""
    project!(C::LinearOperator, J::UniformScaling)

Represent `J` as a [`LinearOperator`](@ref) from `domain(C)` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project`](@ref).
"""
function project!(C::LinearOperator, J::UniformScaling)
    domain_C = domain(C)
    codomain_C = codomain(C)
    _iscompatible(domain_C, codomain_C) || return throw(ArgumentError("spaces must be compatible: C has domain $domain_C, C has codomain $codomain_C"))
    coefficients(C) .= zero(eltype(C))
    _radd!(C, J)
    return C
end

#

"""
    project(A::LinearOperator, domain_dest::VectorSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(A))

Represent `A` as a [`LinearOperator`](@ref) from `domain_dest` to `codomain_dest`.

See also: [`project!`](@ref).
"""
function project(A::LinearOperator, domain_dest::VectorSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(A)) where {T}
    _iscompatible(domain(A), domain_dest) & _iscompatible(codomain(A), codomain_dest) || return throw(ArgumentError("spaces must be compatible"))
    C = LinearOperator(domain_dest, codomain_dest, Matrix{T}(undef, dimension(codomain_dest), dimension(domain_dest)))
    _project!(C, A)
    return C
end

"""
    project!(C::LinearOperator, A::LinearOperator)

Represent `A` as a [`LinearOperator`](@ref) from `domain(C)` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project`](@ref).
"""
function project!(C::LinearOperator, A::LinearOperator)
    _iscompatible(domain(A), domain(C)) & _iscompatible(codomain(A), codomain(C)) || return throw(ArgumentError("spaces must be compatible"))
    _project!(C, A)
    return C
end

"""
    project(a::Sequence, ::ParameterSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(a))

Represent `a` as a [`LinearOperator`](@ref) from `ParameterSpace` to `codomain_dest`.

See also: [`project!`](@ref).
"""
project(a::Sequence, domain_dest::ParameterSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(a)) where {T} =
    project!(LinearOperator(domain_dest, codomain_dest, Matrix{T}(undef, dimension(codomain_dest), 1)), a)

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
        coefficients(C) .= zero(eltype(C))
        @inbounds for β ∈ indices(domain_A ∩ domain_C), α ∈ indices(codomain_A ∩ codomain_C)
            C[α,β] = A[α,β]
        end
    end
    return C
end

# Cartesian spaces

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
