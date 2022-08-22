# NOTE: s₃ = image(+, s₁, s₂) satisfies indices(s₃) == indices(s₁) ∪ indices(s₂)

function add_bar end
const +̄  = add_bar

function sub_bar end
const -̄  = sub_bar

function mul_bar end
const *̄  = mul_bar

function pow_bar end
const ^̄  = pow_bar

# fallback methods

image(::typeof(-), s₁::VectorSpace, s₂::VectorSpace) = image(+, s₁, s₂)
image(::typeof(sub_bar), s₁::VectorSpace, s₂::VectorSpace) = image(add_bar, s₁, s₂)

# Parameter space

image(::typeof(+), ::ParameterSpace, ::ParameterSpace) = ParameterSpace()
image(::typeof(add_bar), ::ParameterSpace, ::ParameterSpace) = ParameterSpace()

# Sequence spaces

for f ∈ (:+, :add_bar, :*, :mul_bar)
    @eval image(::typeof($f), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
        TensorSpace(map((s₁ᵢ, s₂ᵢ) -> image($f, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))
end

function image(::typeof(^), s::SequenceSpace, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return s
    n == 1 && return s
    n == 2 && return image(*, s, s)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        s = image(*, s, s)
    end
    new_s = s
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            s = image(*, s, s)
        end
        new_s = image(*, new_s, s)
    end
    return new_s
end

function image(::typeof(pow_bar), s::SequenceSpace, n::Int)
    n < 0 && return throw(DomainError(n, "pow_bar is only defined for positive integers"))
    n == 0 && return s
    n == 1 && return s
    n == 2 && return image(mul_bar, s, s)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        s = image(mul_bar, s, s)
    end
    new_s = s
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            s = image(mul_bar, s, s)
        end
        new_s = image(mul_bar, new_s, s)
    end
    return new_s
end

# Taylor

image(::typeof(+), s₁::Taylor, s₂::Taylor) = union(s₁, s₂)
image(::typeof(add_bar), s₁::Taylor, s₂::Taylor) = intersect(s₁, s₂)

image(::typeof(*), s₁::Taylor, s₂::Taylor) = Taylor(order(s₁) + order(s₂))
image(::typeof(mul_bar), s₁::Taylor, s₂::Taylor) = intersect(s₁, s₂)

# Fourier

image(::typeof(+), s₁::Fourier, s₂::Fourier) = union(s₁, s₂)
image(::typeof(add_bar), s₁::Fourier, s₂::Fourier) = intersect(s₁, s₂)

function image(::typeof(*), s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    ω₁ = frequency(s₁)
    ω₂ = frequency(s₂)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    R = promote_type(T, S)
    return Fourier(order(s₁) + order(s₂), convert(R, ω₁))
end
image(::typeof(mul_bar), s₁::Fourier, s₂::Fourier) = intersect(s₁, s₂)

# Chebyshev

image(::typeof(+), s₁::Chebyshev, s₂::Chebyshev) = union(s₁, s₂)
image(::typeof(add_bar), s₁::Chebyshev, s₂::Chebyshev) = intersect(s₁, s₂)

image(::typeof(*), s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(order(s₁) + order(s₂))
image(::typeof(mul_bar), s₁::Chebyshev, s₂::Chebyshev) = intersect(s₁, s₂)

# Cartesian spaces

for f ∈ (:+, :add_bar)
    @eval begin
        function image(::typeof($f), s₁::CartesianPower, s₂::CartesianPower)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
            return CartesianPower(image($f, space(s₁), space(s₂)), n)
        end
        function image(::typeof($f), s₁::CartesianProduct, s₂::CartesianProduct)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
            return CartesianProduct(map((s₁ᵢ, s₂ᵢ) -> image($f, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))
        end
        function image(::typeof($f), s₁::CartesianPower, s₂::CartesianProduct)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
            return CartesianProduct(map(s₂ᵢ -> image($f, space(s₁), s₂ᵢ), spaces(s₂)))
        end
        function image(::typeof($f), s₁::CartesianProduct, s₂::CartesianPower)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
            return CartesianProduct(map(s₁ᵢ -> image($f, s₁ᵢ, space(s₂)), spaces(s₁)))
        end
    end
end
