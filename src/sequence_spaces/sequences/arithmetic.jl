# fallback methods

codomain(::typeof(-), s₁::VectorSpace, s₂::VectorSpace) = codomain(+, s₁, s₂)

# Scalar space

codomain(::typeof(+), ::ScalarSpace, ::ScalarSpace) = ScalarSpace()

# Sequence spaces

codomain(::typeof(+), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((s₁ᵢ, s₂ᵢ) -> codomain(+, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))

for T ∈ (:Taylor, :Fourier, :Chebyshev)
    @eval codomain(::typeof(+), s₁::$T, s₂::$T) = union(s₁, s₂)
end

function codomain(::typeof(+), s₁::SymmetricSpace, s₂::SymmetricSpace)
    V = codomain(+, desymmetrize(s₁), desymmetrize(s₂))
    G = intersect(symmetry(s₁), symmetry(s₂))
    return SymmetricSpace(V, G)
end
codomain(::typeof(+), s₁::SymmetricSpace, s₂::NoSymSpace) = codomain(+, s₁, SymmetricSpace(s₂))
codomain(::typeof(+), s₁::NoSymSpace, s₂::SymmetricSpace) = codomain(+, SymmetricSpace(s₁), s₂)

# Cartesian spaces

function codomain(::typeof(+), s₁::CartesianPower, s₂::CartesianPower)
    n = nspaces(s₁)
    m = nspaces(s₂)
    n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
    return CartesianPower(codomain(+, space(s₁), space(s₂)), n)
end
function codomain(::typeof(+), s₁::CartesianProduct, s₂::CartesianProduct)
    n = nspaces(s₁)
    m = nspaces(s₂)
    n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
    return CartesianProduct(map((s₁ᵢ, s₂ᵢ) -> codomain(+, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))
end
function codomain(::typeof(+), s₁::CartesianPower, s₂::CartesianProduct)
    n = nspaces(s₁)
    m = nspaces(s₂)
    n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
    return CartesianProduct(map(s₂ᵢ -> codomain(+, space(s₁), s₂ᵢ), spaces(s₂)))
end
function codomain(::typeof(+), s₁::CartesianProduct, s₂::CartesianPower)
    n = nspaces(s₁)
    m = nspaces(s₂)
    n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
    return CartesianProduct(map(s₁ᵢ -> codomain(+, s₁ᵢ, space(s₂)), spaces(s₁)))
end

#

Base.:+(a::Sequence) = Sequence(space(a), +(coefficients(a)))
Base.:+(a::InfiniteSequence) = InfiniteSequence(+(sequence(a)), sequence_error(a), banachspace(a))

Base.:-(a::Sequence) = Sequence(space(a), -(coefficients(a)))
Base.:-(a::InfiniteSequence) = InfiniteSequence(-(sequence(a)), sequence_error(a), banachspace(a))

Base.:*(a::Sequence, b::Number) = Sequence(space(a), *(coefficients(a), b))
Base.:*(b::Number, a::Sequence) = Sequence(space(a), *(b, coefficients(a)))
Base.:*(a::InfiniteSequence, b::Number) = InfiniteSequence(sequence(a) * b, sequence_error(a) * abs(b), banachspace(a))
Base.:*(a::Number, b::InfiniteSequence) = InfiniteSequence(a * sequence(b), abs(a) * sequence_error(b), banachspace(b))

Base.:/(a::Sequence, b::Number) = Sequence(space(a), /(coefficients(a), b))
Base.:\(b::Number, a::Sequence) = Sequence(space(a), \(b, coefficients(a)))
Base.:/(a::InfiniteSequence, b::Number) = InfiniteSequence(sequence(a) / b, sequence_error(a) / abs(b), banachspace(a))
Base.:\(b::Number, a::InfiniteSequence) = InfiniteSequence(b \ sequence(a), abs(b) \ sequence_error(a), banachspace(a))

rmul!(a::Sequence, b::Number) = Sequence(space(a), rmul!(coefficients(a), b))
lmul!(b::Number, a::Sequence) = Sequence(space(a), lmul!(b, coefficients(a)))
rmul!(a::InfiniteSequence, b::Number) = InfiniteSequence(rmul!(sequence(a), b), sequence_error(a) * abs(b), banachspace(a))
lmul!(b::Number, a::InfiniteSequence) = InfiniteSequence(lmul!(b, sequence(a)), abs(b) * sequence_error(a), banachspace(a))

rdiv!(a::Sequence, b::Number) = Sequence(space(a), rdiv!(coefficients(a), b))
ldiv!(b::Number, a::Sequence) = Sequence(space(a), ldiv!(b, coefficients(a)))
rdiv!(a::InfiniteSequence, b::Number) = InfiniteSequence(rdiv!(sequence(a), b), sequence_error(a) / abs(b), banachspace(a))
ldiv!(b::Number, a::InfiniteSequence) = InfiniteSequence(ldiv!(b, sequence(a)), abs(b) \ sequence_error(a), banachspace(a))

for (f, f!, rf!, lf!, _f!, _rf!, _lf!) ∈ ((:(Base.:+), :add!, :radd!, :ladd!, :_add!, :_radd!, :_ladd!),
        (:(Base.:-), :sub!, :rsub!, :lsub!, :_sub!, :_rsub!, :_lsub!))
    @eval begin
        function $f(a::Sequence, b::Sequence)
            psa, psb = _promote_space(space(a), space(b))
            c = zeros(promote_type(eltype(a), eltype(b)), codomain($f, psa, psb))
            $_f!(c, Sequence(psa, coefficients(a)), Sequence(psb, coefficients(b)))
            return c
        end
        $f(a::InfiniteSequence, b::InfiniteSequence) =
            InfiniteSequence($f(sequence(a), sequence(b)), sequence_error(a) + sequence_error(b), banachspace(a) ∩ banachspace(b))

        function $f!(c::Sequence, a::Sequence, b::Sequence)
            op = $f
            psa, psb = _promote_space(space(a), space(b))
            sop = codomain(op, psa, psb)
            psc, psop = _promote_space(space(c), sop)
            _iscompatible(psc, psop) || return throw(ArgumentError("spaces must be compatible: c has space $(space(c)), a $(op) b has space $sop"))
            $_f!(Sequence(psc, coefficients(c)), Sequence(psa, coefficients(a)), Sequence(psb, coefficients(b)))
            return c
        end

        function $rf!(a::Sequence, b::Sequence)
            psa, psb = _promote_space(space(a), space(b))
            _iscompatible(psa, psb) || return throw(ArgumentError("spaces must be compatible"))
            $_rf!(Sequence(psa, coefficients(a)), Sequence(psb, coefficients(b)))
            return a
        end

        function $lf!(a::Sequence, b::Sequence)
            psa, psb = _promote_space(space(a), space(b))
            _iscompatible(psa, psb) || return throw(ArgumentError("spaces must be compatible"))
            $_lf!(Sequence(psa, coefficients(a)), Sequence(psb, coefficients(b)))
            return b
        end
    end
end

function _add!(c::Sequence, a::Sequence, b::Sequence)
    space_a = space(a)
    space_b = space(b)
    space_c = space(c)
    if space_a == space_b == space_c
        coefficients(c) .= coefficients(a) .+ coefficients(b)
    elseif space_a == space_c
        coefficients(c) .= coefficients(a)
        @inbounds for α ∈ indices(space_c)
            c[α] += getcoefficient(b, (space_c, α))
        end
    elseif space_b == space_c
        coefficients(c) .= coefficients(b)
        @inbounds for α ∈ indices(space_c)
            c[α] += getcoefficient(a, (space_c, α))
        end
    else
        @inbounds for α ∈ indices(space_c)
            c[α] = getcoefficient(a, (space_c, α)) + getcoefficient(b, (space_c, α))
        end
    end
    return c
end

function _sub!(c::Sequence, a::Sequence, b::Sequence)
    space_a = space(a)
    space_b = space(b)
    space_c = space(c)
    if space_a == space_b == space_c
        coefficients(c) .= coefficients(a) .- coefficients(b)
    elseif space_a == space_c
        coefficients(c) .= coefficients(a)
        @inbounds for α ∈ indices(space_c)
            c[α] -= getcoefficient(b, (space_c, α))
        end
    elseif space_b == space_c
        coefficients(c) .= (-).(coefficients(b))
        @inbounds for α ∈ indices(space_c)
            c[α] += getcoefficient(a, (space_c, α))
        end
    else
        coefficients(c) .= zero(eltype(c))
        @inbounds for α ∈ indices(space_c)
            c[α] = getcoefficient(a, (space_c, α)) - getcoefficient(b, (space_c, α))
        end
    end
    return c
end

function _radd!(a::Sequence, b::Sequence)
    space_a = space(a)
    space_b = space(b)
    if space_a == space_b
        coefficients(a) .+= coefficients(b)
    else
        @inbounds for α ∈ indices(space_a)
            a[α] += getcoefficient(b, (space_a, α))
        end
    end
    return a
end

_ladd!(a::Sequence, b::Sequence) = _radd!(b, a)

function _rsub!(a::Sequence, b::Sequence)
    space_a = space(a)
    space_b = space(b)
    if space_a == space_b
        coefficients(a) .-= coefficients(b)
    else
        @inbounds for α ∈ indices(space_a)
            a[α] -= getcoefficient(b, (space_a, α))
        end
    end
    return a
end

function _lsub!(a::Sequence, b::Sequence)
    space_a = space(a)
    space_b = space(b)
    B = coefficients(b)
    if space_a == space_b
        B .= coefficients(a) .- B
    else
        B .= (-).(B)
        @inbounds for α ∈ indices(space_b)
            b[α] += getcoefficient(a, (space_b, α))
        end
    end
    return b
end

# Cartesian spaces

for (f, _f!, _rf!, _lf!) ∈ ((:+, :_add!, :_radd!, :_ladd!), (:-, :_sub!, :_rsub!, :_lsub!))
    @eval begin
        function $_f!(c::Sequence{<:CartesianSpace}, a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
            if space(a) == space(b)
                coefficients(c) .= ($f).(coefficients(a), coefficients(b))
            else
                @inbounds for i ∈ 1:nspaces(space(c))
                    $_f!(block(c, i), block(a, i), block(b, i))
                end
            end
            return c
        end
        function $_f!(c::Sequence{CartesianProduct{T}}, a::Sequence{<:CartesianProduct}, b::Sequence{<:CartesianProduct}) where {N,T<:NTuple{N,VectorSpace}}
            if space(a) == space(b)
                coefficients(c) .= ($f).(coefficients(a), coefficients(b))
            else
                @inbounds $_f!(block(c, 1), block(a, 1), block(b, 1))
                @inbounds $_f!(block(c, 2:N), block(a, 2:N), block(b, 2:N))
            end
            return c
        end
        $_f!(c::Sequence{CartesianProduct{T}}, a::Sequence{<:CartesianProduct}, b::Sequence{<:CartesianProduct}) where {T<:Tuple{VectorSpace}} =
            @inbounds $_f!(block(c, 1), block(a, 1), block(b, 1))

        function $_rf!(a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
            space_a = space(a)
            if space_a == space(b)
                A = coefficients(a)
                A .= ($f).(A, coefficients(b))
            else
                @inbounds for i ∈ 1:nspaces(space_a)
                    $_rf!(block(a, i), block(b, i))
                end
            end
            return a
        end
        function $_rf!(a::Sequence{CartesianProduct{T}}, b::Sequence{<:CartesianProduct}) where {N,T<:NTuple{N,VectorSpace}}
            space_a = space(a)
            if space_a == space(b)
                A = coefficients(a)
                A .= ($f).(A, coefficients(b))
            else
                @inbounds $_rf!(block(a, 1), block(b, 1))
                @inbounds $_rf!(block(a, 2:N), block(b, 2:N))
            end
            return a
        end
        $_rf!(a::Sequence{CartesianProduct{T}}, b::Sequence{<:CartesianProduct}) where {T<:Tuple{VectorSpace}} =
            @inbounds $_rf!(block(a, 1), block(b, 1))

        function $_lf!(a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
            space_a = space(a)
            if space_a == space(b)
                B = coefficients(b)
                B .= ($f).(coefficients(a), B)
            else
                @inbounds for i ∈ 1:nspaces(space_a)
                    $_lf!(block(a, i), block(b, i))
                end
            end
            return b
        end
        function $_lf!(a::Sequence{<:CartesianProduct}, b::Sequence{CartesianProduct{T}}) where {N,T<:NTuple{N,VectorSpace}}
            space_a = space(a)
            if space_a == space(b)
                B = coefficients(b)
                B .= ($f).(coefficients(a), B)
            else
                @inbounds $_lf!(block(a, 1), block(b, 1))
                @inbounds $_lf!(block(a, 2:N), block(b, 2:N))
            end
            return b
        end
        $_lf!(a::Sequence{<:CartesianProduct}, b::Sequence{CartesianProduct{T}}) where {T<:Tuple{VectorSpace}} =
            @inbounds $_lf!(block(a, 1), block(b, 1))
    end
end

# Scalar space

Base.:+(a::Sequence{ScalarSpace}, b::Number) = @inbounds Sequence(space(a), [a[1] + b])
Base.:+(b::Number, a::Sequence{ScalarSpace}) = @inbounds Sequence(space(a), [b + a[1]])

Base.:-(a::Sequence{ScalarSpace}, b::Number) = @inbounds Sequence(space(a), [a[1] - b])
Base.:-(b::Number, a::Sequence{ScalarSpace}) = @inbounds Sequence(space(a), [b - a[1]])

radd!(a::Sequence{ScalarSpace}, b::Number) = _radd!(a, b)
ladd!(b::Number, a::Sequence{ScalarSpace}) = _radd!(a, b)

rsub!(a::Sequence{ScalarSpace}, b::Number) = _radd!(a, -b)
function lsub!(b::Number, a::Sequence{ScalarSpace})
    @inbounds a[1] = b - a[1]
    return a
end

function _radd!(a::Sequence{ScalarSpace}, b::Number)
    @inbounds a[1] += b
    return a
end

# Sequence spaces

_sym_with_cst_coef(s::Group{N,T}) where {N,T} =
    unsafe_group!(Set{GroupElement{N,T}}(g for g ∈ elements(s) if isone(g.coef_action.amplitude)))

function Base.:+(a::Sequence{<:NoSymSpace}, b::Number)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(space(a), Vector{CoefType}(undef, length(a)))
    coefficients(c) .= coefficients(a)
    _radd!(c, b)
    return c
end
function Base.:+(a::Sequence{<:SymmetricSpace}, b::Number)
    dsa = desymmetrize(space(a))
    da = Projection(dsa) * a
    space_c = SymmetricSpace(dsa, _sym_with_cst_coef(symmetry(space(a))))
    return Projection(space_c) * (da + b)
end
Base.:+(b::Number, a::Sequence{<:SequenceSpace}) = +(a, b)
Base.:+(a::InfiniteSequence, b::Number) = InfiniteSequence(sequence(a) + b, sequence_error(a), banachspace(a))
Base.:+(b::Number, a::InfiniteSequence) = InfiniteSequence(b + sequence(a), sequence_error(a), banachspace(a))

Base.:-(a::Sequence{<:SequenceSpace}, b::Number) = +(a, -b)
function Base.:-(b::Number, a::Sequence{<:NoSymSpace})
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(space(a), Vector{CoefType}(undef, length(a)))
    coefficients(c) .= (-).(coefficients(a))
    _radd!(c, b)
    return c
end
function Base.:-(b::Number, a::Sequence{<:SymmetricSpace})
    dsa = desymmetrize(space(a))
    da = Projection(dsa) * a
    space_c = SymmetricSpace(dsa, _sym_with_cst_coef(symmetry(space(a))))
    return Projection(space_c) * (b - da)
end
Base.:-(a::InfiniteSequence, b::Number) = InfiniteSequence(sequence(a) - b, sequence_error(a), banachspace(a))
Base.:-(b::Number, a::InfiniteSequence) = InfiniteSequence(b - sequence(a), sequence_error(a), banachspace(a))

radd!(a::Sequence{<:SequenceSpace}, b::Number) = _radd!(a, b)
ladd!(b::Number, a::Sequence{<:SequenceSpace}) = _radd!(a, b)
radd!(a::InfiniteSequence{<:SequenceSpace}, b::Number) = InfiniteSequence(radd!(sequence(a), b), sequence_error(a), banachspace(a))
ladd!(b::Number, a::InfiniteSequence{<:SequenceSpace}) = InfiniteSequence(ladd!(b, sequence(a)), sequence_error(a), banachspace(a))

rsub!(a::Sequence{<:SequenceSpace}, b::Number) = _radd!(a, -b)
function lsub!(b::Number, a::Sequence{<:SequenceSpace})
    A = coefficients(a)
    A .= (-).(A)
    _radd!(a, b)
    return a
end
rsub!(a::InfiniteSequence{<:SequenceSpace}, b::Number) = InfiniteSequence(rsub!(sequence(a), b), sequence_error(a), banachspace(a))
rsub!(b::Number, a::InfiniteSequence{<:SequenceSpace}) = InfiniteSequence(lsub!(b, sequence(a)), sequence_error(a), banachspace(a))

function _radd!(a::Sequence{<:SequenceSpace}, b::Number)
    @inbounds a[_findindex_constant(space(a))] += b
    return a
end

# Cartesian spaces

function Base.:+(a::Sequence{<:CartesianSpace}, b::AbstractVector{T}) where {T<:Number}
    space_a = space(a)
    Base.OneTo(_deep_nspaces(space_a)) == eachindex(b) || return throw(ArgumentError)
    CoefType = promote_type(eltype(a), T)
    c = Sequence(space_a, Vector{CoefType}(undef, length(a)))
    coefficients(c) .= coefficients(a)
    _radd!(c, b)
    return c
end
Base.:+(b::AbstractVector{T}, a::Sequence{<:CartesianSpace}) where {T<:Number} = +(a, b)

function Base.:-(a::Sequence{<:CartesianSpace}, b::AbstractVector{T}) where {T<:Number}
    space_a = space(a)
    Base.OneTo(_deep_nspaces(space_a)) == eachindex(b) || return throw(ArgumentError)
    CoefType = promote_type(eltype(a), T)
    c = Sequence(space_a, Vector{CoefType}(undef, length(a)))
    coefficients(c) .= coefficients(a)
    _rsub!(c, b)
    return c
end
function Base.:-(b::AbstractVector{T}, a::Sequence{<:CartesianSpace}) where {T<:Number}
    space_a = space(a)
    Base.OneTo(_deep_nspaces(space_a)) == eachindex(b) || return throw(ArgumentError)
    CoefType = promote_type(eltype(a), T)
    c = Sequence(space_a, Vector{CoefType}(undef, length(a)))
    coefficients(c) .= (-).(coefficients(a))
    _radd!(c, b)
    return c
end

function radd!(a::Sequence{<:CartesianSpace}, b::AbstractVector{<:Number})
    Base.OneTo(_deep_nspaces(space(a))) == eachindex(b) || return throw(ArgumentError)
    _radd!(a, b)
    return a
end
ladd!(b::AbstractVector{<:Number}, a::Sequence{<:CartesianSpace}) = radd!(a, b)

function rsub!(a::Sequence{<:CartesianSpace}, b::AbstractVector{<:Number})
    Base.OneTo(_deep_nspaces(space(a))) == eachindex(b) || return throw(ArgumentError)
    _rsub!(a, b)
    return a
end
function lsub!(b::AbstractVector{<:Number}, a::Sequence{<:CartesianSpace})
    Base.OneTo(_deep_nspaces(space(a))) == eachindex(b) || return throw(ArgumentError)
    A = coefficients(a)
    A .= (-).(A)
    _radd!(a, b)
    return a
end

function _radd!(a::Sequence{<:CartesianSpace}, b::AbstractVector{<:Number})
    k = 0
    @inbounds for i ∈ 1:nspaces(space(a))
        aᵢ = block(a, i)
        space_aᵢ = space(aᵢ)
        if space_aᵢ isa CartesianSpace
            k_ = k + 1
            k += _deep_nspaces(space_aᵢ)
            _radd!(aᵢ, view(b, k_:k))
        else
            k += 1
            _radd!(aᵢ, b[k])
        end
    end
    return a
end

function _rsub!(a::Sequence{<:CartesianSpace}, b::AbstractVector{<:Number})
    k = 0
    @inbounds for i ∈ 1:nspaces(space(a))
        aᵢ = block(a, i)
        space_aᵢ = space(aᵢ)
        if space_aᵢ isa CartesianSpace
            k_ = k + 1
            k += _deep_nspaces(space_aᵢ)
            _rsub!(aᵢ, view(b, k_:k))
        else
            k += 1
            _radd!(aᵢ, -b[k])
        end
    end
    return a
end
