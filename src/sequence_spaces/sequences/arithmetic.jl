# fallback methods

codomain(::typeof(-), s₁::VectorSpace, s₂::VectorSpace) = codomain(+, s₁, s₂)

# Parameter space

codomain(::typeof(+), ::ParameterSpace, ::ParameterSpace) = ParameterSpace()

# Sequence spaces

codomain(::typeof(+), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((s₁ᵢ, s₂ᵢ) -> codomain(+, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))

for T ∈ (:Taylor, :Fourier, :Chebyshev, :CosFourier, :SinFourier)
    @eval begin
        codomain(::typeof(+), s₁::$T, s₂::$T) = union(s₁, s₂)
    end
end

codomain(::typeof(+), s₁::CosFourier, s₂::SinFourier) = codomain(+, desymmetrize(s₁), desymmetrize(s₂))
codomain(::typeof(+), s₁::SinFourier, s₂::CosFourier) = codomain(+, desymmetrize(s₁), desymmetrize(s₂))

# Cartesian spaces

for f ∈ (:+,)
    @eval begin
        function codomain(::typeof($f), s₁::CartesianPower, s₂::CartesianPower)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
            return CartesianPower(codomain($f, space(s₁), space(s₂)), n)
        end
        function codomain(::typeof($f), s₁::CartesianProduct, s₂::CartesianProduct)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
            return CartesianProduct(map((s₁ᵢ, s₂ᵢ) -> codomain($f, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))
        end
        function codomain(::typeof($f), s₁::CartesianPower, s₂::CartesianProduct)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
            return CartesianProduct(map(s₂ᵢ -> codomain($f, space(s₁), s₂ᵢ), spaces(s₂)))
        end
        function codomain(::typeof($f), s₁::CartesianProduct, s₂::CartesianPower)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
            return CartesianProduct(map(s₁ᵢ -> codomain($f, s₁ᵢ, space(s₂)), spaces(s₁)))
        end
    end
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
            new_space = codomain($f, space(a), space(b))
            CoefType = promote_type(eltype(a), eltype(b))
            c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
            $_f!(c, a, b)
            return c
        end
        $f(a::InfiniteSequence, b::InfiniteSequence) =
            InfiniteSequence($f(sequence(a), sequence(b)), sequence_error(a) + sequence_error(b), banachspace(a) ∩ banachspace(b))

        function $f!(c::Sequence, a::Sequence, b::Sequence)
            op = $f
            space_c = space(c)
            new_space = codomain(op, space(a), space(b))
            _iscompatible(space_c, new_space) || return throw(ArgumentError("spaces must be compatible: c has space $space_c, a$(op)b has space $new_space"))
            $_f!(c, a, b)
            return c
        end

        function $rf!(a::Sequence, b::Sequence)
            op = $f
            space_a = space(a)
            new_space = codomain(op, space_a, space(b))
            _iscompatible(space_a, new_space) || return throw(ArgumentError("spaces must be compatible: a has space $space_a, a$(op)b has space $new_space"))
            $_rf!(a, b)
            return a
        end

        function $lf!(a::Sequence, b::Sequence)
            op = $f
            space_b = space(b)
            new_space = codomain(op, space(a), space_b)
            _iscompatible(space_b, new_space) || return throw(ArgumentError("spaces must be compatible: b has space $space_b, a$(op)b has space $new_space"))
            $_lf!(a, b)
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
        @inbounds for α ∈ indices(space_b ∩ space_c)
            c[α] += b[α]
        end
    elseif space_b == space_c
        coefficients(c) .= coefficients(b)
        @inbounds for α ∈ indices(space_a ∩ space_c)
            c[α] += a[α]
        end
    else
        coefficients(c) .= zero(eltype(c))
        @inbounds for α ∈ indices(space_a ∩ space_c)
            c[α] = a[α]
        end
        @inbounds for α ∈ indices(space_b ∩ space_c)
            c[α] += b[α]
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
        @inbounds for α ∈ indices(space_b ∩ space_c)
            c[α] -= b[α]
        end
    elseif space_b == space_c
        coefficients(c) .= (-).(coefficients(b))
        @inbounds for α ∈ indices(space_a ∩ space_c)
            c[α] += a[α]
        end
    else
        coefficients(c) .= zero(eltype(c))
        @inbounds for α ∈ indices(space_a ∩ space_c)
            c[α] = a[α]
        end
        @inbounds for α ∈ indices(space_b ∩ space_c)
            c[α] -= b[α]
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
        @inbounds for α ∈ indices(space_a ∩ space_b)
            a[α] += b[α]
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
        @inbounds for α ∈ indices(space_a ∩ space_b)
            a[α] -= b[α]
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
        @inbounds for α ∈ indices(space_a ∩ space_b)
            b[α] += a[α]
        end
    end
    return b
end

for (f, _f!, _rf!, _lf!) ∈ ((:+, :_add!, :_radd!, :_ladd!), (:-, :_sub!, :_rsub!, :_lsub!))
    @eval begin
        function $_f!(c::Sequence{<:CartesianSpace}, a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
            if space(a) == space(b)
                coefficients(c) .= ($f).(coefficients(a), coefficients(b))
            else
                @inbounds for i ∈ 1:nspaces(space(c))
                    $_f!(component(c, i), component(a, i), component(b, i))
                end
            end
            return c
        end
        function $_f!(c::Sequence{CartesianProduct{T}}, a::Sequence{<:CartesianProduct}, b::Sequence{<:CartesianProduct}) where {N,T<:NTuple{N,VectorSpace}}
            if space(a) == space(b)
                coefficients(c) .= ($f).(coefficients(a), coefficients(b))
            else
                @inbounds $_f!(component(c, 1), component(a, 1), component(b, 1))
                @inbounds $_f!(component(c, 2:N), component(a, 2:N), component(b, 2:N))
            end
            return c
        end
        $_f!(c::Sequence{CartesianProduct{T}}, a::Sequence{<:CartesianProduct}, b::Sequence{<:CartesianProduct}) where {T<:Tuple{VectorSpace}} =
            @inbounds $_f!(component(c, 1), component(a, 1), component(b, 1))

        function $_rf!(a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
            space_a = space(a)
            if space_a == space(b)
                A = coefficients(a)
                A .= ($f).(A, coefficients(b))
            else
                @inbounds for i ∈ 1:nspaces(space_a)
                    $_rf!(component(a, i), component(b, i))
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
                @inbounds $_rf!(component(a, 1), component(b, 1))
                @inbounds $_rf!(component(a, 2:N), component(b, 2:N))
            end
            return a
        end
        $_rf!(a::Sequence{CartesianProduct{T}}, b::Sequence{<:CartesianProduct}) where {T<:Tuple{VectorSpace}} =
            @inbounds $_rf!(component(a, 1), component(b, 1))

        function $_lf!(a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
            space_a = space(a)
            if space_a == space(b)
                B = coefficients(b)
                B .= ($f).(coefficients(a), B)
            else
                @inbounds for i ∈ 1:nspaces(space_a)
                    $_lf!(component(a, i), component(b, i))
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
                @inbounds $_lf!(component(a, 1), component(b, 1))
                @inbounds $_lf!(component(a, 2:N), component(b, 2:N))
            end
            return b
        end
        $_lf!(a::Sequence{<:CartesianProduct}, b::Sequence{CartesianProduct{T}}) where {T<:Tuple{VectorSpace}} =
            @inbounds $_lf!(component(a, 1), component(b, 1))
    end
end

# Parameter space

Base.:+(a::Sequence{ParameterSpace}, b::Number) = @inbounds Sequence(space(a), [a[1] + b])
Base.:+(b::Number, a::Sequence{ParameterSpace}) = @inbounds Sequence(space(a), [b + a[1]])

Base.:-(a::Sequence{ParameterSpace}, b::Number) = @inbounds Sequence(space(a), [a[1] - b])
Base.:-(b::Number, a::Sequence{ParameterSpace}) = @inbounds Sequence(space(a), [b - a[1]])

radd!(a::Sequence{ParameterSpace}, b::Number) = _radd!(a, b)
ladd!(b::Number, a::Sequence{ParameterSpace}) = _radd!(a, b)

rsub!(a::Sequence{ParameterSpace}, b::Number) = _radd!(a, -b)
function lsub!(b::Number, a::Sequence{ParameterSpace})
    @inbounds a[1] = b - a[1]
    return a
end

function _radd!(a::Sequence{ParameterSpace}, b::Number)
    @inbounds a[1] += b
    return a
end

# Sequence spaces

function Base.:+(a::Sequence{<:SequenceSpace}, b::Number)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(space(a), Vector{CoefType}(undef, length(a)))
    coefficients(c) .= coefficients(a)
    _radd!(c, b)
    return c
end
Base.:+(b::Number, a::Sequence{<:SequenceSpace}) = +(a, b)
Base.:+(a::InfiniteSequence, b::Number) = InfiniteSequence(sequence(a) + b, sequence_error(a), banachspace(a))
Base.:+(b::Number, a::InfiniteSequence) = InfiniteSequence(b + sequence(a), sequence_error(a), banachspace(a))

Base.:-(a::Sequence{<:SequenceSpace}, b::Number) = +(a, -b)
function Base.:-(b::Number, a::Sequence{<:SequenceSpace})
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(space(a), Vector{CoefType}(undef, length(a)))
    coefficients(c) .= (-).(coefficients(a))
    _radd!(c, b)
    return c
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
        aᵢ = component(a, i)
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
        aᵢ = component(a, i)
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
