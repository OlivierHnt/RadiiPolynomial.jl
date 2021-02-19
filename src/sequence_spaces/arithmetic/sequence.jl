## fallback methods

Base.:+(a::Sequence) = Sequence(space(a), +(coefficients(a)))
Base.:-(a::Sequence) = Sequence(space(a), -(coefficients(a)))

Base.:*(a::Sequence, b) = Sequence(space(a), *(coefficients(a), b))
Base.:*(b, a::Sequence) = Sequence(space(a), *(b, coefficients(a)))

Base.:/(a::Sequence, b) = Sequence(space(a), /(coefficients(a), b))
Base.:\(b, a::Sequence) = Sequence(space(a), \(b, coefficients(a)))

## parameter space

Base.:+(a::Sequence{ParameterSpace}, b::Sequence{ParameterSpace}) =
    Sequence(ParameterSpace(), +(coefficients(a), coefficients(b)))

Base.:-(a::Sequence{ParameterSpace}, b::Sequence{ParameterSpace}) =
    Sequence(ParameterSpace(), -(coefficients(a), coefficients(b)))

+̄(a::Sequence{ParameterSpace}, b::Sequence{ParameterSpace}) = +(a, b)

-̄(a::Sequence{ParameterSpace}, b::Sequence{ParameterSpace}) = -(a, b)

## sequence space

function Base.:+(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_a = space(a)
    space_b = space(b)
    new_space = addition_range(space_a, space_b)
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    if space_a == space_b
        coefficients(c) .= coefficients(a) .+ coefficients(b)
    elseif space_a ⊆ space_b
        coefficients(c) .= coefficients(b)
        @inbounds for α ∈ allindices(space_a)
            c[α] += a[α]
        end
    elseif space_b ⊆ space_a
        coefficients(c) .= coefficients(a)
        @inbounds for α ∈ allindices(space_b)
            c[α] += b[α]
        end
    else
        coefficients(c) .= zero(CoefType)
        @inbounds for α ∈ allindices(space_a)
            c[α] = a[α]
        end
        @inbounds for α ∈ allindices(space_b)
            c[α] += b[α]
        end
    end
    return c
end

function Base.:-(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_a = space(a)
    space_b = space(b)
    new_space = addition_range(space_a, space_b)
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    if space_a == space_b
        coefficients(c) .= coefficients(a) .- coefficients(b)
    elseif space_a ⊆ space_b
        coefficients(c) .= (-).(coefficients(b))
        @inbounds for α ∈ allindices(space_a)
            c[α] += a[α]
        end
    elseif space_b ⊆ space_a
        coefficients(c) .= coefficients(a)
        @inbounds for α ∈ allindices(space_b)
            c[α] -= b[α]
        end
    else
        coefficients(c) .= zero(CoefType)
        @inbounds for α ∈ allindices(space_a)
            c[α] = a[α]
        end
        @inbounds for α ∈ allindices(space_b)
            c[α] -= b[α]
        end
    end
    return c
end

function +̄(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_a = space(a)
    space_b = space(b)
    new_space = addition_bar_range(space_a, space_b)
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    if space_a == space_b
        coefficients(c) .= coefficients(a) .+ coefficients(b)
    elseif space_a ⊆ space_b
        coefficients(c) .= coefficients(a)
        @inbounds for α ∈ allindices(space_a)
            c[α] += b[α]
        end
    elseif space_b ⊆ space_a
        coefficients(c) .= coefficients(b)
        @inbounds for α ∈ allindices(space_b)
            c[α] += a[α]
        end
    else
        coefficients(c) .= zero(CoefType)
        @inbounds for α ∈ allindices(addition_bar_range(space_a, new_space))
            c[α] = a[α]
        end
        @inbounds for α ∈ allindices(addition_bar_range(space_b, new_space))
            c[α] += b[α]
        end
    end
    return c
end

function -̄(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_a = space(a)
    space_b = space(b)
    new_space = addition_bar_range(space_a, space_b)
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    if space_a == space_b
        coefficients(c) .= coefficients(a) .- coefficients(b)
    elseif space_a ⊆ space_b
        coefficients(c) .= coefficients(a)
        @inbounds for α ∈ allindices(space_a)
            c[α] -= b[α]
        end
    elseif space_b ⊆ space_a
        coefficients(c) .= (-).(coefficients(b))
        @inbounds for α ∈ allindices(space_b)
            c[α] += a[α]
        end
    else
        coefficients(c) .= zero(CoefType)
        @inbounds for α ∈ allindices(addition_bar_range(space_a, new_space))
            c[α] = a[α]
        end
        @inbounds for α ∈ allindices(addition_bar_range(space_b, new_space))
            c[α] += b[α]
        end
    end
    return c
end

function Base.:+(a::Sequence{<:SequenceSpace}, b)
    space_a = space(a)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(space_a, Vector{CoefType}(undef, length(a)))
    coefficients(c) .= coefficients(a)
    @inbounds c[_constant_index(space_a)] += b
    return c
end

Base.:+(b, a::Sequence{<:SequenceSpace}) = +(a, b)

function Base.:-(a::Sequence{<:SequenceSpace}, b)
    space_a = space(a)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(space_a, Vector{CoefType}(undef, length(a)))
    coefficients(c) .= coefficients(a)
    @inbounds c[_constant_index(space_a)] -= b
    return c
end

function Base.:-(b, a::Sequence{<:SequenceSpace})
    space_a = space(a)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(space_a, Vector{CoefType}(undef, length(a)))
    coefficients(c) .= .-(coefficients(a))
    @inbounds c[_constant_index(space_a)] += b
    return c
end

+̄(a::Sequence{<:SequenceSpace}, b) = +(a, b)
+̄(b, a::Sequence{<:SequenceSpace}) = +(b, a)
-̄(a::Sequence{<:SequenceSpace}, b) = -(a, b)
-̄(b, a::Sequence{<:SequenceSpace}) = -(b, a)

## cartesian space

for (f, f̄) ∈ ((:+, :+̄), (:-, :-̄))
    @eval begin
        function Base.$f(a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
            space_a = space(a)
            space_b = space(b)
            new_space = addition_range(space_a, space_b)
            CoefType = promote_type(eltype(a), eltype(b))
            c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
            if space_a == space_b
                coefficients(c) .= ($f).(coefficients(a), coefficients(b))
            else
                @inbounds for i ∈ 1:nb_cartesian_product(new_space)
                    coefficients(component(c, i)) .= coefficients($f(component(a, i), component(b, i)))
                end
            end
            return c
        end

        function $f̄(a::Sequence{<:CartesianSpace}, b::Sequence{<:CartesianSpace})
            space_a = space(a)
            space_b = space(b)
            new_space = addition_bar_range(space_a, space_b)
            CoefType = promote_type(eltype(a), eltype(b))
            c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
            if space_a == space_b
                coefficients(c) .= ($f).(coefficients(a), coefficients(b))
            else
                @inbounds for i ∈ 1:nb_cartesian_product(new_space)
                    coefficients(component(c, i)) .= coefficients($f̄(component(a, i), component(b, i)))
                end
            end
            return c
        end

        function Base.$f(a::Sequence{<:CartesianSpace}, b::AbstractVector{T}) where {T<:Sequence}
            new_space = addition_range(space(a), space(b))
            CoefType = promote_type(eltype(a), eltype(T))
            c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
            @inbounds for i ∈ 1:nb_cartesian_product(new_space)
                coefficients(component(c, i)) .= coefficients($f(component(a, i), b[i]))
            end
            return c
        end

        function Base.$f(b::AbstractVector{T}, a::Sequence{<:CartesianSpace}) where {T<:Sequence}
            new_space = addition_range(space(a), space(b))
            CoefType = promote_type(eltype(a), eltype(T))
            c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
            @inbounds for i ∈ 1:nb_cartesian_product(new_space)
                coefficients(component(c, i)) .= coefficients($f(b[i], component(a, i)))
            end
            return c
        end

        function $f̄(a::Sequence{<:CartesianSpace}, b::AbstractVector{T}) where {T<:Sequence}
            new_space = addition_bar_range(space(a), space(b))
            CoefType = promote_type(eltype(a), eltype(T))
            c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
            @inbounds for i ∈ 1:nb_cartesian_product(new_space)
                coefficients(component(c, i)) .= coefficients($f̄(component(a, i), b[i]))
            end
            return c
        end

        function $f̄(b::AbstractVector{T}, a::Sequence{<:CartesianSpace}) where {T<:Sequence}
            new_space = addition_bar_range(space(a), space(b))
            CoefType = promote_type(eltype(a), eltype(T))
            c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
            @inbounds for i ∈ 1:nb_cartesian_product(new_space)
                coefficients(component(c, i)) .= coefficients($f̄(b[i], component(a, i)))
            end
            return c
        end

        function Base.$f(a::Sequence{<:CartesianSpace}, b::AbstractVector{T}) where {T}
            new_space = space(a)
            CoefType = promote_type(eltype(a), T)
            c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
            @inbounds for i ∈ 1:nb_cartesian_product(new_space)
                coefficients(component(c, i)) .= coefficients($f(component(a, i), b[i]))
            end
            return c
        end

        function Base.$f(b::AbstractVector{T}, a::Sequence{<:CartesianSpace}) where {T}
            new_space = space(a)
            CoefType = promote_type(eltype(a), T)
            c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
            @inbounds for i ∈ 1:nb_cartesian_product(new_space)
                coefficients(component(c, i)) .= coefficients($f(b[i], component(a, i)))
            end
            return c
        end
    end
end
