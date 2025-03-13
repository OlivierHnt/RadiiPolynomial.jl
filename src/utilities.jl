# safe equality

_safe_isequal(x, y) = x == y
_safe_isequal(x::Union{Interval,Complex{<:Interval}}, y::Union{Interval,Complex{<:Interval}}) =
    isequal_interval(x, y)

# allocation free reshaping (cf. Issue #36313)

_no_alloc_reshape(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)

# implement fast interval matrix multiplication

@inline __mul!(C, A, B, α, β) = mul!(C, A, B, α, β)

for (T, S) ∈ ((:Interval, :Interval), (:Interval, :Any), (:Any, :Interval))
    @eval __mul!(C, A::AbstractMatrix{<:$T}, B::AbstractVecOrMat{<:$S}, α, β) =
        IntervalArithmetic._mul!(IntervalArithmetic.MatMulMode{:fast}(), C, A, B, α, β)
end

for (T, S) ∈ ((:(Complex{<:Interval}), :(Complex{<:Interval})),
        (:(Complex{<:Interval}), :Complex), (:Complex, :(Complex{<:Interval})))
    @eval __mul!(C, A::AbstractMatrix{<:$T}, B::AbstractVecOrMat{<:$S}, α, β) =
        IntervalArithmetic._mul!(IntervalArithmetic.MatMulMode{:fast}(), C, A, B, α, β)
end

for (T, S) ∈ ((:(Complex{<:Interval}), :Interval), (:(Complex{<:Interval}), :Any), (:Complex, :Interval))
    @eval begin
        __mul!(C, A::AbstractMatrix{<:$T}, B::AbstractVecOrMat{<:$S}, α, β) =
            IntervalArithmetic._mul!(IntervalArithmetic.MatMulMode{:fast}(), C, A, B, α, β)

        __mul!(C, A::AbstractMatrix{<:$S}, B::AbstractVecOrMat{<:$T}, α, β) =
            IntervalArithmetic._mul!(IntervalArithmetic.MatMulMode{:fast}(), C, A, B, α, β)
    end
end
