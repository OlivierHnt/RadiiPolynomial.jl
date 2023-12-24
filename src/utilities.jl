# safe conversion of integers

_safe_mul(x, y) = x * y
_safe_mul(x::Union{Interval{T},Complex{Interval{T}}}, y) where {T} = x * interval(T, y)
_safe_mul(x, y::Union{Interval{T},Complex{Interval{T}}}) where {T} = interval(T, x) * y

_safe_div(x, y) = x / y
_safe_div(x::Union{Interval{T},Complex{Interval{T}}}, y::Integer) where {T} = x / interval(T, y)

_safe_pow(x, y) = x ^ y
_safe_pow(x::Interval{T}, n::Integer) where {T} = x ^ interval(T, n)
_safe_pow(n::Integer, x::Interval{T}) where {T} = interval(T, n) ^ x

_safe_convert(::Type{T}, x) where {T} = convert(T, x)
_safe_convert(::Type{Interval{T}}, x) where {T} = interval(T, x)
_safe_convert(::Type{Complex{Interval{T}}}, x) where {T} = interval(T, complex(x))

_safe_iszero(x) = iszero(x)
_safe_iszero(x::Interval) = isthin(x, 0)

_safe_isone(x) = isone(x)
_safe_isone(x::Interval) = isthin(x, 1)

_safe_isequal(x, y) = x == y
_safe_isequal(x::Union{Interval,Complex{<:Interval}}, y::Union{Interval,Complex{<:Interval}}) =
    isequal_interval(x, y)

# allocation free reshaping (cf. Issue #36313)

_no_alloc_reshape(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)

# implement fast interval matrix multiplication

@inline __mul!(C, A, B, α, β) = mul!(C, A, B, α, β)

for (T, S) ∈ ((:Interval, :Interval), (:Interval, :Any), (:Any, :Interval))
    @eval function __mul!(C, A::AbstractMatrix{$T}, B::AbstractVecOrMat{$S}, α, β)
        CoefType = eltype(C)
        if _safe_iszero(α)
            if _safe_iszero(β)
                C .= zero(CoefType)
            elseif !_safe_isone(β)
                C .*= β
            end
        else
            ABinf, ABsup = __mul(A, B)
            if _safe_isone(α)
                if _safe_iszero(β)
                    C .= interval.(CoefType, ABinf, ABsup)
                elseif _safe_isone(β)
                    C .+= interval.(CoefType, ABinf, ABsup)
                else
                    C .= interval.(CoefType, ABinf, ABsup) .+ C .* β
                end
            else
                if _safe_iszero(β)
                    C .= interval.(CoefType, ABinf, ABsup) .* α
                elseif _safe_isone(β)
                    C .+= interval.(CoefType, ABinf, ABsup) .* α
                else
                    C .= interval.(CoefType, ABinf, ABsup) .* α .+ C .* β
                end
            end
        end
        return C
    end
end

for (T, S) ∈ ((:(Complex{<:Interval}), :(Complex{<:Interval})),
        (:(Complex{<:Interval}), :Complex), (:Complex, :(Complex{<:Interval})))
    @eval function __mul!(C, A::AbstractMatrix{$T}, B::AbstractVecOrMat{$S}, α, β)
        CoefType = eltype(C)
        if _safe_iszero(α)
            if _safe_iszero(β)
                C .= zero(CoefType)
            elseif !_safe_isone(β)
                C .*= β
            end
        else
            A_real, A_imag = reim(A)
            B_real, B_imag = reim(B)
            ABinf_1, ABsup_1 = __mul(A_real, B_real)
            ABinf_2, ABsup_2 = __mul(A_imag, B_imag)
            ABinf_3, ABsup_3 = __mul(A_real, B_imag)
            ABinf_4, ABsup_4 = __mul(A_imag, B_real)
            if _safe_isone(α)
                if _safe_iszero(β)
                    C .= complex.(interval.(CoefType, ABinf_1, ABsup_1) .- interval.(CoefType, ABinf_2, ABsup_2),
                                  interval.(CoefType, ABinf_3, ABsup_3) .+ interval.(CoefType, ABinf_4, ABsup_4))
                elseif _safe_isone(β)
                    C .+= complex.(interval.(CoefType, ABinf_1, ABsup_1) .- interval.(CoefType, ABinf_2, ABsup_2),
                                   interval.(CoefType, ABinf_3, ABsup_3) .+ interval.(CoefType, ABinf_4, ABsup_4))
                else
                    C .= complex.(interval.(CoefType, ABinf_1, ABsup_1) .- interval.(CoefType, ABinf_2, ABsup_2),
                                  interval.(CoefType, ABinf_3, ABsup_3) .+ interval.(CoefType, ABinf_4, ABsup_4)) .+ C .* β
                end
            else
                if _safe_iszero(β)
                    C .= complex.(interval.(CoefType, ABinf_1, ABsup_1) .- interval.(CoefType, ABinf_2, ABsup_2),
                                  interval.(CoefType, ABinf_3, ABsup_3) .+ interval.(CoefType, ABinf_4, ABsup_4)) .* α
                elseif _safe_isone(β)
                    C .+= complex.(interval.(CoefType, ABinf_1, ABsup_1) .- interval.(CoefType, ABinf_2, ABsup_2),
                                   interval.(CoefType, ABinf_3, ABsup_3) .+ interval.(CoefType, ABinf_4, ABsup_4)) .* α
                else
                    C .= complex.(interval.(CoefType, ABinf_1, ABsup_1) .- interval.(CoefType, ABinf_2, ABsup_2),
                                  interval.(CoefType, ABinf_3, ABsup_3) .+ interval.(CoefType, ABinf_4, ABsup_4)) .* α .+ C .* β
                end
            end
        end
        return C
    end
end

for (T, S) ∈ ((:(Complex{<:Interval{<:AbstractFloat}}), :Interval), (:(Complex{<:Interval{<:AbstractFloat}}), :Any), (:Complex, :Interval))
    @eval begin
        function __mul!(C, A::AbstractMatrix{$T}, B::AbstractVecOrMat{$S}, α, β)
            CoefType = eltype(C)
            if _safe_iszero(α)
                if _safe_iszero(β)
                    C .= zero(CoefType)
                elseif !_safe_isone(β)
                    C .*= β
                end
            else
                A_real, A_imag = reim(A)
                ABinf_real, ABsup_real = __mul(A_real, B)
                ABinf_imag, ABsup_imag = __mul(A_imag, B)
                if _safe_isone(α)
                    if _safe_iszero(β)
                        C .= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag))
                    elseif _safe_isone(β)
                        C .+= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag))
                    else
                        C .= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag)) .+ C .* β
                    end
                else
                    if _safe_iszero(β)
                        C .= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag)) .* α
                    elseif _safe_isone(β)
                        C .+= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag)) .* α
                    else
                        C .= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag)) .* α .+ C .* β
                    end
                end
            end
            return C
        end

        function __mul!(C, A::AbstractMatrix{$S}, B::AbstractVecOrMat{$T}, α, β)
            CoefType = eltype(C)
            if _safe_iszero(α)
                if _safe_iszero(β)
                    C .= zero(CoefType)
                elseif !_safe_isone(β)
                    C .*= β
                end
            else
                B_real, B_imag = reim(B)
                ABinf_real, ABsup_real = __mul(A, B_real)
                ABinf_imag, ABsup_imag = __mul(A, B_imag)
                if _safe_isone(α)
                    if _safe_iszero(β)
                        C .= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag))
                    elseif _safe_isone(β)
                        C .+= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag))
                    else
                        C .= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag)) .+ C .* β
                    end
                else
                    if _safe_iszero(β)
                        C .= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag)) .* α
                    elseif _safe_isone(β)
                        C .+= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag)) .* α
                    else
                        C .= complex.(interval.(CoefType, ABinf_real, ABsup_real), interval.(CoefType, ABinf_imag, ABsup_imag)) .* α .+ C .* β
                    end
                end
            end
            return C
        end
    end
end

__mul(A::AbstractMatrix{<:Interval{<:Rational}}, B::AbstractMatrix{<:Interval{<:Rational}}) = A * B

__mul(A::AbstractMatrix{<:Interval{<:Rational}}, B) = A * B

__mul(A, B::AbstractMatrix{<:Interval{<:Rational}}) = A * B

function __mul(A, B)
    mA, rA = mid.(A), radius.(A)
    mB, rB = mid.(B), radius.(B)

    mC, Γ = _mC_Γ(mA, rA, mB, rB)

    NewType = eltype(mC)

    u = eps(NewType)
    η = floatmin(NewType)
    γ = IntervalArithmetic._add_round.(
        IntervalArithmetic._mul_round.(convert(NewType, size(A, 2) + 1), eps.(Γ), RoundUp),
        IntervalArithmetic._div_round.(η, IntervalArithmetic._mul_round.(convert(NewType, 2), u, RoundUp), RoundUp),
        RoundUp)

    H = _matmul_up(IntervalArithmetic._add_round.(abs.(mA), rA, RoundUp), IntervalArithmetic._add_round.(abs.(mB), rB, RoundUp))
    rC = IntervalArithmetic._add_round.(IntervalArithmetic._sub_round.(H, Γ, RoundUp), IntervalArithmetic._mul_round.(convert(NewType, 2), γ, RoundUp), RoundUp)

    return IntervalArithmetic._sub_round.(mC, rC, RoundDown), IntervalArithmetic._add_round.(mC, rC, RoundUp)
end

function _mC_Γ(mA, rA, mB::AbstractVector, rB)
    NewType = promote_type(eltype(mA), eltype(mB))
    n = size(mA, 1)
    mC, Γ = zeros(NewType, n), zeros(NewType, n)

    Threads.@threads for l ∈ axes(mA, 2)
        @inbounds for i ∈ axes(mA, 1)
            a, c = mA[i,l], rA[i,l]
            b, d = mB[l], rB[l]
            e = sign(a)*min(abs(a), c)
            f = sign(b)*min(abs(b), d)
            p = a*b + e*f
            mC[i] += p
            Γ[i] += abs(p)
        end
    end

    return mC, Γ
end

function _mC_Γ(mA, rA, mB::AbstractMatrix, rB)
    NewType = promote_type(eltype(mA), eltype(mB))
    n = size(mA, 1)
    m = size(mB, 2)
    mC, Γ = zeros(NewType, n, m), zeros(NewType, n, m)

    Threads.@threads for j ∈ axes(mB, 2)
        for l ∈ axes(mA, 2)
            @inbounds for i ∈ axes(mA, 1)
                a, c = mA[i,l], rA[i,l]
                b, d = mB[l,j], rB[l,j]
                e = sign(a) * min(abs(a), c)
                f = sign(b) * min(abs(b), d)
                p = a*b + e*f
                mC[i,j] += p
                Γ[i,j] += abs(p)
            end
        end
    end

    return mC, Γ
end

function _matmul_up(A, B::AbstractVector)
    NewType = promote_type(eltype(A), eltype(B))
    C = zeros(NewType, size(A, 1))

    Threads.@threads for j ∈ axes(B, 2)
        for l ∈ axes(A, 2)
            @inbounds for i ∈ axes(A, 1)
                C[i,j] = IntervalArithmetic._add_round.(IntervalArithmetic._mul_round.(A[i,l], B[l,j], RoundUp), C[i,j], RoundUp)
            end
        end
    end

    return C
end

function _matmul_up(A, B::AbstractMatrix)
    NewType = promote_type(eltype(A), eltype(B))
    C = zeros(NewType, size(A, 1), size(B, 2))

    Threads.@threads for j ∈ axes(B, 2)
        for l ∈ axes(A, 2)
            @inbounds for i ∈ axes(A, 1)
                C[i,j] = IntervalArithmetic._add_round.(IntervalArithmetic._mul_round.(A[i,l], B[l,j], RoundUp), C[i,j], RoundUp)
            end
        end
    end

    return C
end
