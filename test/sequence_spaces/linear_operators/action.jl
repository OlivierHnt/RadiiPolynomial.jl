@testset "Action" begin
    M₂ = [1.0 2.0 ; 3.0 4.0] # M₂ * [10, 20] = [50, 110]
    𝒯₀, 𝒯₁, 𝒯₂ = Taylor(0), Taylor(1), Taylor(2)

    @testset "call form A(a) == A*a" begin
        for (s, M, v) ∈ (
                (𝒯₁, M₂, [10.0, 20.0]),
                (Chebyshev(1), M₂, [10.0, 20.0]),
                (evensym(Fourier(1, 1.0)), M₂, [10.0, 20.0]),
                (Fourier(1, 1.5), ComplexF64[1 2 3 ; 4 5 6 ; 7 8 9], ComplexF64[1, 2, 3]),
                (𝒯₁ ⊗ Chebyshev(1), Matrix(reshape(1.0:16.0, 4, 4)), [1.0, 2.0, 3.0, 4.0]),
                (𝒯₁, ComplexF64[1+1im 0 ; 0 1-1im], [2.0, 3.0]))
            A, a = LinearOperator(s, s, M), Sequence(s, v)
            @test A(a) == A * a == Sequence(s, M * v)
        end
        @test LinearOperator(𝒯₁, 𝒯₁, M₂) * Sequence(𝒯₁, [10.0, 20.0]) == Sequence(𝒯₁, [50.0, 110.0])

        @testset "SymmetricSpace against a plain space" begin
            ℱ, es = Fourier(1, 1.0), evensym(Fourier(1, 1.0))
            A = LinearOperator(es, es, M₂)
            B = LinearOperator(ℱ, ℱ, [1.0 2.0 3.0 ; 4.0 5.0 6.0 ; 7.0 8.0 9.0])
            a = Sequence(es, [10.0, 20.0])
            # the operand whose symmetry differs from the operator is projected
            @test B * a == B * project(a, ℱ)
            @test A * project(a, ℱ) == A * a
            @test mul!(zeros(ℱ), A, project(a, ℱ), 1.0, 0.0) ≈ project(A * a, ℱ)
            @test mul!(zeros(es), B, a, 1.0, 0.0) ≈ project(B * project(a, ℱ), es)
            # the trivial symmetry group is the symmetry of a plain space
            t = Sequence(SymmetricSpace(ℱ, symmetry(ℱ)), [10.0, 20.0, 30.0])
            @test B * t ≈ B * Sequence(ℱ, [10.0, 20.0, 30.0])
        end
    end

    @testset "order adaptation" begin
        A = LinearOperator(𝒯₁, 𝒯₁, M₂)
        # the coefficients of the sequence outside domain(A) are discarded, the missing ones are zero
        @test A * Sequence(𝒯₂, [10.0, 20.0, 999.0]) == Sequence(𝒯₁, [50.0, 110.0])
        @test A * Sequence(𝒯₀, [10.0]) == Sequence(𝒯₁, [10.0, 30.0])

        ℱ₁ = Fourier(1, 2.0)
        B = LinearOperator(ℱ₁, ℱ₁, Float64[1 2 3 ; 4 5 6 ; 7 8 9])
        @test B * Sequence(Fourier(2, 2.0), [1.0, 10.0, 20.0, 30.0, 1.0]) == Sequence(ℱ₁, coefficients(B) * [10.0, 20.0, 30.0])
    end

    @testset "mul!(c, A, a, α, β) is c = α(A*a) + βc" begin
        A = LinearOperator(𝒯₁, 𝒯₁, M₂)
        a, a₃ = Sequence(𝒯₁, [10.0, 20.0]), Sequence(𝒯₂, [10.0, 20.0, 99.0])
        # a coefficient of c outside codomain(A) is only scaled by β, which zeroes it when β = 0
        for (c, x, α, β, expected) ∈ (
                ([Inf, Inf],            a,  2.0, 0.0, [100.0, 220.0]),
                ([1000.0, 2000.0],      a,  1.0, 1.0, [1050.0, 2110.0]),
                ([100.0, 200.0],        a,  3.0, 0.5, [200.0, 430.0]),
                ([1000.0, 1000.0, 7.0], a,  2.0, 0.5, [600.0, 720.0, 3.5]),
                ([1000.0, 1000.0, 7.0], a₃, 2.0, 0.5, [600.0, 720.0, 3.5]),
                ([999.0, 999.0, 7.0],   a,  2.0, 0.0, [100.0, 220.0, 0.0]),
                ([999.0, 999.0, 7.0],   a₃, 2.0, 0.0, [100.0, 220.0, 0.0]))
            c = Sequence(Taylor(length(c) - 1), c)
            @test mul!(c, A, x, α, β) == Sequence(space(c), expected) == c
        end
    end

    @testset "cartesian block action" begin
        𝒯₀², 𝒯₁² = 𝒯₀^2, 𝒯₁^2
        col = reshape([1.0, 2.0, 3.0, 4.0], 4, 1)
        M₄ = [1.0 2.0 5.0 6.0 ; 3.0 4.0 7.0 8.0 ; 9.0 10.0 13.0 14.0 ; 11.0 12.0 15.0 16.0]

        # each block of the sequence is adapted to the matching block of the domain
        for (A, b, expected) ∈ (
                (LinearOperator(𝒯₁ × 𝒯₀, 𝒯₀ × 𝒯₁, Float64[1 2 3 ; 4 5 6 ; 7 8 9]),
                    Sequence(𝒯₂ × 𝒯₀, [10.0, 20.0, 30.0, 40.0]), [170.0, 380.0, 590.0]), # M * [10, 20, 40]
                (LinearOperator(𝒯₁, 𝒯₀², M₂), Sequence(𝒯₂, [10.0, 20.0, 999.0]), [50.0, 110.0]),
                (LinearOperator(𝒯₀², 𝒯₁, M₂), Sequence(𝒯₁ × 𝒯₀, [100.0, 200.0, 300.0]), [700.0, 1500.0]), # M₂ * [100, 300]
                (LinearOperator(𝒯₀, 𝒯₁², col), Sequence(𝒯₀, [10.0]), [10.0, 20.0, 30.0, 40.0]))
            @test A(b) == A * b == Sequence(codomain(A), expected)
        end

        row = LinearOperator(𝒯₁², 𝒯₀, [1.0 2.0 3.0 4.0])
        b₄ = Sequence(𝒯₁², [10.0, 20.0, 30.0, 40.0]) # row * b₄ = 300
        for (A, b, c, α, β, expected) ∈ (
                # b is truncated blockwise to [1, 2, 4, 5]
                (LinearOperator(𝒯₁², 𝒯₁², M₄), Sequence(𝒯₂^2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), fill(100.0, 𝒯₁²), 2.0, 3.0,
                    300.0 .+ 2.0 .* (M₄ * [1.0, 2.0, 4.0, 5.0])),
                (row, b₄, Sequence(𝒯₂, [999.0, 999.0, 7.0]), 1.0, 0.0, [300.0, 0.0, 0.0]),
                (row, b₄, Sequence(𝒯₂, [1.0, 2.0, 3.0]), 1.0, 2.0, [302.0, 4.0, 6.0]),
                # only the order 0 coefficient of each block of the domain meets b
                (row, Sequence(𝒯₀², [10.0, 30.0]), Sequence(𝒯₀, [5.0]), 1.0, 2.0, [2.0 * 5.0 + 1.0 * 10.0 + 3.0 * 30.0]))
            @test mul!(c, A, b, α, β) == Sequence(space(c), expected)
        end
    end

    @testset "lazy wrapper action" begin
        A = LinearOperator(𝒯₁, 𝒯₁, M₂)
        B = LinearOperator(𝒯₁, 𝒯₁, [5.0 6.0 ; 7.0 8.0])
        a = Sequence(𝒯₁, [10.0, 20.0]) # A*a = [50, 110], B*a = [170, 230]

        for (L, ref, expected) ∈ (
                (UniformScalingOperator(2.0), 2.0 * a, [20.0, 40.0]),
                (Negate(A), -(A * a), [-50.0, -110.0]),
                (Add(A, B), A * a + B * a, [220.0, 340.0]),
                (ComposedOperator(A, B), A * (B * a), [630.0, 1430.0]))
            @test L(a) == L * a == ref == Sequence(𝒯₁, expected)
        end
        @test (true * I) * a == a
        @test (2 * I) * a == Sequence(𝒯₁, [20.0, 40.0])

        # mul! with an operator that is not a LinearOperator
        @test mul!(Sequence(𝒯₁, [100.0, 200.0]), UniformScalingOperator(2.0), a, 3.0, 0.5) == Sequence(𝒯₁, [110.0, 220.0])
        @test mul!(Sequence(𝒯₁, [1000.0, 2000.0]), Negate(A), a, 1.0, 0.0) == Sequence(𝒯₁, [-50.0, -110.0])
    end

    @testset "Matrix, Diagonal and Vector actions" begin
        𝕂 = ScalarSpace()
        b = Sequence(𝕂^3, [1.0, 2.0, 3.0])
        @test [1 2 3 ; 4 5 6] * b == Sequence(𝕂^2, [14.0, 32.0])
        @test RadiiPolynomial.LinearAlgebra.Diagonal([2.0, 3.0, 4.0]) * b == Sequence(𝕂^3, [2.0, 6.0, 12.0])

        # a Vector of Sequences and Numbers is read as a sequence on a cartesian space
        s = Sequence(𝒯₁, [1.0, 2.0])
        @test LinearOperator(𝒯₁ × 𝒯₀, 𝒯₀, [1.0 2.0 3.0]) * [s, Sequence(𝒯₀, [3.0])] == Sequence(𝒯₀, [14.0])
        @test LinearOperator(𝒯₁ × 𝕂, 𝒯₀, [1.0 2.0 3.0]) * [s, 5.0] == Sequence(𝒯₀, [20.0])
        @test UniformScalingOperator(3.0) * [s, 5.0] == Sequence(𝒯₁ × 𝕂, [3.0, 6.0, 15.0])
    end

    @testset "interval and complex coefficients through mul!" begin
        A = LinearOperator(𝒯₁, 𝒯₁, interval.(M₂))
        a = Sequence(𝒯₁, interval.([10.0, 20.0]))
        c = mul!(Sequence(𝒯₁, [interval(1000.0), interval(1000.0)]), A, a, interval(2.0), interval(0.0))
        @test issubset_interval(interval(100.0), c[0])
        @test issubset_interval(interval(220.0), c[1])
        @test complex(A) * complex(a) == Sequence(𝒯₁, ComplexF64[50.0, 110.0])
    end

end
