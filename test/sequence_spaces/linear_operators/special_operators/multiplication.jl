@testset "Multiplication" begin
    @testset "sequence accessor" begin
        a = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        ℳ = Multiplication(a)
        @test sequence(ℳ) == a
        @test sequence(ℳ) === a # no copy

        Mi = interval(Float64, ℳ)
        @test sequence(Mi) == interval.(a)
        @test sequence(interval(ℳ)) == interval(a)
    end

    @testset "arithmetic on Multiplication operators" begin
        a = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        ℳ = Multiplication(a)
        @test (+ℳ).sequence == a
        @test (-ℳ).sequence == -a
        @test (ℳ^0).sequence == a^0
        @test (ℳ^2).sequence == (ℳ*ℳ).sequence == Multiplication(a*a).sequence == a*a
        @test (2ℳ/1).sequence == (ℳ+ℳ).sequence == (ℳ-(-ℳ)).sequence ==
            Multiplication(a+a).sequence == a+a
        # Multiplication ± / * / \ Number acts entrywise on the underlying sequence
        @test (ℳ+1).sequence == a+1
        @test (1+ℳ).sequence == 1+a
        @test (ℳ-1).sequence == a-1
        @test (1-ℳ).sequence == 1-a
        @test (3ℳ).sequence == 3a
        @test (ℳ*3).sequence == a*3
        @test (ℳ/2).sequence == a/2
        @test (2\ℳ).sequence == 2\a
    end

    @testset "action == convolution" begin
        @testset "Taylor" begin
            a = Sequence(Taylor(2), [1.0, -1.0, 1.0]) # 1 - x + x²
            b = Sequence(Taylor(1), [2.0, 3.0])       # 2 + 3x
            ℳ = Multiplication(a)
            # (1 - x + x²)(2 + 3x) = 2 + (3-2)x + (-3+2)x² + 3x³ = 2 + x - x² + 3x³
            expected = Sequence(Taylor(3), [2.0, 1.0, -1.0, 3.0])
            out = Sequence(Taylor(3), fill(Inf, 4))
            @test ℳ(b) == b*a == sequence(ℳ)*b == mul!(out, ℳ, b) == expected
        end

        @testset "Fourier" begin
            a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5]) # cos(t)
            ℳ = Multiplication(a)
            # cos²(t) = 1/2 + 1/2 cos(2t)
            expected = Sequence(Fourier(2, 1.0), [0.25, 0.0, 0.5, 0.0, 0.25])
            out = Sequence(Fourier(2, 1.0), fill(Inf, 5))
            @test ℳ(a) == a*a == mul!(out, ℳ, a) == expected
        end

        @testset "Chebyshev" begin
            a = Sequence(Chebyshev(2), [1.0, 0.5, 0.5]) # T₀ + ½T₁ + ½T₂
            ℳ = Multiplication(a)
            # direct discrete (folded) convolution: cₖ = Σⱼ a_{|k-j|} a_{|j|}, e.g.
            # c₀ = a₀a₀ + 2a₁a₁ + 2a₂a₂ = 1 + 2(0.25) + 2(0.25) = 2
            # c₁ = 2a₀a₁ + 2a₁a₀ ... (verified against the source's convolution formula)
            expected = Sequence(Chebyshev(4), [2.0, 1.5, 1.25, 0.5, 0.25])
            out = Sequence(Chebyshev(4), fill(Inf, 5))
            @test ℳ(a) == a*a == mul!(out, ℳ, a) == expected
        end

        @testset "TensorSpace" begin
            s = Taylor(1) ⊗ Taylor(1)
            a = Sequence(s, [1.0, 2.0, 3.0, 4.0]) # 1 + 2x + 3y + 4xy (indices (0,0),(1,0),(0,1),(1,1))
            b = Sequence(s, [1.0, 0.0, 0.0, 1.0]) # 1 + xy
            ℳ = Multiplication(a)
            # (1+2x+3y+4xy)(1+xy) = 1+2x+3y+5xy+2x²y+3xy²+4x²y²
            expected = Sequence(Taylor(2) ⊗ Taylor(2),
                [1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0])
            @test ℳ(b) == b*a == expected
        end
    end

    @testset "project-to-matrix consistency" begin
        a = Sequence(Taylor(1), [2.0, 3.0]) # 2 + 3x
        ℳ = Multiplication(a)
        A = project(ℳ, Taylor(1), Taylor(2), Float64)
        # column j is the shifted copy of a corresponding to multiplication by xʲ
        @test coefficients(A) == [2.0 0.0 ; 3.0 2.0 ; 0.0 3.0]
        b = Sequence(Taylor(1), [1.0, 1.0])
        @test A*b == ℳ(b) == a*b

        a_ℱ = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])
        ℳ_ℱ = Multiplication(a_ℱ)
        A_ℱ = project(ℳ_ℱ, Fourier(1, 1.0), Fourier(2, 1.0), Float64)
        @test A_ℱ*a_ℱ == ℳ_ℱ(a_ℱ) == a_ℱ*a_ℱ

        a_𝒞 = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        ℳ_𝒞 = Multiplication(a_𝒞)
        A_𝒞 = project(ℳ_𝒞, Chebyshev(2), Chebyshev(4), Float64)
        @test A_𝒞*a_𝒞 == ℳ_𝒞(a_𝒞) == a_𝒞*a_𝒞
    end

    @testset "symmetric space (evensym Fourier)" begin
        # cos(t): representatives [a₀, a₁] = [0.0, 0.5] of evensym(Fourier(1,1.0))
        a = Sequence(evensym(Fourier(1, 1.0)), [0.0, 0.5])
        ℳ = Multiplication(a)
        # cos²(t) = 1/2 + 1/2 cos(2t): representatives [0.5, 0.0, 0.25] of evensym(Fourier(2,1.0))
        expected = Sequence(evensym(Fourier(2, 1.0)), [0.5, 0.0, 0.25])
        @test ℳ(a) == a*a == expected
        P = project(ℳ, evensym(Fourier(1, 1.0)), evensym(Fourier(2, 1.0)), Float64)
        @test P*a == expected
    end

    @testset "cartesian space (not supported)" begin
        a = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        ℳ = Multiplication(a)
        # Multiplication is deliberately excluded from the shared CartesianSpace
        # block in special_operators.jl; the generic fallback in linear_operator.jl
        # reports that it cannot infer a domain for a CartesianSpace codomain.
        @test_throws DomainError domain(ℳ, Taylor(2)^2)
    end

    @testset "ComplexF64 coefficients" begin
        a = Sequence(Taylor(1), ComplexF64[1.0+1.0im, 2.0]) # (1+i) + 2x
        b = Sequence(Taylor(1), ComplexF64[1.0, 1.0im])     # 1 + ix
        ℳ = Multiplication(a)
        # (1+i)*1 = 1+i ; (1+i)(ix) + 2x = (i-1)x + 2x = (1+i)x ; 2x*ix = 2i x²
        expected = Sequence(Taylor(2), ComplexF64[1.0+1.0im, 1.0+1.0im, 2.0im])
        @test ℳ(b) == b*a == expected
    end

    @testset "Interval{Float64} coefficients" begin
        a = Sequence(Taylor(1), interval.([1.0, 2.0])) # 1 + 2x
        b = Sequence(Taylor(1), interval.([3.0, 4.0])) # 3 + 4x
        ℳ = Multiplication(a)
        # (1+2x)(3+4x) = 3 + 10x + 8x²
        expected = Sequence(Taylor(2), interval.([3.0, 10.0, 8.0]))
        @test ℳ(b) == b*a == expected
    end

    @testset "Complex{Interval{Float64}} coefficients" begin
        a = Sequence(Fourier(1, 1.0), Complex{Interval{Float64}}.([0.5, 0.0, 0.5]))
        ℳ = Multiplication(a)
        expected = Sequence(Fourier(2, 1.0), Complex{Interval{Float64}}.([0.25, 0.0, 0.5, 0.0, 0.25]))
        @test ℳ(a) == a*a == expected
    end
end
