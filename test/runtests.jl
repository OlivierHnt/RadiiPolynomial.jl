using RadiiPolynomial
using Test

@testset "RadiiPolynomial.jl" begin
    @testset "Taylor sequences" begin
        f(x) = -1.0 + x
        a = Sequence(Taylor(1), [-1.0, 1.0])

        @test a*a == a^2 == Sequence(Taylor(2), [1.0, -2.0, 1.0])
        @test a + a == 2a == Sequence(Taylor(1), [-2.0, 2.0])

        @test norm(a) == 2.0

        @test differentiate(integrate(a)) ≈ a

        @test all(x -> f(x) ≈ a(x), -1.0:0.2:1.0)
    end

    @testset "Fourier sequences" begin
        f(x) = cos(x)
        a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])

        @test a*a == a^2 == Sequence(Fourier(2, 1.0), [0.25, 0.0, 0.5, 0.0, 0.25])
        @test a + a == 2a == Sequence(Fourier(1, 1.0), [1.0, 0.0, 1.0])

        @test norm(a) == 1.0

        @test differentiate(integrate(a)) ≈ a

        @test all(x -> f(x) ≈ a(x), -1.0:0.2:1.0)
    end

    @testset "Chebyshev sequences" begin
        f(x) = 1.0 + x + (2x^2 - 1.0)
        a = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])

        @test a*a == a^2 == Sequence(Chebyshev(4), [2.0, 1.5, 1.25, 0.5, 0.25])
        @test a + a == 2a == Sequence(Chebyshev(2), [2.0, 1.0, 1.0])

        @test norm(a) == 2.0

        @test differentiate(integrate(a)) ≈ a

        @test all(x -> f(x) ≈ a(x), -1.0:0.2:1.0)
    end

    @testset "Multivariate sequences" begin
        a = Sequence(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(2), rand(2*3*3))
        selectdim(a, 2, 0) .= 0.0

        @test a*a == a^2
        @test a + a == 2a

        @test differentiate(integrate(a, 1), 1) ≈ a
        @test differentiate(integrate(a, 2), 2) ≈ a
        @test differentiate(integrate(a, 3), 3) ≈ a
    end
end
