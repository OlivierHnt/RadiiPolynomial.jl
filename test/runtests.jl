using RadiiPolynomial
using Test

@testset "RadiiPolynomial.jl" begin
    @testset "Arithmetic for Taylor sequences" begin
        # 1D
        a = Sequence(Taylor(5), rand(6))
        b = Sequence(Taylor(5), rand(6))
        @test a^2 ≈ a*a
        @test a^5 ≈ a*a*a*a*a
        @test a*b ≈ b*a

        @test a + b == b + a
        @test iszero(a - a)

        # ND
        a_ = Sequence(Taylor(5)⊗Taylor(2), rand(6*3))
        b_ = Sequence(Taylor(5)⊗Taylor(2), rand(6*3))
        @test a_^2 ≈ a_*a_
        @test a_^5 ≈ a_*a_*a_*a_*a_
        @test a_*b_ ≈ b_*a_

        @test a_ + b_ == b_ + a_
        @test iszero(a_ - a_)
    end

    @testset "Arithmetic for Fourier sequences" begin
        # 1D
        a = Sequence(Fourier(5, 1.), rand(11))
        b = Sequence(Fourier(5, 1.), rand(11))
        @test a^2 ≈ a*a
        @test a^5 ≈ a*a*a*a*a
        @test a*b ≈ b*a

        @test a + b == b + a
        @test iszero(a - a)

        # ND
        a_ = Sequence(Fourier(5, 1.)⊗Fourier(2, 0.5), rand(11*5))
        b_ = Sequence(Fourier(5, 1.)⊗Fourier(2, 0.5), rand(11*5))
        @test a_^2 ≈ a_*a_
        @test a_^5 ≈ a_*a_*a_*a_*a_
        @test a_*b_ ≈ b_*a_

        @test a_ + b_ == b_ + a_
        @test iszero(a_ - a_)
    end

    @testset "Arithmetic for Chebyshev sequences" begin
        # 1D
        a = Sequence(Chebyshev(5), rand(6))
        b = Sequence(Chebyshev(5), rand(6))
        @test a^2 ≈ a*a
        @test a^5 ≈ a*a*a*a*a
        @test a*b ≈ b*a

        @test a + b == b + a
        @test iszero(a - a)

        # ND
        a_ = Sequence(Chebyshev(5)⊗Chebyshev(2), rand(6*3))
        b_ = Sequence(Chebyshev(5)⊗Chebyshev(2), rand(6*3))
        @test a_^2 ≈ a_*a_
        @test a_^5 ≈ a_*a_*a_*a_*a_
        @test a_*b_ ≈ b_*a_

        @test a_ + b_ == b_ + a_
        @test iszero(a_ - a_)
    end
end
