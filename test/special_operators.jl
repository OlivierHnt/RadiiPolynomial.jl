@testset "Special operators" begin
    @testset "Projection" begin
        s1 = ParameterSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
        s2 = ParameterSpace() × (Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
        s3 = ParameterSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0))^1
        s4 = ParameterSpace() × (Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0))^1
        a = Sequence(s1, [1.0 ; 1.0:12.0])
        @test project(a, s1) == a
        @test project(a, s2) == Sequence(s2, [1.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 7.0, 8.0, 0.0, 9.0, 10.0, 0.0, 11.0, 12.0, 0.0])
        @test project(a, s3) == Sequence(s3, [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        @test project(a, s4) == Sequence(s4, [1.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0])
    end

    @testset "Multiplication" begin
        a_𝒯 = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        ℳ_𝒯 = Multiplication(a_𝒯)
        @test (2ℳ_𝒯/1).sequence == (ℳ_𝒯 + ℳ_𝒯).sequence == (ℳ_𝒯 - (-ℳ_𝒯)).sequence == Multiplication(a_𝒯 + a_𝒯).sequence
        @test (ℳ_𝒯^2).sequence == (ℳ_𝒯 * ℳ_𝒯).sequence == Multiplication(a_𝒯 * a_𝒯).sequence
        @test ℳ_𝒯(a_𝒯) == a_𝒯*a_𝒯 == project(ℳ_𝒯, Taylor(2), Taylor(4), Float64)*a_𝒯
        #
        a_ℱ = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])
        ℳ_ℱ = Multiplication(a_ℱ)
        @test (2ℳ_ℱ/1).sequence == (ℳ_ℱ + ℳ_ℱ).sequence == (ℳ_ℱ - (-ℳ_ℱ)).sequence == Multiplication(a_ℱ + a_ℱ).sequence
        @test (ℳ_ℱ^2).sequence == (ℳ_ℱ * ℳ_ℱ).sequence == Multiplication(a_ℱ * a_ℱ).sequence
        @test ℳ_ℱ(a_ℱ) == a_ℱ*a_ℱ == project(ℳ_ℱ, Fourier(1, 1.0), Fourier(2, 1.0), Float64)*a_ℱ
        #
        a_𝒞 = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        ℳ_𝒞 = Multiplication(a_𝒞)
        @test (2ℳ_𝒞/1).sequence == (ℳ_𝒞 + ℳ_𝒞).sequence == (ℳ_𝒞 - (-ℳ_𝒞)).sequence == Multiplication(a_𝒞 + a_𝒞).sequence
        @test (ℳ_𝒞^2).sequence == (ℳ_𝒞 * ℳ_𝒞).sequence == Multiplication(a_𝒞 * a_𝒞).sequence
        @test ℳ_𝒞(a_𝒞) == a_𝒞*a_𝒞 == project(ℳ_𝒞, Chebyshev(2), Chebyshev(4), Float64)*a_𝒞
    end

    @testset "Derivative" begin
        ∂¹ = Derivative(1)
        ∂⁴ = Derivative(4)
        @test ∂¹*∂¹ == ∂¹^2 == Derivative(2)
        #
        a_𝒯 = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        @test ∂¹(a_𝒯) == project(∂¹, Taylor(2), Taylor(1), Float64)(a_𝒯) ==
            differentiate!(Sequence(Taylor(1), [Inf, Inf]), a_𝒯) == Sequence(Taylor(1), [-1.0, 2.0]) ==
            mul!(Sequence(Taylor(1), [Inf, Inf]), ∂¹, a_𝒯) == Sequence(Taylor(1), [-1.0, 2.0])
        @test ∂⁴(a_𝒯) == project(∂⁴, Taylor(2), Taylor(0), Float64)(a_𝒯) ==
            differentiate!(Sequence(Taylor(0), [Inf]), a_𝒯, 4) ==
            mul!(Sequence(Taylor(0), [Inf]), ∂⁴, a_𝒯) == Sequence(Taylor(0), [0.0])
        #
        a_ℱ = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])
        @test ∂¹(a_ℱ) == project(∂¹, Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a_ℱ) ==
            differentiate!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), a_ℱ) ==
            mul!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), ∂¹, a_ℱ) == Sequence(Fourier(1, 1.0), [-0.5im, 0.0, 0.5im])
        @test ∂⁴(a_ℱ) == project(∂⁴, Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a_ℱ) ==
            differentiate!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), a_ℱ, 4) ==
            mul!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), ∂⁴, a_ℱ) == a_ℱ
        #
        a_𝒞 = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        @test ∂¹(a_𝒞) == project(∂¹, Chebyshev(2), Chebyshev(1), Float64)(a_𝒞) ==
            differentiate!(Sequence(Chebyshev(1), [Inf, Inf]), a_𝒞) ==
            mul!(Sequence(Chebyshev(1), [Inf, Inf]), ∂¹, a_𝒞) == Sequence(Chebyshev(1), [1.0, 2.0])
        #
        a_𝑇 = Sequence(Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(2), collect(1.0:27.0))
        selectdim(a_𝑇, 2, 0) .= 0.0
        @test differentiate(integrate(a_𝑇, (1, 2, 1)), (1, 2, 1)) ≈ a_𝑇
        #
        a = Sequence(Taylor(2)^2 × Fourier(1, 1.0) × Chebyshev(2), collect(1.0:6.0+3.0+3.0))
        @test ∂¹(a) ==
            differentiate!(Sequence(Taylor(1)^2 × Fourier(1, 1.0) × Chebyshev(1), fill(complex(Inf), 2*2+3+2)), a) ==
            mul!(Sequence(Taylor(1)^2 × Fourier(1, 1.0) × Chebyshev(1), fill(complex(Inf), 2*2+3+2)), ∂¹, a)
    end

    @testset "Integral" begin
        ∫¹ = Integral(1)
        ∫² = Integral(2)
        @test ∫¹*∫¹ == ∫¹^2 == Integral(2)
        #
        a_𝒯 = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        @test ∫¹(a_𝒯) == project(∫¹, Taylor(2), Taylor(3), Float64)(a_𝒯) ==
            integrate!(Sequence(Taylor(3), [Inf, Inf, Inf, Inf]), a_𝒯) ==
            mul!(Sequence(Taylor(3), [Inf, Inf, Inf, Inf]), ∫¹, a_𝒯) == Sequence(Taylor(3), [0.0, 1.0, -1/2, 1/3])
        @test ∫²(a_𝒯) == project(∫², Taylor(2), Taylor(4), Float64)(a_𝒯) ==
            integrate!(Sequence(Taylor(4), [Inf, Inf, Inf, Inf, Inf]), a_𝒯, 2) ==
            mul!(Sequence(Taylor(4), [Inf, Inf, Inf, Inf, Inf]), ∫², a_𝒯) == Sequence(Taylor(4), [0.0, 0.0, 1/2, -1/6, 1/12])
        #
        a_ℱ = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])
        @test ∫¹(a_ℱ) == project(∫¹, Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a_ℱ) ==
            integrate!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), a_ℱ) ==
            mul!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), ∫¹, a_ℱ) == Sequence(Fourier(1, 1.0), [0.5im, 0.0, -0.5im])
        @test ∫²(a_ℱ) == project(∫², Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a_ℱ) ==
            integrate!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), a_ℱ, 2) ==
            mul!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), ∫², a_ℱ) == -a_ℱ
        #
        a_𝒞 = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        @test project(∫¹, Chebyshev(2), Chebyshev(3), Float64)(a_𝒞) ==
            mul!(Sequence(Chebyshev(3), [Inf, Inf, Inf, Inf]), ∫¹, a_𝒞) ≈
            ∫¹(a_𝒞) == integrate!(Sequence(Chebyshev(3), [Inf, Inf, Inf, Inf]), a_𝒞)
        #
        a = Sequence(Taylor(2)^2 × Fourier(1, 1.0) × Chebyshev(2), collect(1.0:6.0+3.0+3.0))
        component(a, 2)[0] = 0.0
        @test ∫¹(a) ==
            integrate!(Sequence(Taylor(3)^2 × Fourier(1, 1.0) × Chebyshev(3), fill(complex(Inf), 8+3+4)), a) ==
            mul!(Sequence(Taylor(3)^2 × Fourier(1, 1.0) × Chebyshev(3), fill(complex(Inf), 8+3+4)), ∫¹, a)
    end

    @testset "Scale" begin
        𝒮 = Scale(0.5)
        a  = Sequence(Taylor(3), [1.0, 2.0, 4.0, 8.0])
        @test 𝒮(a) == project(𝒮, Taylor(3), Taylor(3), Float64)(a) == Sequence(Taylor(3), [1.0, 1.0, 1.0, 1.0])
        #
        𝒮 = Scale((0.5, 0.5))
        a  = Sequence(Taylor(3) ⊗ Taylor(2), [2.0 ^ sum(α) for α ∈ indices(Taylor(3) ⊗ Taylor(2))])
        @test 𝒮(a) == project(𝒮, Taylor(3) ⊗ Taylor(2), Taylor(3) ⊗ Taylor(2), Float64)(a) == Sequence(Taylor(3) ⊗ Taylor(2), ones(12))
    end

    @testset "Shift" begin
        𝒮 = Shift(π/2)
        a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])
        @test 𝒮(a) ≈ Sequence(Fourier(1, 1.0), [-0.5im, 0.0, 0.5im])
    end
end
