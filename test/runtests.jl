using RadiiPolynomial
using Test

@testset "Spaces" begin
    @testset "Parameter" begin
        𝒫 = ParameterSpace()
        @test 𝒫 ⊆ 𝒫
        @test 𝒫 ∩ 𝒫 == 𝒫 ∪ 𝒫 == 𝒫
        @test dimension(𝒫) == startindex(𝒫) == endindex(𝒫) == 1
        @test allindices(𝒫) == Base.OneTo(1)
        @test isindexof(1, 𝒫)
        @test isindexof(1:1, 𝒫)
        @test isindexof(:, 𝒫)
        @test isindexof([1,1,1], 𝒫)
    end

    @testset "Taylor" begin
        𝒯₂ = Taylor(2)
        𝒯₃ = Taylor(3)
        @test 𝒯₂ ≠ 𝒯₃
        @test 𝒯₂ ⊆ 𝒯₃
        @test 𝒯₂ ∩ 𝒯₃ == 𝒯₂
        @test 𝒯₂ ∪ 𝒯₃ == 𝒯₃
        @test dimension(𝒯₂) == 3
        @test startindex(𝒯₂) == 0
        @test endindex(𝒯₂) == 2
        @test allindices(𝒯₂) == 0:2
        @test isindexof(1, 𝒯₂)
        @test isindexof(0:1, 𝒯₂)
        @test isindexof(:, 𝒯₂)
        @test isindexof([0,1,2], 𝒯₂)
    end

    @testset "Fourier" begin
        ℱ₂ = Fourier(2, 1.0)
        ℱ₃ = Fourier(3, 1.0)
        @test ℱ₂ ≠ ℱ₃
        @test ℱ₂ ⊆ ℱ₃
        @test ℱ₂ ∩ ℱ₃ == ℱ₂
        @test ℱ₂ ∪ ℱ₃ == ℱ₃
        @test dimension(ℱ₂) == 5
        @test startindex(ℱ₂) == -2
        @test endindex(ℱ₂) == 2
        @test allindices(ℱ₂) == -2:2
        @test isindexof(1, ℱ₂)
        @test isindexof(0:1, ℱ₂)
        @test isindexof(:, ℱ₂)
        @test isindexof([0,1,2], ℱ₂)
    end

    @testset "Chebyshev" begin
        𝒞₂ = Chebyshev(2)
        𝒞₃ = Chebyshev(3)
        @test 𝒞₂ ≠ 𝒞₃
        @test 𝒞₂ ⊆ 𝒞₃
        @test 𝒞₂ ∩ 𝒞₃ == 𝒞₂
        @test 𝒞₂ ∪ 𝒞₃ == 𝒞₃
        @test dimension(𝒞₂) == 3
        @test startindex(𝒞₂) == 0
        @test endindex(𝒞₂) == 2
        @test allindices(𝒞₂) == 0:2
        @test isindexof(1, 𝒞₂)
        @test isindexof(0:1, 𝒞₂)
        @test isindexof(:, 𝒞₂)
        @test isindexof([0,1,2], 𝒞₂)
    end

    @testset "Tensor" begin
        𝑇 = TensorSpace((Taylor(2), Fourier(2, 1.0), Chebyshev(2)))
        @test 𝑇 == Taylor(2) ⊗ (Fourier(2, 1.0) ⊗ Chebyshev(2)) == (Taylor(2) ⊗ Fourier(2, 1.0)) ⊗ Chebyshev(2)
        @test 𝑇 ⊆ Taylor(3) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(2)
        @test 𝑇 ∩ (Taylor(3) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(3)) == Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(2)
        @test 𝑇 ∪ (Taylor(3) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(3)) == Taylor(3) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(3)
        @test dimension(𝑇) == 45
        @test dimensions(𝑇) == (3, 5, 3)
        @test dimensions(𝑇, 1) == 3
        @test startindex(𝑇) == (0, -2, 0)
        @test endindex(𝑇) == (2, 2, 2)
        @test allindices(𝑇) == Base.Iterators.product(0:2, -2:2, 0:2)
        @test isindexof((1, 2, 1), 𝑇)
        @test isindexof((0:1, -1, 1:2), 𝑇)
        @test isindexof(:, 𝑇)
        @test isindexof([(0, 1, 2), (1, -2, 1)], 𝑇)
    end

    @testset "Cartesian" begin
        𝒯₂² = CartesianPowerSpace(Taylor(2), 2)
        𝒯₃² = CartesianPowerSpace(Taylor(3), 2)
        @test 𝒯₂² == Taylor(2)^2
        @test CartesianPowerSpace(𝒯₂², 2) == 𝒯₂²^2
        @test 𝒯₂² ∩ 𝒯₃² == 𝒯₂²
        @test 𝒯₂² ∪ 𝒯₃² == 𝒯₃²
        @test spaces(𝒯₂²) == [Taylor(2), Taylor(2)]
        @test dimension(𝒯₂²) == 6
        @test startindex(𝒯₂²) == 1
        @test endindex(𝒯₂²) == 6
        @test allindices(𝒯₂²) == Base.OneTo(6)
        @test isindexof(1, 𝒯₂²)
        @test isindexof(1:2, 𝒯₂²)
        @test isindexof(:, 𝒯₂²)
        @test isindexof([1,2,3], 𝒯₂²)

        @test Taylor(2) × Chebyshev(3) == CartesianProductSpace((Taylor(2), Chebyshev(3)))
        @test Taylor(2) × Chebyshev(3) == CartesianProductSpace((Taylor(2), Chebyshev(3)))
        @test Taylor(2) × Chebyshev(3) == CartesianProductSpace((Taylor(2), Chebyshev(3)))
        @test (Taylor(1) × Taylor(2)) × Taylor(3) == CartesianProductSpace((Taylor(1), Taylor(2), Taylor(3)))
    end
end

@testset "Sequences" begin
    a = Sequence(Taylor(1) × Chebyshev(2), [1,2,1,2,3])
    @test space(a) == Taylor(1) × Chebyshev(2)
    @test coefficients(a) == [1,2,1,2,3]
    @test component(a, 1) == Sequence(Taylor(1), view(coefficients(a), 1:2))
    @test component(a, 2) == Sequence(Chebyshev(2), view(coefficients(a), 3:5))
end

@testset "Arithmetic" begin
    @testset "Taylor" begin
        a = Sequence(Taylor(1), [-1.0, 1.0])
        @test -(1.0 - a) == a - 1.0 == Sequence(Taylor(1), [-2.0, 1.0])
        @test 1.0 + a == a + 1.0 == Sequence(Taylor(1), [0.0, 1.0])
        @test a - (-a) == a + a == 2a == Sequence(Taylor(1), [-2.0, 2.0])
        @test a*a == a^2 == Sequence(Taylor(2), [1.0, -2.0, 1.0])
    end

    @testset "Fourier" begin
        a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])
        @test -(1.0 - a) == a - 1.0 == Sequence(Fourier(1, 1.0), [0.5, -1.0, 0.5])
        @test 1.0 + a == a + 1.0 == Sequence(Fourier(1, 1.0), [0.5, 1.0, 0.5])
        @test a - (-a) == a + a == 2a == Sequence(Fourier(1, 1.0), [1.0, 0.0, 1.0])
        @test a*a == a^2 == Sequence(Fourier(2, 1.0), [0.25, 0.0, 0.5, 0.0, 0.25])
    end

    @testset "Chebyshev" begin
        a = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        @test -(1.0 - a) == a - 1.0 == Sequence(Chebyshev(2), [0.0, 0.5, 0.5])
        @test 1.0 + a == a + 1.0 == Sequence(Chebyshev(2), [2.0, 0.5, 0.5])
        @test a - (-a) == a + a == 2a == Sequence(Chebyshev(2), [2.0, 1.0, 1.0])
        @test a*a == a^2 == Sequence(Chebyshev(4), [2.0, 1.5, 1.25, 0.5, 0.25])
    end

    @testset "Tensor" begin
        a = Sequence(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        @test -(1.0 - a) == a - 1.0 == Sequence(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0), [1.0, 2.0, 2.0, 4.0, 5.0, 6.0])
        @test 1.0 + a == a + 1.0 == Sequence(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0), [1.0, 2.0, 4.0, 4.0, 5.0, 6.0])
        @test a - (-a) == a + a == 2a
        @test a*a == a^2
    end
end

@testset "Special operators" begin
    @testset "Taylor" begin
        a = Sequence(Taylor(1), [-1.0, 1.0])
        @test differentiate(integrate(a)) ≈ a
        @test all(x -> -1.0 + x ≈ a(x), -1.0:0.2:1.0)
        @test project(a, Taylor(3)) == Sequence(Taylor(3), [-1.0, 1.0, 0.0, 0.0])
        @test scale(Sequence(Taylor(2), [1.0, 1.0, 1.0]), 2.0) == Sequence(Taylor(2), [1.0, 2.0, 4.0])
    end

    @testset "Fourier" begin
        a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])
        @test differentiate(integrate(a)) ≈ a
        @test all(x -> cos(x) ≈ a(x), -1.0:0.2:1.0)
        @test project(a, Fourier(2, 1.0)) == Sequence(Fourier(2, 1.0), [0.0, 0.5, 0.0, 0.5, 0.0])
        @test shift(a, 0.0, -π/2) ≈ Sequence(Fourier(1, 1.0), [0.5im, 0.0, -0.5im])
    end

    @testset "Chebyshev" begin
        a = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        @test differentiate(integrate(a)) ≈ a
        @test project(a, Chebyshev(3)) == Sequence(Chebyshev(3), [1.0, 0.5, 0.5, 0.0])
        @test all(x -> 1.0 + x + (2x^2 - 1.0) ≈ a(x), -1.0:0.2:1.0)
    end

    @testset "Tensor" begin
        a = Sequence(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0), [1.0, 2.0, 0.0, 0.0, 5.0, 6.0])
        @test differentiate(integrate(a, 1), 1) ≈ a
    end
end
