@testset "Norm" begin
    n = 1_000
    ν = 1.2
    p = 2.7

    a_geo = Sequence(Taylor(n), [inv(ν^abs(i)) for i ∈ 0:n])
    a_alg = Sequence(Taylor(n), [inv((1.0 + abs(i))^p) for i ∈ 0:n])
    @test rate(geometricweight(a_geo)) ≈ ν
    @test rate(algebraicweight(a_alg)) ≈ p
    @test norm(a_geo, ℓ¹(GeometricWeight(ν))) ==
        opnorm(project(Multiplication(a_geo), space(a_geo), space(a_geo), Float64), ℓ¹(GeometricWeight(ν))) ≈
        n+1
    @test norm(a_alg, ℓ¹(AlgebraicWeight(p))) ==
        opnorm(project(Multiplication(a_alg), space(a_alg), space(a_alg), Float64), ℓ¹(AlgebraicWeight(p))) ≈
        n+1

    a_geo = Sequence(Fourier(n, 1.), [inv(ν^abs(i)) for i ∈ -n:n])
    a_alg = Sequence(Fourier(n, 1.), [inv((1.0 + abs(i))^p) for i ∈ -n:n])
    @test rate(geometricweight(a_geo)) ≈ ν
    @test rate(algebraicweight(a_alg)) ≈ p
    @test norm(a_geo, ℓ¹(GeometricWeight(ν))) ==
        opnorm(project(Multiplication(a_geo), space(a_geo), space(a_geo), Float64), ℓ¹(GeometricWeight(ν))) ≈
        2n+1
    @test norm(a_alg, ℓ¹(AlgebraicWeight(p))) ==
        opnorm(project(Multiplication(a_alg), space(a_alg), space(a_alg), Float64), ℓ¹(AlgebraicWeight(p))) ≈
        2n+1

    a_geo = Sequence(Chebyshev(n), [inv(ν^abs(i)) for i ∈ 0:n])
    a_alg = Sequence(Chebyshev(n), [inv((1.0 + abs(i))^p) for i ∈ 0:n])
    @test rate(geometricweight(a_geo)) ≈ ν
    @test rate(algebraicweight(a_alg)) ≈ p
    @test norm(a_geo, ℓ¹(GeometricWeight(ν))) ==
        opnorm(project(Multiplication(a_geo), space(a_geo), space(a_geo), Float64), ℓ¹(GeometricWeight(ν))) ≈
        2n+1
    @test norm(a_alg, ℓ¹(AlgebraicWeight(p))) ==
        opnorm(project(Multiplication(a_alg), space(a_alg), space(a_alg), Float64), ℓ¹(AlgebraicWeight(p))) ≈
        2n+1

    a_geo = Sequence(Taylor(10) ⊗ Fourier(10, 1.) ⊗ Chebyshev(10),
        [inv(ν^abs(α[1]) * ν^abs(α[2]) * ν^abs(α[3])) for α ∈ indices(Taylor(10) ⊗ Fourier(10, 1.) ⊗ Chebyshev(10))])
    a_alg = Sequence(Taylor(10) ⊗ Fourier(10, 1.) ⊗ Chebyshev(10),
        [inv((1.0 + abs(α[1]))^p * (1.0 + abs(α[2]))^p * (1.0 + abs(α[3]))^p) for α ∈ indices(Taylor(10) ⊗ Fourier(10, 1.) ⊗ Chebyshev(10))])
    @test all( rate.(geometricweight(a_geo)) .≈ (ν, ν, ν) )
    @test all( rate.(algebraicweight(a_alg)) .≈ (p, p, p) )
    @test norm(a_geo, ℓ¹( (GeometricWeight(ν), GeometricWeight(ν), GeometricWeight(ν)) )) ==
        opnorm(project(Multiplication(a_geo), space(a_geo), space(a_geo), Float64), ℓ¹( (GeometricWeight(ν), GeometricWeight(ν), GeometricWeight(ν)) ))
    @test norm(a_alg, ℓ¹( (AlgebraicWeight(p), AlgebraicWeight(p), AlgebraicWeight(p)) )) ==
        opnorm(project(Multiplication(a_alg), space(a_alg), space(a_alg), Float64), ℓ¹( (AlgebraicWeight(p), AlgebraicWeight(p), AlgebraicWeight(p)) ))

    s1 = ParameterSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
    s2 = ParameterSpace() × (Taylor(2) ⊗ Fourier(0, 1.0) ⊗ Chebyshev(1))^1
    a_ = Sequence(s1, collect(1.0:13.0))
    A_ = LinearOperator(s1, s2, ones(dimension(s2), dimension(s1)))

    X = NormedCartesianSpace((ℓ∞(), NormedCartesianSpace(ℓ¹((GeometricWeight(1.0), GeometricWeight(1.0), GeometricWeight(1.0))), ℓ∞())), ℓ∞())

    @test norm(a_, X) == 2+4+6+3+5+7 + 2*( 8+10+12+9+11+13 )
    @test opnorm(A_, X, X) == 18
end
