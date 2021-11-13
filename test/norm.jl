@testset "Norm" begin
    n = 1_000
    ν = 1.2
    p = 2.7

    a_geo = Sequence(Taylor(n), [inv(ν^abs(i)) for i ∈ 0:n])
    a_alg = Sequence(Taylor(n), [inv((1.0 + abs(i))^p) for i ∈ 0:n])
    @test rate(geometricweights(a_geo)) ≈ ν
    @test rate(algebraicweights(a_alg)) ≈ p
    @test norm(a_geo, Weightedℓ¹Norm(GeometricWeights(ν))) ==
        opnorm(project(Multiplication(a_geo), space(a_geo), space(a_geo), Float64), Weightedℓ¹Norm(GeometricWeights(ν))) ≈
        n+1
    @test norm(a_alg, Weightedℓ¹Norm(AlgebraicWeights(p))) ==
        opnorm(project(Multiplication(a_alg), space(a_alg), space(a_alg), Float64), Weightedℓ¹Norm(AlgebraicWeights(p))) ≈
        n+1

    a_geo = Sequence(Fourier(n, 1.), [inv(ν^abs(i)) for i ∈ -n:n])
    a_alg = Sequence(Fourier(n, 1.), [inv((1.0 + abs(i))^p) for i ∈ -n:n])
    @test rate(geometricweights(a_geo)) ≈ ν
    @test rate(algebraicweights(a_alg)) ≈ p
    @test norm(a_geo, Weightedℓ¹Norm(GeometricWeights(ν))) ==
        opnorm(project(Multiplication(a_geo), space(a_geo), space(a_geo), Float64), Weightedℓ¹Norm(GeometricWeights(ν))) ≈
        2n+1
    @test norm(a_alg, Weightedℓ¹Norm(AlgebraicWeights(p))) ==
        opnorm(project(Multiplication(a_alg), space(a_alg), space(a_alg), Float64), Weightedℓ¹Norm(AlgebraicWeights(p))) ≈
        2n+1

    a_geo = Sequence(Chebyshev(n), [inv(ν^abs(i)) for i ∈ 0:n])
    a_alg = Sequence(Chebyshev(n), [inv((1.0 + abs(i))^p) for i ∈ 0:n])
    @test rate(geometricweights(a_geo)) ≈ ν
    @test rate(algebraicweights(a_alg)) ≈ p
    @test norm(a_geo, Weightedℓ¹Norm(GeometricWeights(ν))) ==
        opnorm(project(Multiplication(a_geo), space(a_geo), space(a_geo), Float64), Weightedℓ¹Norm(GeometricWeights(ν))) ≈
        2n+1
    @test norm(a_alg, Weightedℓ¹Norm(AlgebraicWeights(p))) ==
        opnorm(project(Multiplication(a_alg), space(a_alg), space(a_alg), Float64), Weightedℓ¹Norm(AlgebraicWeights(p))) ≈
        2n+1

    a_geo = Sequence(Taylor(10) ⊗ Fourier(10, 1.) ⊗ Chebyshev(10),
        [inv(ν^abs(α[1]) * ν^abs(α[2]) * ν^abs(α[3])) for α ∈ indices(Taylor(10) ⊗ Fourier(10, 1.) ⊗ Chebyshev(10))])
    a_alg = Sequence(Taylor(10) ⊗ Fourier(10, 1.) ⊗ Chebyshev(10),
        [inv((1.0 + abs(α[1]))^p * (1.0 + abs(α[2]))^p * (1.0 + abs(α[3]))^p) for α ∈ indices(Taylor(10) ⊗ Fourier(10, 1.) ⊗ Chebyshev(10))])
    @test all( rate.(geometricweights(a_geo)) .≈ (ν, ν, ν) )
    @test all( rate.(algebraicweights(a_alg)) .≈ (p, p, p) )
    @test norm(a_geo, Weightedℓ¹Norm( (GeometricWeights(ν), GeometricWeights(ν), GeometricWeights(ν)) )) ==
        opnorm(project(Multiplication(a_geo), space(a_geo), space(a_geo), Float64), Weightedℓ¹Norm( (GeometricWeights(ν), GeometricWeights(ν), GeometricWeights(ν)) ))
    @test norm(a_alg, Weightedℓ¹Norm( (AlgebraicWeights(p), AlgebraicWeights(p), AlgebraicWeights(p)) )) ==
        opnorm(project(Multiplication(a_alg), space(a_alg), space(a_alg), Float64), Weightedℓ¹Norm( (AlgebraicWeights(p), AlgebraicWeights(p), AlgebraicWeights(p)) ))

    s1 = ParameterSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
    s2 = ParameterSpace() × (Taylor(2) ⊗ Fourier(0, 1.0) ⊗ Chebyshev(1))^1
    a_ = Sequence(s1, collect(1.0:13.0))
    A_ = LinearOperator(s1, s2, ones(dimension(s2), dimension(s1)))

    metric = CartesianProductNorm((ℓᵖNorm(Inf), CartesianPowerNorm(Weightedℓ¹Norm((GeometricWeights(1.0), GeometricWeights(1.0), GeometricWeights(1.0))), ℓᵖNorm(Inf))), ℓᵖNorm(Inf))

    @test norm(a_, metric) == 2+4+6+3+5+7 + 2*( 8+10+12+9+11+13 )
    @test opnorm(A_, metric, metric) == 18
end
