@testset "Refinement" begin

    @testset "envelope primitives: the box in interval, its midpoint in float" begin
        # `_envelope_box(T, μ)` is the uncertainty box [-μ, μ] a coefficient below the
        # envelope μ must lie in; floating point coefficients keep only its midpoint, 0
        box = RadiiPolynomial._envelope_box(Interval{Float64}, interval(0.5))
        @test inf(box) == -0.5
        @test sup(box) == 0.5

        cbox = RadiiPolynomial._envelope_box(Complex{Interval{Float64}}, interval(0.5))
        @test in_interval(0.3, real(cbox))
        @test in_interval(-0.4, imag(cbox))

        @test RadiiPolynomial._envelope_box(Float64, interval(0.5)) === 0.0
        @test RadiiPolynomial._envelope_box(ComplexF64, interval(0.5)) === zero(ComplexF64)

        # an NG envelope must yield an NG box: the radius is passed to the constructor as is,
        # since reading `sup(μ)` first would launder the flag away
        μ_ng = 1.0 * interval(0.5)
        @test !isguaranteed(μ_ng)
        @test !isguaranteed(RadiiPolynomial._envelope_box(Interval{Float64}, μ_ng))
        @test isguaranteed(RadiiPolynomial._envelope_box(Interval{Float64}, interval(0.5)))

        # the ℓ¹/ℓ∞ envelope is err/w; the ℓ² one carries the square root
        @test RadiiPolynomial._coefficient_bound(Ell1(), 1.0, 4.0) == 0.25
        @test RadiiPolynomial._coefficient_bound(Ell2(), 1.0, 4.0) == 0.5
    end








    @testset "polish!" begin
        # Clean geometric decay 2^{-i}, i = 0..4, plus a wild outlier at the last index (i=5).
        # The regression correctly identifies rate ≈ 2 from indices 0..4 (ord = 4) and,
        # since |a[5]| = 1000 vastly exceeds norm(a,1)/2^5 ≈ 31.3, that single entry is zeroed.
        a = Sequence(Taylor(5), [1.0, 0.5, 0.25, 0.125, 0.0625, 1000.0])
        pa = polish!(copy(a))
        @test coefficients(pa) == [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0]

        # Same decay without any outlier: nothing exceeds the fitted model, so polish! is a no-op.
        b = Sequence(Taylor(5), [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
        @test polish!(copy(b)) == b

        # CartesianSpace: polish! acts component-wise.
        𝕊 = Taylor(5)^2
        c = Sequence(𝕊, vcat([1.0, 0.5, 0.25, 0.125, 0.0625, 1000.0], [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]))
        pc = polish!(copy(c))
        @test coefficients(pc) == vcat([1.0, 0.5, 0.25, 0.125, 0.0625, 0.0], [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])

        # no-ops by definition
        s = Sequence(3.0)
        @test polish!(s) === s
        t = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(1.0:6.0))
        @test polish!(t) === t
    end
end
