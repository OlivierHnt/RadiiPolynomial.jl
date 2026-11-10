# the projected multiplication agrees with the product of the desymmetrized sequences, which uses no symmetric code
function _check_projected_multiplication(a, x, ::Type{T}) where {T}
    ax = a * x
    M = project(Multiplication(a), space(x), space(ax), T)
    @test M * x ≈ ax
    full = (Projection(desymmetrize(space(a))) * a) * (Projection(desymmetrize(space(x))) * x)
    @test all(k -> isapprox(RadiiPolynomial.getcoefficient(M * x, (space(full), k)), full[k]; atol = 1e-12), indices(space(full)))
    return M
end

@testset "Multiplication" begin
    @testset "sequence accessor" begin
        a = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        ℳ = Multiplication(a)
        @test sequence(ℳ) === a
        @test sequence(interval(Float64, ℳ)) == interval.(a)
        @test sequence(interval(ℳ)) == interval(a)
    end

    @testset "arithmetic acts on the underlying sequence" begin
        a = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        ℳ = Multiplication(a)
        for (𝒪, expected) ∈ (
                (+ℳ, a), (-ℳ, -a), (ℳ^0, a^0), (ℳ^2, a*a), (ℳ*ℳ, a*a),
                (2ℳ/1, a+a), (ℳ+ℳ, a+a), (ℳ-(-ℳ), a+a),
                (ℳ+1, a+1), (1+ℳ, 1+a), (ℳ-1, a-1), (1-ℳ, 1-a),
                (3ℳ, 3a), (ℳ*3, a*3), (ℳ/2, a/2), (2\ℳ, 2\a))
            @test sequence(𝒪) == expected
        end
    end

    @testset "action == convolution" begin
        s = Taylor(1) ⊗ Taylor(1)
        cos_ℱ = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])
        a_𝒞 = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        cos_e = Sequence(evensym(Fourier(1, 1.0)), [0.0, 0.5])
        for (a, b, expected) ∈ (
                # (1 - x + x²)(2 + 3x) = 2 + x - x² + 3x³
                (Sequence(Taylor(2), [1.0, -1.0, 1.0]), Sequence(Taylor(1), [2.0, 3.0]), Sequence(Taylor(3), [2.0, 1.0, -1.0, 3.0])),
                # cos²(t) = 1/2 + 1/2 cos(2t)
                (cos_ℱ, cos_ℱ, Sequence(Fourier(2, 1.0), [0.25, 0.0, 0.5, 0.0, 0.25])),
                (cos_e, cos_e, Sequence(evensym(Fourier(2, 1.0)), [0.5, 0.0, 0.25])),
                # cₖ = Σⱼ a_{|k-j|} a_{|j|}, e.g. c₀ = a₀² + 2a₁² + 2a₂² = 2 and c₁ = 2a₀a₁ + 2a₁a₂ = 1.5
                (a_𝒞, a_𝒞, Sequence(Chebyshev(4), [2.0, 1.5, 1.25, 0.5, 0.25])),
                # (1 + 2x + 3y + 4xy)(1 + xy) = 1 + 2x + 3y + 5xy + 2x²y + 3xy² + 4x²y²
                (Sequence(s, [1.0, 2.0, 3.0, 4.0]), Sequence(s, [1.0, 0.0, 0.0, 1.0]),
                    Sequence(Taylor(2) ⊗ Taylor(2), [1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0])),
                # ((1+i) + 2x)(1 + ix) = (1+i) + (1+i)x + 2ix²
                (Sequence(Taylor(1), ComplexF64[1.0+1.0im, 2.0]), Sequence(Taylor(1), ComplexF64[1.0, 1.0im]),
                    Sequence(Taylor(2), ComplexF64[1.0+1.0im, 1.0+1.0im, 2.0im])))
            @test b * a == a * b == expected
            test_action(Multiplication(a), b, expected)
        end

        # (1 + 2x)(3 + 4x) = 3 + 10x + 8x²
        ai, bi = Sequence(Taylor(1), interval.([1.0, 2.0])), Sequence(Taylor(1), interval.([3.0, 4.0]))
        @test Multiplication(ai)(bi) == bi * ai == Sequence(Taylor(2), interval.([3.0, 10.0, 8.0]))
        ci = Sequence(Fourier(1, 1.0), Complex{Interval{Float64}}.([0.5, 0.0, 0.5]))
        @test Multiplication(ci)(ci) == ci * ci == Sequence(Fourier(2, 1.0), Complex{Interval{Float64}}.([0.25, 0.0, 0.5, 0.0, 0.25]))

        # the column j of the matrix is the sequence shifted by j
        @test coefficients(project(Multiplication(Sequence(Taylor(1), [2.0, 3.0])), Taylor(1), Taylor(2), Float64)) == [2.0 0.0 ; 3.0 2.0 ; 0.0 3.0]
    end

    @testset "symmetric domain with a non-trivial cocycle" begin
        # odd (sine) domain: x₋₁ = -x₁, so (a*x)₁ = (a₀ - a₂) x₁ and not (a₀ + a₂) x₁
        a = Sequence(evensym(Fourier(2, 1.0)), [1.0, 0.0, 1.0])
        x = Sequence(oddsym(Fourier(1, 1.0)), [1.0im])
        M = _check_projected_multiplication(a, x, ComplexF64)
        @test coefficients(M) == [0.0; 0.0; 1.0;;]
        # phase group a_k = iᵏ a₋ₖ: the factors are ±i, the projected matrix must be complex
        Gp = Group(GroupElement(LatticeAut([-1;;]), Cocycle(0//1, Rational{Int}[1//2])))
        sp = SymmetricSpace(Fourier(1, 1.0), Gp)
        ap = Sequence(sp, ComplexF64[1.0, 1.0])
        xp = Sequence(sp, ComplexF64[0.0, 1.0])
        _check_projected_multiplication(ap, xp, ComplexF64)
        @test ap * xp == Sequence(SymmetricSpace(Fourier(2, 1.0), Gp), ComplexF64[-2.0im, 1.0, 1.0])
        # on intervalized spaces the cocycle factors are thin intervals, which keeps the guarantee
        sI1 = interval(evensym(Fourier(1, 1.0)))
        sI2 = interval(evensym(Fourier(2, 1.0)))
        aI = Sequence(sI1, interval.([0.0, 0.5]))
        PI = project(Multiplication(aI), sI1, sI2, Interval{Float64})
        @test all(isguaranteed, coefficients(PI))
        @test all(in_interval.([0.5, 0.0, 0.25], coefficients(PI * aI)))
    end

    @testset "mixed symmetric and plain factors" begin
        ℱ = Fourier(2, 1.0)
        es = evensym(ℱ)
        c = Sequence(ℱ, [0.0, 0.3, 1.0, 0.3, 0.0]) # even, stored on the plain space
        ce = project(c, es)
        x = Sequence(es, [1.0, 0.5, 0.2])
        xf = project(x, ℱ)
        # a factor without symmetry leaves none on the product
        @test space(c * x) == space(ce * xf) == Fourier(4, 1.0)
        @test c * x == ce * xf == c * xf
        @test x * c ≈ c * xf
        @test Multiplication(c) * x == c * x
        # a symmetric multiplier on the plain space is the plain multiplication operator
        @test coefficients(project(Multiplication(ce), ℱ, ℱ)) == coefficients(project(Multiplication(c), ℱ, ℱ))
        @test Multiplication(ce) * Projection(ℱ) == Multiplication(c) * Projection(ℱ)
        # a non symmetric multiplier compressed onto a symmetric space averages the rows of the product
        cn = Sequence(ℱ, [0.1, -0.2, 1.0, 0.4, 0.3])
        M = Projection(es) * Multiplication(cn)
        @test domain(M) == Fourier(4, 1.0) && codomain(M) == es
        @test coefficients(M) ≈ coefficients(Projection(es) * project(Multiplication(cn), Fourier(4, 1.0), ℱ))
        @test coefficients(Multiplication(cn) * Projection(es)) ≈ coefficients(project(Multiplication(cn), ℱ, Fourier(4, 1.0)) * Projection(es))
        @test project(Multiplication(cn), es, es) * x ≈ project(cn * xf, es)
        # intervalized spaces keep the guarantee
        ℱi, esi = interval(ℱ), interval(es)
        cei = Sequence(esi, interval.(coefficients(ce)))
        Pi = project(Multiplication(cei), esi, ℱi)
        @test all(isguaranteed, coefficients(Pi))
        @test all(in_interval.(coefficients(project(Multiplication(ce), es, ℱ)), coefficients(Pi)))
        @test all(isguaranteed, coefficients(cei * Sequence(ℱi, interval.(coefficients(xf)))))
    end

    @testset "cartesian space: unsupported" begin
        @test_throws DomainError domain(Multiplication(Sequence(Taylor(2), [1.0, -1.0, 1.0])), Taylor(2)^2)
    end

    @testset "domain symmetry keeps the multiplier's phase" begin
        # If a_k = α_a(k) a_{βk} and b_k = α_b(k) b_{βk}, then (a*b)_k = α_a(k) α_b(k) (a*b)_{βk}, so the domain
        # must carry the phase φ of the multiplier and not -φ. The two differ only when 2φ ≢ 0, as for
        # the group below: β of order 3 with φ = (2/3, 2/3, 2/3), which satisfies βᵀφ ≡ φ.
        P = RadiiPolynomial.StaticArrays.SMatrix{3,3,Int}([0 0 1; 1 0 0; 0 1 0])
        φ = RadiiPolynomial.StaticArrays.SVector{3,Rational{Int}}(2//3, 2//3, 2//3)
        G = Group(GroupElement(LatticeAut(P), Cocycle(0//1, φ)))
        @test length(elements(G)) == 3

        G_dom = RadiiPolynomial._domain_convolution_symmetry(G, G)
        @test RadiiPolynomial._codomain_convolution_symmetry(G, G_dom) == G
        @test G_dom == G

        V = Fourier(2, 1.0) ⊗ Fourier(2, 1.0) ⊗ Fourier(2, 1.0)
        sG = SymmetricSpace(V, G)
        @test RadiiPolynomial.symmetry(RadiiPolynomial._domain(*, sG, sG)) == G

        # the product of two invariant sequences is invariant: c_k = α_g(k) c_{β_g(k)}
        coeffs(s) = ComplexF64[(1 + 0.3im) / (1 + sum(abs, k))^2 for k ∈ indices(s)]
        a = Projection(V) * Sequence(sG, coeffs(sG))
        b = Projection(V) * Sequence(SymmetricSpace(V, G_dom), coeffs(SymmetricSpace(V, G_dom)))
        c = a * b
        worst = maximum(elements(G)) do g
            maximum(indices(space(c))) do k
                βk = g.lattice_aut(k)
                all(abs.(βk) .≤ order(space(c))) || return 0.0
                abs(c[k] - cispi(RadiiPolynomial._angle(g.cocycle, k)) * c[βk])
            end
        end
        @test worst < 1e-12
    end

    @testset "complex cocycle factors" begin
        # a sequence invariant under a₋ₖ = e^{iπk/3} aₖ has complex coefficients off its representatives, even
        # when the stored ones are real
        ℱ = Fourier(2, 1.0)
        c = SymmetricSpace(ℱ, Group(GroupElement(LatticeAut([-1;;]), Cocycle(0//1, Rational{Int}[1//3]))))
        a = Sequence(c, [1.0, 2.0, 3.0])
        @test eltype(project(Multiplication(a), ℱ, Fourier(4, 1.0))) == ComplexF64
        @test eltype(project(Multiplication(a), c, Fourier(4, 1.0))) == ComplexF64
        @test eltype(project(Multiplication(Sequence(ℱ, [1, 2, 3, 4, 5])), ℱ, Fourier(4, 1.0), Int)) == Int
    end
end
