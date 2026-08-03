@testset "Integral" begin

    @testset "Constructors and algebra" begin
        @test order(Integral(1)) == 1
        @test order(Integral(1, 2)) == (1, 2) == order(Integral((1, 2)))
        @test Integral(1, 2) == Integral((1, 2)) # varargs constructor
        # composition and powers
        @test Integral(1) * Integral(1) == Integral(1)^2 == Integral(2)
        @test Integral((1, 2)) * Integral((2, 1)) == Integral((3, 3))
        @test Integral((1, 2))^2 == Integral((2, 4))
        @test Integral((1, 2))^(2, 3) == Integral((2, 6))
        # guards: only non-negative orders, and at least one order in a tuple
        @test_throws DomainError Integral(-1)
        @test_throws DomainError Integral((1, -1))
        @test_throws ArgumentError Integral(())
        @test_throws ArgumentError Integral()
    end

    @testset "Taylor" begin
        ∫¹ = Integral(1)
        ∫² = Integral(2)

        # domain/codomain: ∫ⁿ maps Taylor(N) to Taylor(N+n); domain is the inverse map (via Derivative)
        @test domain(∫², Taylor(3)) == Taylor(1)
        @test codomain(∫², Taylor(3)) == Taylor(5)

        # n = 0: identity, but the coefficient type is always promoted to typeof(inv(one(T))*zero(T))
        a_int = Sequence(Taylor(2), [1, 2, 3])
        c_int = integrate(a_int, 0)
        @test eltype(c_int) == Float64
        @test c_int == Sequence(Taylor(2), [1.0, 2.0, 3.0])

        # n = 1, only dyadic denominators so an exact `==` chain is safe: c[i+1] = a[i]/(i+1), c[0] = 0
        a_𝒯 = Sequence(Taylor(1), [2.0, 4.0])
        @test ∫¹(a_𝒯) == project(∫¹, Taylor(1), Taylor(2), Float64)(a_𝒯) ==
            integrate!(Sequence(Taylor(2), [Inf, Inf, Inf]), a_𝒯) ==
            mul!(Sequence(Taylor(2), [Inf, Inf, Inf]), ∫¹, a_𝒯) == Sequence(Taylor(2), [0.0, 2.0, 2.0])

        # n = 1, general case (1/3 is not exactly representable, but both sides round the same way)
        a_𝒯2 = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        @test ∫¹(a_𝒯2) == project(∫¹, Taylor(2), Taylor(3), Float64)(a_𝒯2) ==
            integrate!(Sequence(Taylor(3), [Inf, Inf, Inf, Inf]), a_𝒯2) ==
            mul!(Sequence(Taylor(3), [Inf, Inf, Inf, Inf]), ∫¹, a_𝒯2) == Sequence(Taylor(3), [0.0, 1.0, -1/2, 1/3])

        # n = 2: ∫∫(1 - x + x²) = x²/2 - x³/6 + x⁴/12
        @test ∫²(a_𝒯2) == project(∫², Taylor(2), Taylor(4), Float64)(a_𝒯2) ==
            integrate!(Sequence(Taylor(4), [Inf, Inf, Inf, Inf, Inf]), a_𝒯2, 2) ==
            mul!(Sequence(Taylor(4), [Inf, Inf, Inf, Inf, Inf]), ∫², a_𝒯2) == Sequence(Taylor(4), [0.0, 0.0, 1/2, -1/6, 1/12])

        # composition ∫∘∫ == ∫²
        @test ∫¹(∫¹(a_𝒯2)) == ∫²(a_𝒯2)

        # derivative ∘ integral == identity (exact, no information lost)
        @test differentiate(∫¹(a_𝒯2)) == a_𝒯2

        # spaces must match in the in-place forms
        @test_throws ArgumentError integrate!(Sequence(Taylor(1), [Inf, Inf]), a_𝒯2)

        # ComplexF64 coefficients, dyadic denominators
        a_𝒯c = Sequence(Taylor(2), ComplexF64[2 + 2im, -4.0, 6im])
        @test ∫¹(a_𝒯c) == Sequence(Taylor(3), ComplexF64[0.0, 2 + 2im, -2.0, 2im])

        # Interval{Float64} coefficients
        a_𝒯i = Sequence(Taylor(1), interval.([2.0, 4.0]))
        @test integrate(a_𝒯i) == Sequence(Taylor(2), interval.([0.0, 2.0, 2.0])) # dyadic ⇒ degenerate intervals, `==` is safe
        a_𝒯i2 = Sequence(Taylor(2), interval.([1.0, -1.0, 1.0]))
        c_𝒯i2 = ∫¹(a_𝒯i2)
        @test isequal_interval(c_𝒯i2[0], interval(0.0)) & isequal_interval(c_𝒯i2[1], interval(1.0)) & isequal_interval(c_𝒯i2[2], interval(-0.5))
        @test in_interval(1//3, c_𝒯i2[3]) # 1/3 is not exactly representable: enclosure check instead of `==`
    end

    @testset "Fourier" begin
        ∫¹ = Integral(1)
        ∫² = Integral(2)
        ∫³ = Integral(3)
        ∫⁴ = Integral(4)

        # the mean-free antiderivative is diagonal (zero column at j = 0), so domain and codomain
        # are both the Fourier space itself; validity of the input (a₀ = 0) is checked at action time
        @test domain(∫¹, Fourier(3, 1.0)) == Fourier(3, 1.0)
        @test domain(Integral(0), Fourier(3, 1.0)) == Fourier(3, 1.0)
        @test codomain(∫¹, Fourier(3, 1.0)) == Fourier(3, 1.0)

        # n = 0: identity; the mean-free guard does not apply, but the coefficient type is still forced to Complex
        a_int = Sequence(Fourier(1, 1.0), [0.5, 1.0, 0.5])
        @test integrate(a_int, 0) == Sequence(Fourier(1, 1.0), ComplexF64[0.5, 1.0, 0.5])

        # guard: for n ≥ 1 the zero mode of `a` must vanish (mean-free antiderivative convention)
        a_bad = Sequence(Fourier(1, 1.0), [0.5, 1.0, 0.5]) # nonzero mean
        @test_throws DomainError integrate(a_bad)
        @test_throws DomainError ∫¹(a_bad)

        # n = 1: ∫cos = sin
        a_ℱ = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5]) # cos(x)
        @test ∫¹(a_ℱ) == project(∫¹, Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a_ℱ) ==
            integrate!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), a_ℱ) ==
            mul!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), ∫¹, a_ℱ) == Sequence(Fourier(1, 1.0), [0.5im, 0.0, -0.5im]) # sin(x)

        # mean-free antiderivative: ∫∫cos = -cos (zero mode stays zero)
        @test ∫¹(∫¹(a_ℱ)) == ∫²(a_ℱ) == project(∫¹, Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)^2 * a_ℱ ==
            integrate!(Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf]), integrate(a_ℱ, 1)) == -a_ℱ

        # the i⁻ⁿ phase cycles with period 4: ∫³cos = -∫¹cos = -sin, ∫⁴cos = cos
        @test ∫³(a_ℱ) == ∫¹(∫²(a_ℱ)) == -Sequence(Fourier(1, 1.0), ComplexF64[0.5im, 0.0, -0.5im])
        @test ∫⁴(a_ℱ) == ∫²(∫²(a_ℱ)) == Sequence(Fourier(1, 1.0), ComplexF64[0.5, 0.0, 0.5])

        # `project` builds the matrix entrywise via `getcoefficient`, which branches on n % 4;
        # the action calls above only ever exercise n % 4 == 1 (via ∫¹). Cross-check the
        # remaining residues (and n = 0) against the already-verified action results
        @test project(Integral(0), Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a_ℱ) == integrate(a_ℱ, 0) # n = 0
        @test project(∫², Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a_ℱ) == ∫²(a_ℱ) # n % 4 == 2
        @test project(∫³, Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a_ℱ) == ∫³(a_ℱ) # n % 4 == 3
        @test project(∫⁴, Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a_ℱ) == ∫⁴(a_ℱ) # n % 4 == 0

        # derivative ∘ integral == identity for a mean-free sequence
        @test differentiate(∫¹(a_ℱ)) == a_ℱ

        @test_throws ArgumentError integrate!(Sequence(Fourier(2, 1.0), fill(complex(Inf), 5)), a_ℱ)

        # Interval{Float64}/Complex{Interval{Float64}} coefficients
        a_ℱi = Sequence(Fourier(1, 1.0), interval.([0.5, 0.0, 0.5]))
        c_ℱi = integrate(a_ℱi)
        @test isequal_interval(real(c_ℱi[-1]), interval(0.0)) & isequal_interval(imag(c_ℱi[-1]), interval(0.5))
        @test isequal_interval(real(c_ℱi[0]), interval(0.0)) & isequal_interval(imag(c_ℱi[0]), interval(0.0))
        @test isequal_interval(real(c_ℱi[1]), interval(0.0)) & isequal_interval(imag(c_ℱi[1]), interval(-0.5))
    end

    @testset "Chebyshev" begin
        ∫¹ = Integral(1)

        # unlike Fourier, the Chebyshev antiderivative's T₀ row is dense (the constant of
        # integration depends on every input mode), so no finite domain can be inferred
        @test domain(∫¹, Chebyshev(3)) == EmptySpace()
        @test domain(Integral(0), Chebyshev(3)) == Chebyshev(3)
        @test codomain(∫¹, Chebyshev(3)) == Chebyshev(4)

        # n = 0: identity, and (unlike Taylor/Fourier) the coefficient type is preserved
        a_int = Sequence(Chebyshev(2), [1, 2, 3])
        c_int = integrate(a_int, 0)
        @test eltype(c_int) == Int
        @test c_int == a_int

        # n = 1, order(a) = 0: ∫a₀dx ↦ [a₀, a₀/2]
        a0 = Sequence(Chebyshev(0), [2.0])
        @test ∫¹(a0) == project(∫¹, Chebyshev(0), Chebyshev(1), Float64)(a0) ==
            integrate!(Sequence(Chebyshev(1), [Inf, Inf]), a0) ==
            mul!(Sequence(Chebyshev(1), [Inf, Inf]), ∫¹, a0) == Sequence(Chebyshev(1), [2.0, 1.0])

        # n = 1, order(a) = 1
        a1 = Sequence(Chebyshev(1), [1.0, 3.0])
        @test ∫¹(a1) == project(∫¹, Chebyshev(1), Chebyshev(2), Float64)(a1) ==
            integrate!(Sequence(Chebyshev(2), [Inf, Inf, Inf]), a1) ==
            mul!(Sequence(Chebyshev(2), [Inf, Inf, Inf]), ∫¹, a1) == Sequence(Chebyshev(2), [-0.5, 0.5, 0.75])

        # n = 1, order(a) = 2 (general recursive branch; floating-point summation order ⇒ ≈)
        a_𝒞 = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        @test project(∫¹, Chebyshev(2), Chebyshev(3), Float64)(a_𝒞) ≈
            mul!(Sequence(Chebyshev(3), [Inf, Inf, Inf, Inf]), ∫¹, a_𝒞) ==
            ∫¹(a_𝒞) == integrate!(Sequence(Chebyshev(3), [Inf, Inf, Inf, Inf]), a_𝒞)

        # n = 1, order(a) = 3 (exercises both loops in the general branch; all dyadic ⇒ exact)
        a3 = Sequence(Chebyshev(3), [1.0, 2.0, 3.0, 4.0])
        @test ∫¹(a3) == integrate!(Sequence(Chebyshev(4), [Inf, Inf, Inf, Inf, Inf]), a3) ==
            mul!(Sequence(Chebyshev(4), [Inf, Inf, Inf, Inf, Inf]), ∫¹, a3) ==
            Sequence(Chebyshev(4), [-1.0, -1.0, -0.5, 0.5, 0.5])

        # derivative ∘ integral == identity (exact, no information lost)
        @test differentiate(∫¹(a3)) == a3

        # `project` builds the matrix entrywise via `getcoefficient`: n = 0 (identity), and for
        # n = 1 with i = 0 the odd-j ≥ 3 branch (j = 1 is a separate case, and j = 2 already
        # exercises the even branch above via `a_𝒞`/`a3`)
        @test project(Integral(0), Chebyshev(2), Chebyshev(2), Float64)(a_int) == Sequence(Chebyshev(2), Float64.(coefficients(a_int)))
        @test project(∫¹, Chebyshev(3), Chebyshev(4), Float64)(a3) == ∫¹(a3) # i = 0, j = 3 (odd, ≥ 3)

        # n ≥ 2 is an explicit TODO restriction in the source
        @test_throws DomainError integrate(a_𝒞, 2)

        # ComplexF64 coefficients (exact, dyadic)
        a_𝒞c = Sequence(Chebyshev(1), ComplexF64[2 + 2im, 4.0])
        @test ∫¹(a_𝒞c) == Sequence(Chebyshev(2), ComplexF64[0 + 2im, 1 + 1im, 1.0])

        # Interval{Float64} coefficients (dyadic ⇒ degenerate intervals, `==` is safe)
        a_𝒞i = Sequence(Chebyshev(1), interval.([1.0, 3.0]))
        @test integrate(a_𝒞i) == Sequence(Chebyshev(2), interval.([-0.5, 0.5, 0.75]))
    end

    @testset "Tensor space" begin
        a_𝑇 = Sequence(Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(2), collect(1.0:27.0))
        selectdim(a_𝑇, 2, 0) .= 0.0 # enforce the Fourier mean-free requirement along dimension 2
        ∫₁₁₁ = Integral((1, 1, 1))
        c_𝑇 = integrate(a_𝑇, (1, 1, 1))
        @test project(∫₁₁₁, space(a_𝑇), codomain(∫₁₁₁, space(a_𝑇)), ComplexF64) * a_𝑇 ≈ c_𝑇 == ∫₁₁₁(a_𝑇)
        @test differentiate(c_𝑇, (1, 1, 1)) ≈ a_𝑇 # derivative ∘ integral == identity (Chebyshev factor ⇒ ≈)

        # guard propagates through each tensor dimension: nonzero Fourier mean throws DomainError
        a_bad = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(1.0:6.0)) # Fourier is dimension 2
        @test_throws DomainError integrate(a_bad, (0, 1))
        a_bad2 = Sequence(Fourier(1, 1.0) ⊗ Taylor(1), collect(1.0:6.0)) # Fourier is dimension 1
        @test_throws DomainError integrate(a_bad2, (1, 0))

        @testset "Array-based application per tensor dimension" begin
            # `_apply!`/`_apply` for Integral on a `TensorSpace` dispatch differently depending on
            # whether a factor sits at dimension 1 (in-place `_apply!`, hardcoded to dimension 1)
            # or at a later dimension (functional `_apply`, `Val{D}`). The (1,1,1) case above only
            # ever exercises n = 1 for whichever factor happens to occupy each dimension; cross-check
            # `project` (built entrywise via the independent `getcoefficient` code path) against the
            # action for a spread of factor positions and orders n

            # Taylor at dim 1 (n = 0, n ≥ 2), Fourier at dim 2 (n = 0, odd n ≥ 3, even n ≥ 2),
            # Chebyshev(order 1) at dim 3 (n = 1)
            s_𝑇a = Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1)
            a_𝑇a = Sequence(s_𝑇a, collect(1.0:12.0))
            selectdim(a_𝑇a, 2, 0) .= 0.0 # Fourier mean-free
            for α ∈ ((0,1,1), (2,1,1), (1,0,1), (1,3,1), (1,2,1))
                ℐα = Integral(α)
                @test project(ℐα, s_𝑇a, codomain(ℐα, s_𝑇a), ComplexF64) * a_𝑇a ≈ integrate(a_𝑇a, α) == ℐα(a_𝑇a)
            end

            # Fourier at dim 1 (n = 0, n = 1, odd n ≥ 3, even n ≥ 2), Taylor at dim 2 (n ≥ 2),
            # Chebyshev(order 3) at dim 3 (n = 0, and the general n = 1 branch with both loops
            # non-empty since order ≥ 3)
            s_𝑇b = Fourier(1, 1.0) ⊗ Taylor(1) ⊗ Chebyshev(3)
            a_𝑇b = Sequence(s_𝑇b, collect(1.0:24.0))
            selectdim(a_𝑇b, 1, 0) .= 0.0 # Fourier mean-free
            for α ∈ ((0,2,1), (1,0,0), (3,0,0), (2,0,0))
                ℐα = Integral(α)
                @test project(ℐα, s_𝑇b, codomain(ℐα, s_𝑇b), ComplexF64) * a_𝑇b ≈ integrate(a_𝑇b, α) == ℐα(a_𝑇b)
            end

            # Chebyshev(order 0) at dim 3, n = 1 (the ord == 0 branch, distinct from order 1 and
            # order ≥ 3 above)
            s_𝑇c = Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0)
            a_𝑇c = Sequence(s_𝑇c, collect(1.0:6.0))
            selectdim(a_𝑇c, 2, 0) .= 0.0 # Fourier mean-free
            ℐ111c = Integral((1, 1, 1))
            @test project(ℐ111c, s_𝑇c, codomain(ℐ111c, s_𝑇c), ComplexF64) * a_𝑇c ≈ integrate(a_𝑇c, (1, 1, 1)) == ℐ111c(a_𝑇c)
        end
    end

    @testset "Cartesian space" begin
        a = Sequence(Taylor(2)^2 × Fourier(1, 1.0) × Chebyshev(2), collect(1.0:6.0+3.0+3.0))
        component(a, 2)[0] = 0.0 # Fourier component mean-free
        ∫¹ = Integral(1)
        c = ∫¹(a)
        @test c == integrate!(Sequence(Taylor(3)^2 × Fourier(1, 1.0) × Chebyshev(3), fill(complex(Inf), 8+3+4)), a) ==
            mul!(Sequence(Taylor(3)^2 × Fourier(1, 1.0) × Chebyshev(3), fill(complex(Inf), 8+3+4)), ∫¹, a)
        @test component(c, 1) == integrate(component(a, 1))
        @test component(c, 3) == integrate(component(a, 3))
    end

    @testset "Symmetric space" begin
        # only Fourier's symmetric bookkeeping is implemented (mirrors Derivative)
        sE = evensym(Fourier(3, 1.0))
        sO = oddsym(Fourier(3, 1.0))
        ∫¹ = Integral(1)

        # ∫ maps evensym Fourier to oddsym Fourier, same group transform as Derivative
        @test codomain(∫¹, sE) == sO
        @test codomain(Integral(2), sE) == sE # (-1)² = 1: order-2 integral preserves the symmetry

        # `domain(∫, codom)` inverts the group transform; the diagonal mean-free convention makes
        # the domain well-defined for any Fourier symmetry (validity of a₀ is checked at action time)
        @test domain(∫¹, sE) == sO
        @test domain(∫¹, sO) == sE
        @test domain(Integral(2), sE) == sE
        @test domain(Integral(2), sO) == sO

        a = Sequence(sE, ComplexF64[0.0, 2.0, 0.0, 3.0]) # a₀=0, a₁=a₋₁=2, a₃=a₋₃=3 (a₂=a₋₂=0), mean-free
        out = Sequence(sO, fill(complex(Inf), dimension(sO)))
        expected = Sequence(sO, ComplexF64[-2.0im, 0.0, -1.0im]) # a_j/(iωj): a₁/i=-2i, a₃/(3i)=-i
        @test ∫¹(a) == project(∫¹, sE, sO, ComplexF64)(a) == integrate!(out, a) ==
            mul!(Sequence(sO, fill(complex(Inf), dimension(sO))), ∫¹, a) == expected

        # cross-check against the desymmetrized (full Fourier) computation
        full_a = Sequence(desymmetrize(sE), [RadiiPolynomial.getcoefficient(a, (desymmetrize(sE), j)) for j ∈ -3:3])
        full_expected = Sequence(desymmetrize(sO), [RadiiPolynomial.getcoefficient(expected, (desymmetrize(sO), j)) for j ∈ -3:3])
        @test integrate(full_a) == full_expected

        # oddsym is naturally mean-free (index 0 is not a representative), so `differentiate ∘ integrate`
        # round-trips without any guard tripping
        @test differentiate(∫¹(a)) == a

        # guard: nonzero mean throws DomainError, same convention as the plain Fourier case
        a_bad = Sequence(sE, ComplexF64[1.0, 2.0, 0.0, 3.0])
        @test_throws DomainError integrate(a_bad)
        @test_throws DomainError ∫¹(a_bad)
    end

    @testset "InfiniteSequence error propagation" begin
        X = Ell1(GeometricWeight(2.0))

        # Taylor integral: finite ν^α/α!, tail ν^α (N+1)!/(N+α+1)!, total ν^α/α!
        a = InfiniteSequence(Sequence(Taylor(2), [1.0, 1.0, 1.0]), 0.5, 0.25, 0.75, X)
        ∫a = integrate(a)
        @test finite_error(∫a) == 2.0 * 0.5
        @test tail_error(∫a) == 0.5 * 0.25
        @test total_error(∫a) == min(2.0 * 0.5 + 0.5 * 0.25, 2.0 * 0.75)
        @test banachspace(∫a) == X # unlike `differentiate`, `integrate` does not force `Ell1()`

        # guaranteed-zero errors skip the factors: error-free Fourier integration works
        b = InfiniteSequence(Sequence(Fourier(1, 1.0), [0.5im, 0.0, -0.5im]), X)
        ∫b = integrate(b)
        @test sequence(∫b) == integrate(sequence(b))
        @test iszero(total_error(∫b))

        # the skip is genuine: with a weight that has no `_integral_*_error` method at all,
        # zero errors still integrate successfully because the factor is never computed
        X_id = Ell1(IdentityWeight())
        b_id = InfiniteSequence(Sequence(Fourier(1, 1.0), [0.5im, 0.0, -0.5im]), X_id)
        ∫b_id = integrate(b_id)
        @test sequence(∫b_id) == integrate(sequence(b_id))
        @test banachspace(∫b_id) == X_id

        # tensor tail factor covers the box complement: max_i tailᵢ · ∏_{l≠i} totalₗ
        X² = Ell1((GeometricWeight(2.0), GeometricWeight(2.0)))
        s² = Taylor(2) ⊗ Taylor(2)
        @test RadiiPolynomial._integral_finite_error(X², s², (1, 1)) == 2.0 * 2.0
        @test RadiiPolynomial._integral_total_error(X², s², (1, 1)) == 2.0 * 2.0
        @test RadiiPolynomial._integral_tail_error(X², s², (1, 1)) == 0.5 * 2.0

        a² = InfiniteSequence(Sequence(s², ones(9)), 0.0, 1.0, 1.0, X²)
        ∫a² = integrate(a², (1, 1))
        @test tail_error(∫a²) == 1.0

        # Fourier integral (mean-free antiderivative): finite ω^{-α} (0 if N = 0),
        # tail (ω(N+1))^{-α}, total ω^{-α} (no N = 0 special-case for the total)
        f = InfiniteSequence(Sequence(Fourier(3, 0.5), [0, 0, 0.5im, 0, -0.5im, 0, 0]), 0.25, 1.0, 1.25, X)
        ∫f = integrate(f)
        @test sequence(∫f) == integrate(sequence(f))
        @test finite_error(∫f) == 2.0 * 0.25
        @test tail_error(∫f) == 0.5 * 1.0
        @test total_error(∫f) == min(2.0 * 0.25 + 0.5 * 1.0, 2.0 * 1.25)
        @test RadiiPolynomial._integral_tail_error(X, Fourier(3, 0.5), 2) == 0.25

        # finite/total asymmetry for a Fourier space with no nonzero mode (N = 0)
        @test RadiiPolynomial._integral_finite_error(X, Fourier(0, 0.5), 1) == 0.0
        @test RadiiPolynomial._integral_total_error(X, Fourier(0, 0.5), 1) == inv(0.5)

        # Chebyshev error propagation is unconditionally unimplemented (even for α = 0)
        @test_throws DomainError RadiiPolynomial._integral_finite_error(X, Chebyshev(2), 1)
        @test_throws DomainError RadiiPolynomial._integral_tail_error(X, Chebyshev(2), 0)
        @test_throws DomainError RadiiPolynomial._integral_total_error(X, Chebyshev(2), 1)
        a_cheb = InfiniteSequence(Sequence(Chebyshev(2), [1.0, 1.0, 1.0]), 0.5, 0.25, 0.75, X)
        @test_throws DomainError integrate(a_cheb)
    end

end
