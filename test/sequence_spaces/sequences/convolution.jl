# value of `f()` under the :loop, then the :fft, convolution algorithm
function _conv_loop_fft(f)
    try
        set_conv_algorithm(:loop)
        loop = f()
        set_conv_algorithm(:fft)
        return loop, f()
    finally
        set_conv_algorithm(:loop)
    end
end

# sequence on `s` with the coefficient `f(k)` at the index `k`
_conv_seq(f, s) = Sequence(s, [f(k) for k ∈ indices(s)])

@testset "Convolution" begin

    @testset "set_conv_algorithm" begin
        @test RadiiPolynomial.CONV_ALGORITHM[] == :loop # package default
        @test_throws ArgumentError set_conv_algorithm(:bogus)
        @test RadiiPolynomial.CONV_ALGORITHM[] == :loop
        @test set_conv_algorithm(:fft) == :fft == RadiiPolynomial.CONV_ALGORITHM[]
        @test set_conv_algorithm(:loop) == :loop == RadiiPolynomial.CONV_ALGORITHM[]
    end

    @testset "$(nameof(typeof(space(a))))" for (a, b, ab, a², mb, pb) ∈ (
            # (1+2x)(1+2x+3x²) = 1 + 4x + 7x² + 6x³ and (1+2x)² = 1 + 4x + 4x²
            (Sequence(Taylor(1), [1.0, 2.0]), Sequence(Taylor(2), [1.0, 2.0, 3.0]),
                Sequence(Taylor(3), [1.0, 4.0, 7.0, 6.0]), Sequence(Taylor(2), [1.0, 4.0, 4.0]),
                Sequence(Taylor(1), [1.0, 4.0]), Sequence(Taylor(1), [1.0, 4.0])),
            # Laurent series in z = e^{iθ}: (z⁻¹+2+3z)(4z⁻¹+5+6z) and (z⁻¹+2+3z)²
            (Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0]), Sequence(Fourier(1, 1.0), [4.0, 5.0, 6.0]),
                Sequence(Fourier(2, 1.0), [4.0, 13.0, 28.0, 27.0, 18.0]), Sequence(Fourier(2, 1.0), [1.0, 4.0, 10.0, 12.0, 9.0]),
                Sequence(Fourier(1, 1.0), [13.0, 28.0, 27.0]), Sequence(Fourier(1, 1.0), [4.0, 10.0, 12.0])),
            # the coefficients extend to negative indices by a₋ₖ = aₖ: c[k] = Σⱼ a[|k-j|] b[|j|], e.g.
            # c[0] = a[1]b[1] + a[0]b[0] + a[1]b[1] = 9 and c[1] = a[1]b[0] + a[0]b[1] + a[1]b[2] = 10
            (Sequence(Chebyshev(1), [1.0, 2.0]), Sequence(Chebyshev(2), [1.0, 2.0, 3.0]),
                Sequence(Chebyshev(3), [9.0, 10.0, 7.0, 6.0]), Sequence(Chebyshev(2), [9.0, 4.0, 4.0]),
                Sequence(Chebyshev(1), [9.0, 10.0]), Sequence(Chebyshev(1), [9.0, 4.0])),
        )
        @test a * b == ab
        @test a^2 == a * a == a²
        @test a^1 == copy(a) == a
        @test a^0 == one(a) == _conv_seq(k -> float(k == 0), space(a))
        n = -1 # a literal exponent is lowered to `inv(a)` and never reaches the guard of `^`
        @test_throws DomainError a^n

        # mul_bar and pow_bar truncate the product to the intersection of the spaces
        @test mul_bar(a, b) == mb
        @test pow_bar(a, 2) == pb
        @test pow_bar(a, 3) == mul_bar(a, mul_bar(a, a))
        @test pow_bar(a, 1) == copy(a)
        @test pow_bar(a, 0) == one(a)
        @test_throws DomainError pow_bar(a, -1)
        @test all(n -> codomain(pow_bar, space(a), n) == space(a), 3:5)
    end

    @testset "powers" begin
        a = Sequence(Taylor(1), [1.0, 2.0])
        @test pow_bar(a, 3) == Sequence(Taylor(1), [1.0, 6.0]) # (1+2x)³ = 1 + 6x + 12x² + 8x³

        # power by squaring agrees with repeated multiplication
        c = Sequence(Taylor(1), [1.0, 1.0])
        @test all(n -> c^n == foldl(*, fill(c, n)), 2:6)
    end

    @testset "frequency mismatch" begin
        a, b = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0]), Sequence(Fourier(1, 2.0), [1.0, 2.0, 3.0])
        @test_throws ArgumentError a * b
        @test_throws ArgumentError mul_bar(a, b)
    end

    @testset "complex and interval coefficients" begin
        # z⁻²: 1·i = i, z⁻¹: 1·1+2·i = 1+2i, z⁰: 1·2+2·1+i·i = 3, z¹: 2·2+i·1 = 4+i, z²: i·2 = 2i
        ac = Sequence(Fourier(1, 1.0), ComplexF64[1.0, 2.0, im])
        bc = Sequence(Fourier(1, 1.0), ComplexF64[im, 1.0, 2.0])
        @test ac * bc == Sequence(Fourier(2, 1.0), ComplexF64[im, 1+2im, 3, 4+im, 2im])

        # (1/3 + (2/3)x)(1/7 + (3/7)x) = 1/21 + (5/21)x + (2/7)x² and (1/3 + (2/3)x)² = 1/9 + (4/9)x + (4/9)x²
        ia = Sequence(Taylor(1), [interval(Float64, 1//3), interval(Float64, 2//3)])
        ib = Sequence(Taylor(1), [interval(Float64, 1//7), interval(Float64, 3//7)])
        for (c, exact) ∈ ((ia * ib, (1//21, 5//21, 2//7)), (mul_bar(ia, ib), (1//21, 5//21)), (pow_bar(ia, 2), (1//9, 4//9)))
            @test all(in_interval.(exact, coefficients(c)))
        end

        # (1/3 + i/4)(1/5 + i/6) = 1/40 + 19i/180
        a = Sequence(Taylor(0), [complex(interval(Float64, 1//3), interval(Float64, 1//4))])
        b = Sequence(Taylor(0), [complex(interval(Float64, 1//5), interval(Float64, 1//6))])
        @test in_interval(1//40, real((a * b)[0]))
        @test in_interval(19//180, imag((a * b)[0]))
    end

    @testset "TensorSpace" begin
        # the convolution of outer products is the outer product of the convolutions
        aT, bT = Sequence(Taylor(1), [1.0, 2.0]), Sequence(Taylor(2), [1.0, 2.0, 3.0])
        aC, bC = Sequence(Chebyshev(1), [1.0, 2.0]), Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
        outer(x, y) = _conv_seq(((i, k),) -> x[i] * y[k], space(x) ⊗ space(y))
        a, b = outer(aT, aC), outer(bT, bC)

        @test a * b == outer(aT * bT, aC * bC)
        @test a^2 == a * a
        @test mul_bar(a, b) == outer(mul_bar(aT, bT), mul_bar(aC, bC))
    end

    # the four axis-aligned modes of order 1 share one value, which is d4-invariant
    d4_full = _conv_seq(k -> float(sum(abs, k) == 1), Fourier(2, 1.0) ⊗ Fourier(2, 1.0))
    d4_sym = _conv_seq(k -> d4_full[k], d4sym(space(d4_full)))

    @testset "SymmetricSpace" begin
        e₁, e₂ = Sequence(Taylor(2), [1.0, 0.0, 3.0]), Sequence(Taylor(2), [2.0, 0.0, 4.0]) # 1 + 3x², 2 + 4x²
        o₁, o₂ = Sequence(Taylor(2), [0.0, 2.0, 0.0]), Sequence(Taylor(2), [0.0, 3.0, 0.0]) # 2x, 3x
        even(a) = _conv_seq(k -> a[k], evensym(space(a)))
        odd(a) = _conv_seq(k -> a[k], oddsym(space(a)))

        for (x, y, full, idx) ∈ ((even(e₁), even(e₂), e₁ * e₂, [0, 2, 4]),
                                 (odd(o₁), odd(o₂), o₁ * o₂, [0, 2, 4]),
                                 (even(e₁), odd(o₁), e₁ * o₁, [1, 3]))
            @test collect(indices(space(x * y))) == idx
            @test all(k -> (x * y)[k] == full[k], idx)
        end
        @test desymmetrize(space(even(e₁) * even(e₂))) == Taylor(4)

        p_sym = d4_sym * d4_sym
        @test desymmetrize(space(p_sym)) == Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
        @test all(k -> p_sym[k] == (d4_full * d4_full)[k], indices(space(p_sym)))
    end

    @testset ":fft agrees with :loop" begin
        geometric = _conv_seq(i -> 16.0^-i, Taylor(10))
        sparse = _conv_seq(k -> float(k == (0, 2) || k == (0, 3)), Taylor(2) ⊗ Chebyshev(3))
        products(x, y) = (x * y, x^2, x^3, mul_bar(x, y), pow_bar(x, 3))
        product(x, y) = (x * y,) # mul_bar and pow_bar are not defined on a SymmetricSpace
        cases = (
            (products, Sequence(Taylor(3), [1.0, 2.0, 0.0, 3.0]), Sequence(Taylor(3), [2.0, 0.0, 1.0, 1.0]), 1e-10),
            (products, geometric, copy(geometric), 1e-9),
            # under :fft a symmetric sequence is expanded on its full space and the product projected back
            (product, Sequence(evensym(Taylor(2)), [1.0, 3.0]), Sequence(evensym(Taylor(2)), [2.0, 4.0]), 1e-9),
            (product, d4_sym, d4_sym, 1e-9),
            # overlapping sparse supports: TᵢTᵢ contributes to T₀, which the cleanup of the aliased
            # coefficients must not zero
            (products, Sequence(Chebyshev(3), [0.0, 0.0, 1.0, 1.0]), Sequence(Chebyshev(3), [0.0, 0.0, 0.0, 1.0]), 1e-12),
            (products, Sequence(Taylor(3), [0.0, 0.0, 1.0, 1.0]), Sequence(Taylor(3), [0.0, 0.0, 0.0, 1.0]), 1e-12),
            (products, Sequence(Fourier(3, 1.0), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]), Sequence(Fourier(3, 1.0), [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), 1e-12),
            (products, sparse, _conv_seq(k -> float(k == (0, 3)), space(sparse)), 1e-12),
        )
        for (f, x, y, tol) ∈ cases
            loop, fft = _conv_loop_fft(() -> f(x, y))
            for (c_loop, c_fft) ∈ zip(loop, fft)
                @test space(c_fft) == space(c_loop)
                @test coefficients(c_fft) ≈ coefficients(c_loop) atol=tol
            end
        end
    end

    @testset "banach_rounding!" begin
        # the tail of the :fft product beyond the fitted rounding order is replaced by the Banach
        # algebra bound, which is 0.0 for floats although the exact coefficient is positive
        taylor = _conv_seq(i -> 16.0^-i, Taylor(10))
        algebraic = _conv_seq(i -> inv((1.0 + i)^20.0), Taylor(10))
        @test RadiiPolynomial.weight(taylor) isa RadiiPolynomial.GeometricWeight
        @test rate(RadiiPolynomial.weight(taylor)) > 1
        @test RadiiPolynomial.weight(algebraic) isa RadiiPolynomial.AlgebraicWeight

        for (a, tail) ∈ (
                (taylor, (15,)), # rounding order 15
                (algebraic, (7,)), # rounding order 7
                (_conv_seq(i -> 16.0^-abs(i), Fourier(10, 1.0)), (16, -16)), # written at ±i at once
                # rounding order (17, 17), and 12/17 + 12/17 ≥ 1
                (_conv_seq(((i, k),) -> 16.0^-i * 16.0^-abs(k), Taylor(6) ⊗ Fourier(6, 1.0)), ((12, 12),)),
            )
            loop, fft = _conv_loop_fft(() -> a * copy(a))
            @test coefficients(fft) ≈ coefficients(loop) atol=1e-9
            @test all(k -> fft[k] == 0.0 && loop[k] > 0, tail)
        end

        # c[k] = Σⱼ a[k-j]a[j] = (k+1) 16⁻ᵏ for k ≤ 10 and 6·16⁻¹⁵ for k = 15
        loop, fft = _conv_loop_fft(() -> taylor * copy(taylor))
        @test fft[0] == 1.0
        @test loop[15] ≈ 6 * 16.0^-15

        @testset "the product is rounded relative to its scale" begin
            # coefficients of size 1e-9 have a product below eps(Float64) in absolute terms, which is no noise
            for a ∈ (Sequence(Taylor(2), [1e-9, 2e-9, 3e-9]), _conv_seq(i -> 1e-9 * 16.0^-i, Taylor(10)))
                loop, fft = _conv_loop_fft(() -> a * a)
                @test coefficients(fft)[1:3] ≈ coefficients(loop)[1:3] rtol=1e-10
                _, ifft = _conv_loop_fft(() -> interval.(a) * interval.(a))
                @test all(k -> in_interval(loop[k], ifft[k]), 0:2)
                @test maximum(radius, coefficients(ifft)[1:3]) < 1e-30
            end
        end

        @testset "interval coefficients: the rounded tail encloses the exact coefficients" begin
            # 16⁻ⁱ is a float, so the exact convolution is a rational computed independently
            q(i) = Rational{BigInt}(1, big(16)^abs(i))
            exact_taylor(k) = sum(q(k - j) * q(j) for j ∈ max(k - 10, 0):min(k, 10))
            exact_fourier(k) = sum(q(k - j) * q(j) for j ∈ max(k - 10, -10):min(k + 10, 10))

            a = interval.(taylor)
            loop, fft = _conv_loop_fft(() -> a * copy(a))
            @test in_interval(1, fft[0])
            @test in_interval(exact_taylor(15), fft[15])
            @test in_interval(exact_taylor(15), loop[15])

            af = _conv_seq(i -> complex(interval(16.0^-abs(i))), Fourier(10, 1.0))
            loop, fft = _conv_loop_fft(() -> af * copy(af))
            @test in_interval(exact_fourier(0), real(fft[0]))
            @test in_interval(0, imag(fft[0]))
            @test in_interval(exact_fourier(16), real(fft[16]))
            @test in_interval(exact_fourier(16), real(loop[16]))
        end
    end

    @testset "InfiniteSequence: product of symmetric sequences" begin
        # the finite part carries the symmetry of the product, with the order of the full product: odd times odd is even
        F = Fourier(4, 1.0)
        X = Ell1(GeometricWeight(interval(11) / interval(10)))
        ao = InfiniteSequence(interval(Sequence(oddsym(F), [0.5, 0.3, 0.2, 0.1])), X; total_error = interval(1e-6))
        ap = InfiniteSequence(project(sequence(ao), interval(F)), X; total_error = interval(1e-6))
        co, cp = ao * ao, ap * ap
        @test space(co) == evensym(interval(Fourier(8, 1.0)))
        @test all(k -> mid(real(project(sequence(co), interval(Fourier(8, 1.0)))[k])) ≈ mid(sequence(cp)[k]), -8:8)
        @test mid(total_error(co)) ≈ mid(total_error(cp))
        # factors with different symmetries whose product is representable on neither
        σ = GroupElement(LatticeAut([-1;;]), Cocycle(0//1, Rational{Int}[0//1]))
        s⁺ = SymmetricSpace(F, Group(σ, GroupElement(LatticeAut([1;;]), Cocycle(0//1, Rational{Int}[1//1])))) # even modes
        s⁻ = SymmetricSpace(F, Group(σ, GroupElement(LatticeAut([1;;]), Cocycle(1//1, Rational{Int}[1//1])))) # odd modes
        a = InfiniteSequence(interval(Sequence(s⁺, [1.0, 0.5, 0.2])), X; total_error = interval(1e-6))
        b = InfiniteSequence(interval(Sequence(s⁻, [0.4, 0.1])), X; total_error = interval(1e-6))
        @test collect(indices(space(a * b))) == [1, 3, 5, 7]
    end

    @testset "InfiniteSequence: power by squaring" begin
        # the coefficients and errors are dyadic, so every product below is exact and the power
        # equals the association of `*` it is computed with, in every field
        a = InfiniteSequence(Sequence(Taylor(1), [1.0, 2.0]), 0.0, 0.5, 0.5, Ell1())
        a² = a * a
        for (n, ref) ∈ ((3, a * a²), (6, a² * (a² * a²)))
            r = a^n
            @test sequence(r) == sequence(ref) == Sequence(Taylor(n), [binomial(n, k) * 2.0^k for k ∈ 0:n]) # (1+2x)ⁿ
            @test finite_error(r) == finite_error(ref)
            @test tail_error(r) == tail_error(ref)
            @test total_error(r) == total_error(ref)
            @test sequence_norm(r) == sequence_norm(ref)
            @test r.full_norm == ref.full_norm
        end
    end

    @testset "interval enclosures of tensor products" begin
        # the product is accumulated with one nested sum per dimension so that every partial sum
        # stays small, where a flat loop over all index pairs loses about one ulp of the running sum
        # per addition and gives enclosures an order of magnitude wider
        N, K = 32, 20
        S = Chebyshev(N) ⊗ Fourier(K, 1.0)
        mk(ϕ) = Sequence(S, vec([(m = 0.5^n * 0.7^abs(k) * sinpi(ϕ * (n + 1) + 0.3 * k); r = 1e-16 * 0.5^n * 0.7^abs(k); interval(m - r, m + r)) for n ∈ 0:N, k ∈ -K:K]))
        a, b = mk(0.37), mk(0.71)

        function flat_conv(a, b, space_c)
            dsa, dsb = space(a), space(b)
            c = zeros(eltype(a), space_c)
            for k ∈ indices(space_c), j ∈ RadiiPolynomial._convolution_indices(dsa, dsb, k)
                c[k] += a[RadiiPolynomial._extract_valid_index(dsa, k, j)] * b[RadiiPolynomial._extract_valid_index(dsb, j)]
            end
            return c
        end

        for (c, c_flat) ∈ ((a * b, flat_conv(a, b, codomain(*, S, S))),
                           (mul_bar(a, b), flat_conv(a, b, codomain(mul_bar, S, S))),
                           (a ^ 2, flat_conv(a, a, codomain(^, S, 2))))
            @test mid.(coefficients(c)) ≈ mid.(coefficients(c_flat))
            @test maximum(radius, coefficients(c)) ≤ maximum(radius, coefficients(c_flat)) / 10
            @test sum(radius, coefficients(c)) ≤ sum(radius, coefficients(c_flat)) / 10
        end
        @test maximum(radius, coefficients(a * b)) < 1e-14 # the flat loop gives ≈ 8e-14
    end
end
