@testset "Convolution" begin

    @testset "set_conv_algorithm" begin
        @test RadiiPolynomial.CONV_ALGORITHM[] == :loop # package default

        @test_throws ArgumentError set_conv_algorithm(:bogus)
        @test RadiiPolynomial.CONV_ALGORITHM[] == :loop # unchanged after the failed attempt

        @test set_conv_algorithm(:fft) == :fft
        @test RadiiPolynomial.CONV_ALGORITHM[] == :fft

        @test set_conv_algorithm(:loop) == :loop
        @test RadiiPolynomial.CONV_ALGORITHM[] == :loop
    end

    @testset "Taylor" begin
        𝒯₁ = Taylor(1)
        𝒯₂ = Taylor(2)
        a = Sequence(𝒯₁, [1.0, 2.0]) # 1 + 2x
        b = Sequence(𝒯₂, [1.0, 2.0, 3.0]) # 1 + 2x + 3x²

        # (1+2x)(1+2x+3x²) = (1+2x+3x²) + (2x+4x²+6x³) = 1 + 4x + 7x² + 6x³
        @test a * b == Sequence(Taylor(3), [1.0, 4.0, 7.0, 6.0])
        # (1+2x)² = 1 + 4x + 4x²
        @test a^2 == a * a == Sequence(𝒯₂, [1.0, 4.0, 4.0])
        @test a^1 == copy(a) == a
        @test a^0 == one(a) == Sequence(𝒯₁, [1.0, 0.0])
        # `n` must not be a literal integer here: Julia's `literal_pow` intercepts
        # compile-time literals like `a^(-1)` and calls `inv(a)` directly, bypassing
        # the custom `^` method (and its DomainError guard) entirely.
        n = -1
        @test_throws DomainError a^n

        # mul_bar truncates the full product down to intersect(space(a), space(b)) = Taylor(1)
        @test mul_bar(a, b) == Sequence(𝒯₁, [1.0, 4.0])
        # pow_bar(a,2) truncates a² = 1+4x+4x² down to Taylor(1)
        @test pow_bar(a, 2) == Sequence(𝒯₁, [1.0, 4.0])
        # pow_bar composes as mul_bar(a, mul_bar(a,a)); (1+2x)³ = 1+6x+12x²+8x³ truncated to order 1
        @test pow_bar(a, 3) == mul_bar(a, mul_bar(a, a)) == Sequence(𝒯₁, [1.0, 6.0])
        @test pow_bar(a, 1) == copy(a)
        @test pow_bar(a, 0) == one(a)
        @test_throws DomainError pow_bar(a, -1)

        @testset "codomain(pow_bar, s, n): recursive branch for n ≥ 3" begin
            # the codomain of mul_bar is an intersection, and intersect(𝒯₁, 𝒯₁) = 𝒯₁ is
            # idempotent, so the codomain stays 𝒯₁ for every n ≥ 1
            @test codomain(pow_bar, 𝒯₁, 3) == 𝒯₁
            @test codomain(pow_bar, 𝒯₁, 4) == 𝒯₁
            @test codomain(pow_bar, 𝒯₁, 5) == 𝒯₁
        end

        @testset "power-by-squaring agrees with repeated pairwise multiplication" begin
            c = Sequence(Taylor(1), [1.0, 1.0]) # 1 + x
            for n ∈ 2:6
                rhs = c
                for _ ∈ 2:n
                    rhs = rhs * c
                end
                @test c^n == rhs
            end
        end

        @testset "Interval{Float64} coefficients: exact rational enclosure" begin
            ia = Sequence(𝒯₁, [interval(Float64, 1//3), interval(Float64, 2//3)]) # 1/3 + (2/3)x
            ib = Sequence(𝒯₁, [interval(Float64, 1//7), interval(Float64, 3//7)]) # 1/7 + (3/7)x
            ic = ia * ib
            # exact product: c₀=1/21, c₁=(1/3)(3/7)+(2/3)(1/7)=5/21, c₂=(2/3)(3/7)=2/7
            for (k, ex) ∈ zip(0:2, (1//21, 5//21, 2//7))
                @test in_interval(ex, ic[k])
            end

            imb = mul_bar(ia, ib)
            for (k, ex) ∈ zip(0:1, (1//21, 5//21))
                @test in_interval(ex, imb[k])
            end

            # (1/3+(2/3)x)² = 1/9 + (4/9)x + (4/9)x², truncated to order 1
            ipb = pow_bar(ia, 2)
            for (k, ex) ∈ zip(0:1, (1//9, 4//9))
                @test in_interval(ex, ipb[k])
            end
        end
    end

    @testset "Fourier" begin
        ℱ₁ = Fourier(1, 1.0)
        a = Sequence(ℱ₁, [1.0, 2.0, 3.0]) # a₋₁=1, a₀=2, a₁=3
        b = Sequence(ℱ₁, [4.0, 5.0, 6.0]) # b₋₁=4, b₀=5, b₁=6

        # treat a, b as Laurent series in z=e^{iθ}: (z⁻¹+2+3z)(4z⁻¹+5+6z)
        # z⁻²: 1·4=4 ; z⁻¹: 1·5+2·4=13 ; z⁰: 1·6+2·5+3·4=28 ; z¹: 2·6+3·5=27 ; z²: 3·6=18
        @test a * b == Sequence(Fourier(2, 1.0), [4.0, 13.0, 28.0, 27.0, 18.0])
        # a² : z⁻²=1 ; z⁻¹=2·(1·2)=4 ; z⁰=2·(1·3)+2²=10 ; z¹=2·(2·3)=12 ; z²=3²=9
        @test a^2 == a * a == Sequence(Fourier(2, 1.0), [1.0, 4.0, 10.0, 12.0, 9.0])
        @test a^0 == one(a) == Sequence(ℱ₁, [0.0, 1.0, 0.0])
        n = -1 # a literal exponent would be intercepted before reaching the guard
        @test_throws DomainError a^n

        # mul_bar truncates back down to intersect(ℱ₁,ℱ₁) = Fourier(1)
        @test mul_bar(a, b) == Sequence(ℱ₁, [13.0, 28.0, 27.0])
        @test pow_bar(a, 2) == Sequence(ℱ₁, [4.0, 10.0, 12.0])

        @testset "frequency mismatch" begin
            b_wrong_freq = Sequence(Fourier(1, 2.0), [1.0, 2.0, 3.0])
            @test_throws ArgumentError a * b_wrong_freq
            @test_throws ArgumentError mul_bar(a, b_wrong_freq)
        end

        @testset "ComplexF64 coefficients" begin
            ac = Sequence(ℱ₁, ComplexF64[1.0, 2.0, im]) # a₋₁=1, a₀=2, a₁=i
            bc = Sequence(ℱ₁, ComplexF64[im, 1.0, 2.0]) # b₋₁=i, b₀=1, b₁=2
            # z⁻²: 1·i=i ; z⁻¹: 1·1+2·i=1+2i ; z⁰: 1·2+2·1+i·i=2+2-1=3
            # z¹: 2·2+i·1=4+i ; z²: i·2=2i
            @test ac * bc == Sequence(Fourier(2, 1.0), ComplexF64[im, 1+2im, 3, 4+im, 2im])
        end
    end

    @testset "Chebyshev" begin
        𝒞₁ = Chebyshev(1)
        𝒞₂ = Chebyshev(2)
        a = Sequence(𝒞₁, [1.0, 2.0]) # a₀=1, a₁=2
        b = Sequence(𝒞₂, [1.0, 2.0, 3.0]) # b₀=1, b₁=2, b₂=3

        # Chebyshev sequences are stored so that {a₀,2a₁,…,2aₙ} are the true Tₖ coefficients;
        # the product extends a, b to negative indices via a₋ₖ=aₖ and computes
        # c[k] = Σⱼ a[|k-j|]·b[|j|] for j ∈ [max(k-order(a),-order(b)), min(k+order(a),order(b))]
        # k=0, j∈{-1,0,1}: a[1]b[1] + a[0]b[0] + a[1]b[1] = 2·2+1·1+2·2 = 9
        # k=1, j∈{0,1,2}: a[1]b[0] + a[0]b[1] + a[1]b[2] = 2·1+1·2+2·3 = 10
        # k=2, j∈{1,2}:   a[1]b[1] + a[0]b[2]           = 2·2+1·3    = 7
        # k=3, j∈{2}:     a[1]b[2]                       = 2·3        = 6
        @test a * b == Sequence(Chebyshev(3), [9.0, 10.0, 7.0, 6.0])
        # a² : k=0: a[1]²+a[0]²+a[1]² = 4+1+4 = 9 ; k=1: 2·(a[1]a[0]) = 4 ; k=2: a[1]² = 4
        @test a^2 == a * a == Sequence(𝒞₂, [9.0, 4.0, 4.0])
        @test a^0 == one(a) == Sequence(𝒞₁, [1.0, 0.0])
        n = -1 # a literal exponent would be intercepted before reaching the guard
        @test_throws DomainError a^n

        # mul_bar truncates down to intersect(𝒞₁,𝒞₂) = Chebyshev(1)
        @test mul_bar(a, b) == Sequence(𝒞₁, [9.0, 10.0])
        @test pow_bar(a, 2) == Sequence(𝒞₁, [9.0, 4.0])
    end

    @testset "TensorSpace" begin
        # rank-1 (outer-product) inputs: the tensor convolution factors exactly into the
        # per-axis Cauchy products already checked above, i.e. (aT⊗aC)*(bT⊗bC) = (aT*bT)⊗(aC*bC)
        aT = Sequence(Taylor(1), [1.0, 2.0])
        bT = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        aC = Sequence(Chebyshev(1), [1.0, 2.0])
        bC = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])

        a = zeros(Taylor(1) ⊗ Chebyshev(1))
        for i ∈ indices(Taylor(1)), k ∈ indices(Chebyshev(1))
            a[(i, k)] = aT[i] * aC[k]
        end
        b = zeros(Taylor(2) ⊗ Chebyshev(2))
        for i ∈ indices(Taylor(2)), k ∈ indices(Chebyshev(2))
            b[(i, k)] = bT[i] * bC[k]
        end

        prodT, prodC = aT * bT, aC * bC
        expected = zeros(Taylor(3) ⊗ Chebyshev(3))
        for m ∈ indices(Taylor(3)), n ∈ indices(Chebyshev(3))
            expected[(m, n)] = prodT[m] * prodC[n]
        end
        @test a * b == expected
        @test a^2 == a * a

        mbT, mbC = mul_bar(aT, bT), mul_bar(aC, bC)
        mb = mul_bar(a, b)
        expected_mb = zeros(space(mb))
        for i ∈ indices(Taylor(1)), k ∈ indices(Chebyshev(1))
            expected_mb[(i, k)] = mbT[i] * mbC[k]
        end
        @test mb == expected_mb
    end

    @testset "SymmetricSpace" begin
        @testset "evensym / oddsym parity rules on Taylor" begin
            # even * even = even ; odd * odd = even ; even * odd = odd
            a_full = Sequence(Taylor(2), [1.0, 0.0, 3.0]) # 1 + 3x²  (even)
            b_full = Sequence(Taylor(2), [2.0, 0.0, 4.0]) # 2 + 4x²  (even)
            o1_full = Sequence(Taylor(2), [0.0, 2.0, 0.0]) # 2x  (odd)
            o2_full = Sequence(Taylor(2), [0.0, 3.0, 0.0]) # 3x  (odd)

            a_sym = Sequence(evensym(Taylor(2)), [1.0, 3.0])
            b_sym = Sequence(evensym(Taylor(2)), [2.0, 4.0])
            o1_sym = Sequence(oddsym(Taylor(2)), [2.0])
            o2_sym = Sequence(oddsym(Taylor(2)), [3.0])

            ee_full, ee_sym = a_full * b_full, a_sym * b_sym
            @test desymmetrize(space(ee_sym)) == Taylor(4)
            @test collect(indices(space(ee_sym))) == [0, 2, 4] # even * even stays even
            for k ∈ indices(space(ee_sym))
                @test ee_sym[k] == ee_full[k]
            end

            oo_full, oo_sym = o1_full * o2_full, o1_sym * o2_sym
            @test collect(indices(space(oo_sym))) == [0, 2, 4] # odd * odd becomes even
            for k ∈ indices(space(oo_sym))
                @test oo_sym[k] == oo_full[k]
            end

            eo_full, eo_sym = a_full * o1_full, a_sym * o1_sym
            @test collect(indices(space(eo_sym))) == [1, 3] # even * odd stays odd
            for k ∈ indices(space(eo_sym))
                @test eo_sym[k] == eo_full[k]
            end
        end

        @testset "d4sym on Fourier ⊗ Fourier" begin
            # a manifestly d4-invariant input: the four axis-aligned order-1 modes share one value,
            # so it is fixed by both the 90°-rotation and the swap-reflection generators of d4sym
            sT = Fourier(2, 1.0) ⊗ Fourier(2, 1.0)
            a_full = zeros(sT)
            a_full[(1, 0)] = a_full[(-1, 0)] = a_full[(0, 1)] = a_full[(0, -1)] = 1.0

            sS = d4sym(sT)
            a_sym = zeros(sS)
            for k ∈ indices(sS)
                a_sym[k] = a_full[k]
            end

            p_full, p_sym = a_full * a_full, a_sym * a_sym
            @test desymmetrize(space(p_sym)) == Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
            for k ∈ indices(space(p_sym))
                @test p_sym[k] == p_full[k]
            end
        end

        @testset "products under :fft exercise _maybe_desym / _maybe_sym on SymmetricSpace" begin
            # under :fft a symmetric sequence must first be expanded onto its full space,
            # then the result projected back onto the symmetric codomain
            try
                a_sym = Sequence(evensym(Taylor(2)), [1.0, 3.0]) # 1 + 3x² (even)
                b_sym = Sequence(evensym(Taylor(2)), [2.0, 4.0]) # 2 + 4x² (even)

                set_conv_algorithm(:loop)
                ee_sum = a_sym * b_sym
                set_conv_algorithm(:fft)
                ee_fft = a_sym * b_sym

                @test space(ee_fft) == space(ee_sum)
                @test coefficients(ee_fft) ≈ coefficients(ee_sum) atol=1e-9

                sT = Fourier(2, 1.0) ⊗ Fourier(2, 1.0)
                a_full = zeros(sT)
                a_full[(1, 0)] = a_full[(-1, 0)] = a_full[(0, 1)] = a_full[(0, -1)] = 1.0
                sS = d4sym(sT)
                a_d4 = zeros(sS)
                for k ∈ indices(sS)
                    a_d4[k] = a_full[k]
                end

                set_conv_algorithm(:loop)
                p_sum = a_d4 * a_d4
                set_conv_algorithm(:fft)
                p_fft = a_d4 * a_d4

                @test space(p_fft) == space(p_sum)
                @test coefficients(p_fft) ≈ coefficients(p_sum) atol=1e-9
            finally
                set_conv_algorithm(:loop)
            end
        end
    end

    @testset "Complex{Interval{Float64}} coefficients" begin
        # constant sequences: (1/3+i/4)(1/5+i/6) = (1/15-1/24) + i(1/18+1/20) = 1/40 + i·19/180
        a = Sequence(Taylor(0), [complex(interval(Float64, 1//3), interval(Float64, 1//4))])
        b = Sequence(Taylor(0), [complex(interval(Float64, 1//5), interval(Float64, 1//6))])
        c = a * b
        @test in_interval(1//40, real(c[0]))
        @test in_interval(19//180, imag(c[0]))
    end

    @testset ":fft vs :loop agreement" begin
        try
            a = Sequence(Taylor(3), [1.0, 2.0, 0.0, 3.0])
            b = Sequence(Taylor(3), [2.0, 0.0, 1.0, 1.0])

            set_conv_algorithm(:loop)
            ab_sum, pow_sum, mb_sum, pb_sum = a * b, a^3, mul_bar(a, b), pow_bar(a, 3)
            set_conv_algorithm(:fft)
            ab_fft, pow_fft, mb_fft, pb_fft = a * b, a^3, mul_bar(a, b), pow_bar(a, 3)

            @test coefficients(ab_fft) ≈ coefficients(ab_sum) atol=1e-10
            @test coefficients(pow_fft) ≈ coefficients(pow_sum) atol=1e-10
            @test coefficients(mb_fft) ≈ coefficients(mb_sum) atol=1e-10
            @test coefficients(pb_fft) ≈ coefficients(pb_sum) atol=1e-10
        finally
            set_conv_algorithm(:loop) # restore the default
        end
    end

    @testset "sparse supports (_enforce_zeros! / _pow_enforce_zeros! aliasing cleanup)" begin
        # Regression: for Chebyshev with overlapping supports, TᵢTᵢ contributes to T₀,
        # and the FFT cleanup must not zero those coefficients.
        try
            cheb_a = Sequence(Chebyshev(3), [0.0, 0.0, 1.0, 1.0])
            cheb_b = Sequence(Chebyshev(3), [0.0, 0.0, 0.0, 1.0])
            tay_a  = Sequence(Taylor(3), [0.0, 0.0, 1.0, 1.0])
            tay_b  = Sequence(Taylor(3), [0.0, 0.0, 0.0, 1.0])
            fou_a  = Sequence(Fourier(3, 1.0), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
            fou_b  = Sequence(Fourier(3, 1.0), [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            ten_a  = zeros(Taylor(2) ⊗ Chebyshev(3)); ten_a[(0, 2)] = 1.0; ten_a[(0, 3)] = 1.0
            ten_b  = zeros(Taylor(2) ⊗ Chebyshev(3)); ten_b[(0, 3)] = 1.0

            for (x, y) ∈ ((cheb_a, cheb_b), (tay_a, tay_b), (fou_a, fou_b), (ten_a, ten_b))
                set_conv_algorithm(:loop)
                xy_sum, sq_sum, cb_sum = x * y, x^2, x^3
                set_conv_algorithm(:fft)
                xy_fft, sq_fft, cb_fft = x * y, x^2, x^3
                @test coefficients(xy_fft) ≈ coefficients(xy_sum) atol=1e-12
                @test coefficients(sq_fft) ≈ coefficients(sq_sum) atol=1e-12
                @test coefficients(cb_fft) ≈ coefficients(cb_sum) atol=1e-12
            end
        finally
            set_conv_algorithm(:loop)
        end
    end

    @testset "banach_rounding!: genuine geometric decay triggers the non-trivial branch" begin
        # coefficients with no particular decay fit a rate ≤ 1, for which no rounding
        # order is finite and the tail is never rounded; a[i] = 16⁻ⁱ instead fits a
        # clean geometric rate ≈ 16, so the tail does get rounded
        try
            base = 16.0
            ord = 10
            a = Sequence(Taylor(ord), [inv(base^i) for i ∈ 0:ord])
            b = Sequence(Taylor(ord), [inv(base^i) for i ∈ 0:ord])
            @test RadiiPolynomial.weight(a) isa RadiiPolynomial.GeometricWeight
            @test rate(RadiiPolynomial.weight(a)) > 1

            set_conv_algorithm(:loop)
            ab_sum, pow_sum, mb_sum, pb_sum = a * b, a^3, mul_bar(a, b), pow_bar(a, 3)
            set_conv_algorithm(:fft)
            ab_fft, pow_fft, mb_fft, pb_fft = a * b, a^3, mul_bar(a, b), pow_bar(a, 3)

            @test coefficients(ab_fft) ≈ coefficients(ab_sum) atol=1e-9
            @test coefficients(pow_fft) ≈ coefficients(pow_sum) atol=1e-9
            @test coefficients(mb_fft) ≈ coefficients(mb_sum) atol=1e-9
            @test coefficients(pb_fft) ≈ coefficients(pb_sum) atol=1e-9

            # c[k] = Σⱼ a[k-j]a[j] = (#valid j)·16⁻ᵏ; c[0] has a single term (=1)
            # and is well below the rounding order, so it is untouched
            @test ab_fft[0] == 1.0
            # the fitted rounding order here is 15 (< the full order 20): every c[k] with
            # k ≥ 15 is overwritten by the Banach-algebra tail bound, which for plain
            # Float64 coefficients collapses to exactly 0.0, even though the true value
            # (computed exactly by :loop, 6·16⁻¹⁵ = 3/2⁵⁹) is tiny but nonzero
            @test ab_fft[15] == 0.0
            @test ab_sum[15] ≈ 6 * base^(-15.0)
            @test ab_sum[15] > 0
        finally
            set_conv_algorithm(:loop)
        end
    end

    @testset "banach_rounding!: algebraic decay branch" begin
        # a[i] = (1+i)⁻²⁰ decays algebraically, so the fit picks an algebraic weight
        # over a geometric one and the rounding order is derived from that weight
        try
            p = 20.0
            ord = 10
            a = Sequence(Taylor(ord), [inv((1.0 + i)^p) for i ∈ 0:ord])
            b = Sequence(Taylor(ord), [inv((1.0 + i)^p) for i ∈ 0:ord])
            @test RadiiPolynomial.weight(a) isa RadiiPolynomial.AlgebraicWeight

            set_conv_algorithm(:loop)
            ab_sum = a * b
            set_conv_algorithm(:fft)
            ab_fft = a * b

            @test coefficients(ab_fft) ≈ coefficients(ab_sum) atol=1e-9
            # rounding_order = 7 here (< the full order 20): c[7] is snapped to 0.0
            # although the true value (a sum of strictly positive terms) is > 0
            @test ab_fft[7] == 0.0
            @test ab_sum[7] > 0
        finally
            set_conv_algorithm(:loop)
        end
    end

    @testset "banach_rounding!: Fourier tail write via _write_symmetric!" begin
        # on Fourier the rounded tail is written symmetrically, c[i] and c[-i] at
        # once; a[i] = 16⁻|ⁱ| makes both ±16 fall in that tail
        try
            base = 16.0
            ord = 10
            af = Sequence(Fourier(ord, 1.0), [inv(base^abs(i)) for i ∈ -ord:ord])
            bf = Sequence(Fourier(ord, 1.0), [inv(base^abs(i)) for i ∈ -ord:ord])

            set_conv_algorithm(:loop)
            ab_sum = af * bf
            set_conv_algorithm(:fft)
            ab_fft = af * bf

            @test coefficients(ab_fft) ≈ coefficients(ab_sum) atol=1e-9
            @test ab_fft[16] == 0.0 == ab_fft[-16]
            @test ab_sum[16] > 0
            @test ab_sum[-16] > 0
        finally
            set_conv_algorithm(:loop)
        end
    end

    @testset "banach_rounding!: TensorSpace NTuple rounding_order branch" begin
        # a[(i,k)] = 16⁻ⁱ·16⁻|ᵏ| on Taylor(6)⊗Fourier(6) fits a rounding order of
        # (17,17); the corner (12,12) of the full Taylor(12)⊗Fourier(12) codomain
        # satisfies 12/17 + 12/17 ≥ 1, so it belongs to the rounded tail
        try
            base = 16.0
            ordT = 6
            ordF = 6
            s = Taylor(ordT) ⊗ Fourier(ordF, 1.0)
            a = zeros(s)
            b = zeros(s)
            for i ∈ indices(Taylor(ordT)), k ∈ indices(Fourier(ordF, 1.0))
                a[(i, k)] = b[(i, k)] = inv(base^i) * inv(base^abs(k))
            end

            set_conv_algorithm(:loop)
            ab_sum = a * b
            set_conv_algorithm(:fft)
            ab_fft = a * b

            @test coefficients(ab_fft) ≈ coefficients(ab_sum) atol=1e-9
            @test ab_fft[(12, 12)] == 0.0
            @test ab_sum[(12, 12)] > 0
            # a corner well inside the rounding order is untouched by rounding
            @test ab_fft[(0, 0)] ≈ ab_sum[(0, 0)] atol=1e-9
        finally
            set_conv_algorithm(:loop)
        end
    end

    @testset "banach_rounding!: rigorous Interval / Complex{Interval} tail enclosure" begin
        # with interval (resp. complex interval) coefficients the tail bound is itself
        # an interval, and the rounded tail holds a genuine rigorous enclosure of the
        # true coefficients rather than 0.0
        base = 16.0
        base_big = big(16) # exact integer base, used for the independent Rational ground truth
        ord = 10
        # exact ground truth: a[i] = 16⁻ⁱ = 2⁻⁴ⁱ is exactly representable, so the true
        # convolution coefficients are exact dyadic rationals we can compute independently
        aQ = [Rational{BigInt}(1, base_big^i) for i ∈ 0:ord]
        conv_taylor(k) = sum(aQ[k-j+1] * aQ[j+1] for j ∈ max(k - ord, 0):min(k, ord))

        try
            a = Sequence(Taylor(ord), [interval(inv(base^i)) for i ∈ 0:ord])
            b = Sequence(Taylor(ord), [interval(inv(base^i)) for i ∈ 0:ord])

            set_conv_algorithm(:fft)
            ab_fft = a * b
            set_conv_algorithm(:loop)
            ab_sum = a * b

            @test in_interval(1, ab_fft[0]) # c[0] = 1, well below the rounding order
            exact15 = conv_taylor(15)
            @test in_interval(exact15, ab_fft[15]) # rigorous enclosure of the rounded tail
            @test in_interval(exact15, ab_sum[15])
        finally
            set_conv_algorithm(:loop)
        end

        aQf = Dict(i => Rational{BigInt}(1, base_big^abs(i)) for i ∈ -ord:ord)
        conv_fourier(k) = sum(aQf[k-j] * aQf[j] for j ∈ max(k - ord, -ord):min(k + ord, ord))

        try
            acf = Sequence(Fourier(ord, 1.0), [complex(interval(inv(base^abs(i))), interval(0.0)) for i ∈ -ord:ord])
            bcf = Sequence(Fourier(ord, 1.0), [complex(interval(inv(base^abs(i))), interval(0.0)) for i ∈ -ord:ord])

            set_conv_algorithm(:fft)
            abcf_fft = acf * bcf
            set_conv_algorithm(:loop)
            abcf_sum = acf * bcf

            # note: unlike the one-sided Taylor case above, a[k-j]a[j] = 16⁻|k-j|16⁻|j|
            # is *not* independent of j here (Fourier decays in |i|, not i), so c[0] is
            # not simply 1; conv_fourier computes the exact value directly instead
            exact0 = conv_fourier(0)
            @test in_interval(exact0, real(abcf_fft[0])) # well below the rounding order
            @test in_interval(0, imag(abcf_fft[0]))
            exact16 = conv_fourier(16)
            @test in_interval(exact16, real(abcf_fft[16]))
            @test in_interval(exact16, real(abcf_sum[16]))
        finally
            set_conv_algorithm(:loop)
        end
    end

    @testset "InfiniteSequence: power-by-squaring (^) for n ≥ 3" begin
        # a = 1 + 2x (Taylor(1)) with finite_error=0, tail_error=total_error=0.5 in Ell1();
        # all coefficients/errors are exact binary fractions, so every association of the
        # underlying `*` product below is bit-for-bit reproducible.
        a = InfiniteSequence(Sequence(Taylor(1), [1.0, 2.0]), 0.0, 0.5, 0.5, Ell1())

        @testset "n = 3: exercises the first (no-op) while, then the outer/inner while once" begin
            r3 = a^3
            # (1+2x)³ = 1+6x+12x²+8x³ truncated to Taylor(1) by InfiniteSequence's `*`
            @test sequence(r3) == Sequence(Taylor(1), [1.0, 6.0])

            # a^3 is computed as a*(a*a), and for n = 3 the error bound is symmetric in its
            # two operands, so this reference built from `*` alone matches r3 in every field
            ref = a * (a * a)
            @test sequence(r3) == sequence(ref)
            @test finite_error(r3) == finite_error(ref)
            @test tail_error(r3) == tail_error(ref)
            @test total_error(r3) == total_error(ref)
            @test sequence_norm(r3) == sequence_norm(ref)
            @test r3.full_norm == ref.full_norm
        end

        @testset "n = 6: exercises the first while body and the outer/inner while loops" begin
            r6 = a^6
            # (1+2x)⁶ = ... + 12x + ...; coefficient of x¹ is C(6,1)·2¹ = 12
            @test sequence(r6) == Sequence(Taylor(1), [1.0, 12.0])

            # a^6 is computed as a² * (a²)² = a² * a⁴; this reference, built from `*` alone,
            # matches r6 exactly in every field
            a2 = a * a
            a4 = a2 * a2
            ref = a2 * a4
            @test sequence(r6) == sequence(ref)
            @test finite_error(r6) == finite_error(ref)
            @test tail_error(r6) == tail_error(ref)
            @test total_error(r6) == total_error(ref)
            @test sequence_norm(r6) == sequence_norm(ref)
            @test r6.full_norm == ref.full_norm
        end
    end
end
