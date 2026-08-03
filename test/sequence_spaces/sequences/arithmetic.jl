@testset "Sequence arithmetic" begin
    @testset "codomain" begin
        @testset "ScalarSpace" begin
            @test codomain(+, ScalarSpace(), ScalarSpace()) == ScalarSpace()
        end

        @testset "Taylor, Fourier, Chebyshev" begin
            # same-type spaces: codomain is the union, i.e. the bigger order
            @test codomain(+, Taylor(1), Taylor(3)) == Taylor(3)
            @test codomain(+, Taylor(3), Taylor(1)) == Taylor(3)
            @test codomain(+, Fourier(3, 1.0), Fourier(1, 1.0)) == Fourier(3, 1.0)
            @test codomain(+, Chebyshev(0), Chebyshev(2)) == Chebyshev(2)
            # `-` falls back to the same codomain as `+`
            @test codomain(-, Taylor(1), Taylor(3)) == codomain(+, Taylor(1), Taylor(3))
            # Fourier requires equal frequencies to combine
            @test_throws ArgumentError codomain(+, Fourier(1, 1.0), Fourier(1, 2.0))
        end

        @testset "TensorSpace" begin
            @test codomain(+, Taylor(1) ⊗ Fourier(0, 1.0), Taylor(2) ⊗ Fourier(1, 1.0)) == Taylor(2) ⊗ Fourier(1, 1.0)
            @test_throws ArgumentError codomain(+, Taylor(1) ⊗ Fourier(1, 1.0), Taylor(1) ⊗ Fourier(1, 2.0))
        end

        @testset "SymmetricSpace" begin
            # same symmetry group on both sides (both built by `evensym`/`oddsym`/`d4sym`):
            # codomain is the union of the underlying spaces, group unchanged
            @test codomain(+, evensym(Taylor(2)), evensym(Taylor(3))) == evensym(Taylor(3))
            @test codomain(+, oddsym(Taylor(2)), oddsym(Taylor(3))) == oddsym(Taylor(3))
            d4a = d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))
            d4b = d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))
            @test codomain(+, d4a, d4b) == d4b

            # mixing a SymmetricSpace with its own plain (non-symmetric) base space: intersecting
            # the evensym group with the trivial group collapses to the trivial group, so the
            # codomain is the trivial-symmetry wrapping of the underlying union, i.e. `Taylor(2)`
            # with no restriction
            r1 = codomain(+, evensym(Taylor(2)), Taylor(2))
            r2 = codomain(+, Taylor(2), evensym(Taylor(2)))
            @test r1 isa SymmetricSpace
            @test r1 == r2 == SymmetricSpace(Taylor(2))
            @test indices(r1) == 0:2

            # end-to-end: adding a genuinely restricted evensym sequence into a plain target
            # recovers the plain sum (odd-order coefficient of `a` is implicitly zero)
            a = Sequence(evensym(Taylor(2)), [1.0, 2.0]) # a₀=1, a₂=2 (a₁=0)
            b = Sequence(Taylor(2), [10.0, 20.0, 30.0])
            @test add!(zeros(Taylor(2)), a, b) == add!(zeros(Taylor(2)), b, a) == Sequence(Taylor(2), [11.0, 20.0, 32.0])
        end

        @testset "CartesianPower" begin
            @test codomain(+, Taylor(1)^2, Taylor(2)^2) == Taylor(2)^2
            @test_throws ArgumentError codomain(+, Taylor(1)^2, Taylor(1)^3)
        end

        @testset "CartesianProduct" begin
            @test codomain(+, Taylor(1) × Fourier(1, 1.0), Taylor(2) × Fourier(0, 1.0)) == Taylor(2) × Fourier(1, 1.0)
            @test_throws ArgumentError codomain(+, Taylor(1) × Fourier(1, 1.0), (Taylor(1) × Fourier(1, 1.0)) × Chebyshev(1))
        end

        @testset "CartesianPower / CartesianProduct mix" begin
            @test codomain(+, Taylor(1)^2, Taylor(1) × Taylor(2)) == Taylor(1) × Taylor(2)
            @test codomain(+, Taylor(1) × Taylor(2), Taylor(1)^2) == Taylor(1) × Taylor(2)
        end
    end

    @testset "unary + and -" begin
        a = Sequence(Taylor(2), [1.0, -2.0, 3.0])
        @test +a == a
        @test -a == Sequence(Taylor(2), [-1.0, 2.0, -3.0])
    end

    @testset "scalar */÷ (rmul!, lmul!, rdiv!, ldiv!)" begin
        𝒯 = Taylor(2)
        a = Sequence(𝒯, [1.0, 2.0, 3.0])
        expected_mul = Sequence(𝒯, [3.0, 6.0, 9.0])
        @test a * 3.0 == 3.0 * a == rmul!(copy(a), 3.0) == lmul!(3.0, copy(a)) == expected_mul

        expected_div = Sequence(𝒯, [0.5, 1.0, 1.5])
        @test a / 2.0 == 2.0 \ a == rdiv!(copy(a), 2.0) == ldiv!(2.0, copy(a)) == expected_div

        # Fourier with Interval coefficients
        af = Sequence(Fourier(1, 1.0), [interval(1.0), interval(2.0), interval(3.0)])
        r = 2.0 * af
        @test all(isequal_interval.(coefficients(r), interval.([2.0, 4.0, 6.0])))
        @test all(isequal_interval.(coefficients(rmul!(copy(af), 2.0)), coefficients(r)))

        # Chebyshev with ComplexF64 coefficients
        ac = Sequence(Chebyshev(1), ComplexF64[1.0+1.0im, 2.0-1.0im])
        @test 2.0 * ac == ac * 2.0 == rmul!(copy(ac), 2.0) == lmul!(2.0, copy(ac)) ==
            Sequence(Chebyshev(1), ComplexF64[2.0+2.0im, 4.0-2.0im])

        # TensorSpace
        att = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(1.0:6.0))
        @test att * 2.0 == Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(2.0:2.0:12.0))

        # CartesianPower
        acart = Sequence(Taylor(1)^2, [1.0, 2.0, 3.0, 4.0])
        @test acart * 2.0 == rmul!(copy(acart), 2.0) == Sequence(Taylor(1)^2, [2.0, 4.0, 6.0, 8.0])
    end

    @testset "add!, sub! and in-place variants" begin
        @testset "Taylor: automatic order enlargement" begin
            a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
            b = Sequence(Taylor(3), [10.0, 20.0, 30.0, 40.0])
            expected_add = Sequence(Taylor(3), [11.0, 22.0, 33.0, 40.0]) # a is zero-padded past order 2
            expected_sub = Sequence(Taylor(3), [-9.0, -18.0, -27.0, -40.0])

            out = Sequence(Taylor(3), fill(Inf, 4))
            @test a + b == add!(out, a, b) == expected_add
            @test a - (-b) == expected_add
            out2 = Sequence(Taylor(3), fill(Inf, 4))
            @test a - b == sub!(out2, a, b) == expected_sub

            # radd!/rsub! write into a's own (smaller) space: result is truncated to order 2
            @test radd!(copy(a), b) == Sequence(Taylor(2), [11.0, 22.0, 33.0])
            @test rsub!(copy(a), b) == Sequence(Taylor(2), [-9.0, -18.0, -27.0])
            # ladd!/lsub! write into b's own (bigger) space: result matches the full sum/difference
            @test ladd!(a, copy(b)) == expected_add
            @test lsub!(a, copy(b)) == Sequence(Taylor(3), [-9.0, -18.0, -27.0, -40.0])

            # add! accepts a target order smaller than the union: it truncates instead of erroring
            small = Sequence(Taylor(1), fill(Inf, 2))
            @test add!(small, a, b) == Sequence(Taylor(1), [11.0, 22.0])

            # sub! with `a` bigger than `b`: c's space matches a's space directly (not b's),
            # exercising the `space_a == space_c` branch of `_sub!` (distinct from the
            # `space_b == space_c` branch exercised by `sub!(out2, a, b)` above)
            a_big = Sequence(Taylor(3), [1.0, 2.0, 3.0, 4.0])
            b_small = Sequence(Taylor(1), [10.0, 20.0])
            expected_sub_bigsmall = Sequence(Taylor(3), [-9.0, -18.0, 3.0, 4.0]) # b is zero-padded past order 1
            out3 = Sequence(Taylor(3), fill(Inf, 4))
            @test a_big - b_small == sub!(out3, a_big, b_small) == expected_sub_bigsmall
        end

        @testset "rsub!/lsub! fast path when a and b already share the same space" begin
            # exercises the `space_a == space_b` branch of `_rsub!`/`_lsub!` (the mismatched-space
            # branch is already exercised above via order-mismatched Taylor sequences)
            a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
            b = Sequence(Taylor(2), [10.0, 20.0, 30.0])
            @test rsub!(copy(a), b) == Sequence(Taylor(2), [-9.0, -18.0, -27.0]) # a .-= b, in place
            @test lsub!(copy(a), copy(b)) == Sequence(Taylor(2), [-9.0, -18.0, -27.0]) # b .= a .- b, returned
        end

        @testset "TensorSpace: order bigger in one factor, smaller in the other" begin
            # a is bigger in the Taylor factor but smaller in the Fourier factor (and vice
            # versa for b), so the codomain differs from *both* a's and b's own space; this
            # exercises the fully-general (`else`) branch of `_sub!` where none of a, b, c
            # share a space
            a = Sequence(Taylor(2) ⊗ Fourier(0, 1.0), [1.0, 2.0, 3.0]) # a[(i,0)] = i+1
            b = Sequence(Taylor(0) ⊗ Fourier(1, 1.0), [10.0, 20.0, 30.0]) # b[(0,j)]: j=-1,0,1 -> 10,20,30
            sc = Taylor(2) ⊗ Fourier(1, 1.0)
            # elementwise a[(i,j)] - b[(i,j)], zero-padding whichever side lacks that index
            expected = Sequence(sc, [-10.0, 0.0, 0.0, -19.0, 2.0, 3.0, -30.0, 0.0, 0.0])
            out = Sequence(sc, fill(Inf, 9))
            @test a - b == sub!(out, a, b) == expected
            # spot checks via direct indexing
            @test expected[(0, 0)] == a[(0, 0)] - b[(0, 0)] # 1 - 20 = -19
            @test expected[(1, 0)] == a[(1, 0)] # b has no i=1 mode: 2 - 0 = 2
            @test expected[(0, 1)] == -b[(0, 1)] # a has no j=1 mode: 0 - 30 = -30
        end

        @testset "Fourier: same frequency required" begin
            a = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0]) # a[-1]=1, a[0]=2, a[1]=3
            b = Sequence(Fourier(2, 1.0), [10.0, 20.0, 30.0, 40.0, 50.0]) # b[-2..2]
            expected = Sequence(Fourier(2, 1.0), [10.0, 21.0, 32.0, 43.0, 50.0])
            out = Sequence(Fourier(2, 1.0), fill(Inf, 5))
            @test a + b == add!(out, a, b) == expected

            c = Sequence(Fourier(1, 2.0), [1.0, 2.0, 3.0]) # different frequency
            @test_throws ArgumentError a + c
            @test_throws ArgumentError radd!(copy(a), c)
            @test_throws ArgumentError ladd!(a, copy(c))
            @test_throws ArgumentError rsub!(copy(a), c)
            @test_throws ArgumentError lsub!(a, copy(c))
            # add!/sub! with an incompatible target space also throws
            wrong_target = zeros(Fourier(1, 2.0))
            @test_throws ArgumentError add!(wrong_target, a, b)
            @test_throws ArgumentError sub!(wrong_target, a, b)
        end

        @testset "Chebyshev" begin
            a = Sequence(Chebyshev(1), [1.0, 2.0])
            b = Sequence(Chebyshev(2), [10.0, 20.0, 30.0])
            expected = Sequence(Chebyshev(2), [11.0, 22.0, 30.0])
            out = Sequence(Chebyshev(2), fill(Inf, 3))
            @test a + b == add!(out, a, b) == expected
        end

        @testset "TensorSpace" begin
            a = Sequence(Taylor(1) ⊗ Fourier(0, 1.0), [1.0, 2.0])
            b = Sequence(Taylor(2) ⊗ Fourier(1, 1.0), collect(1.0:9.0))
            r = a + b
            out = Sequence(space(r), fill(Inf, length(r)))
            @test r == add!(out, a, b) == Sequence(Taylor(2) ⊗ Fourier(1, 1.0), [1.0, 2.0, 3.0, 5.0, 7.0, 6.0, 7.0, 8.0, 9.0])
            # spot check via direct indexing: a only has a k=0 Fourier mode, so it only adds
            # into b's k=0 slice; elsewhere r matches b exactly
            @test r[(0, 0)] == a[(0, 0)] + b[(0, 0)]
            @test r[(0, -1)] == b[(0, -1)] # a has no k=-1 Fourier mode
        end
    end

    @testset "cartesian sequence arithmetic" begin
        @testset "CartesianPower: matching inner space" begin
            s = Taylor(1)^2
            a = Sequence(s, [1.0, 2.0, 3.0, 4.0])
            b = Sequence(s, [10.0, 20.0, 30.0, 40.0])
            expected = Sequence(s, [11.0, 22.0, 33.0, 44.0])
            out = Sequence(s, fill(Inf, 4))
            @test a + b == add!(out, a, b) == radd!(copy(a), b) == ladd!(a, copy(b)) == expected
        end

        @testset "CartesianPower: mismatched inner space (auto enlarge)" begin
            a = Sequence(Taylor(1)^2, [1.0, 2.0, 3.0, 4.0])
            b = Sequence(Taylor(2)^2, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            expected = Sequence(Taylor(2)^2, [11.0, 22.0, 30.0, 43.0, 54.0, 60.0])
            out = Sequence(Taylor(2)^2, fill(Inf, 6))
            @test a + b == add!(out, a, b) == expected
            # radd! truncates the result back to a's own (smaller) order
            @test radd!(copy(a), b) == Sequence(Taylor(1)^2, [11.0, 22.0, 43.0, 54.0])
            # ladd!/lsub! write into b's own (bigger) space and recurse component-by-component
            # (mismatched CartesianPower spaces), exercising the `else` branch of the
            # CartesianSpace-specific `_lf!` (ladd!/lsub!)
            @test ladd!(a, copy(b)) == expected
            @test lsub!(a, copy(b)) == Sequence(Taylor(2)^2, [-9.0, -18.0, -30.0, -37.0, -46.0, -60.0])
        end

        @testset "CartesianProduct: matching inner spaces (fast path)" begin
            # 2-component CartesianProduct with a and b already sharing the same space:
            # exercises the `space(a) == space(b)` fast path of the CartesianProduct-specific
            # `_add!`/`_sub!` (the mismatched-space branch is already covered above)
            sp = Taylor(1) × Fourier(1, 1.0)
            a = Sequence(sp, [1.0, 2.0, 3.0, 4.0, 5.0])
            b = Sequence(sp, [10.0, 20.0, 30.0, 40.0, 50.0])
            expected_add = Sequence(sp, [11.0, 22.0, 33.0, 44.0, 55.0])
            expected_sub = Sequence(sp, [-9.0, -18.0, -27.0, -36.0, -45.0])
            out = Sequence(sp, fill(Inf, 5))
            out2 = Sequence(sp, fill(Inf, 5))
            @test a + b == add!(out, a, b) == expected_add
            @test a - b == sub!(out2, a, b) == expected_sub
        end

        @testset "CartesianProduct: 3 components, mismatched inner spaces" begin
            sp1 = Taylor(1) × Fourier(0, 1.0) × Chebyshev(1)
            sp2 = Taylor(2) × Fourier(1, 1.0) × Chebyshev(1)
            a = Sequence(sp1, collect(1.0:5.0))
            b = Sequence(sp2, collect(1.0:8.0))
            expected = Sequence(sp2, [2.0, 4.0, 3.0, 4.0, 8.0, 6.0, 11.0, 13.0])
            out = Sequence(sp2, fill(Inf, 8))
            @test a + b == add!(out, a, b) == expected
            @test radd!(copy(a), b) == Sequence(sp1, [2.0, 4.0, 8.0, 11.0, 13.0])
        end

        @testset "single-element CartesianProduct" begin
            a = Sequence(CartesianProduct(Taylor(1)), [1.0, 2.0])
            b = Sequence(CartesianProduct(Taylor(2)), [10.0, 20.0, 30.0])
            @test a + b == Sequence(CartesianProduct(Taylor(2)), [11.0, 22.0, 30.0])
        end

        @testset "CartesianPower vs CartesianProduct mismatch throws" begin
            @test_throws ArgumentError Sequence(Taylor(1)^2, zeros(4)) + Sequence(Taylor(1)^3, zeros(6))
            @test_throws ArgumentError radd!(Sequence(Taylor(1)^2, zeros(4)), Sequence(Taylor(1)^3, zeros(6)))
        end
    end

    @testset "ScalarSpace sequence arithmetic (*, /, \\\\, ^, +, -)" begin
        s1 = Sequence(ScalarSpace(), [2.0])
        s2 = Sequence(ScalarSpace(), [3.0])
        @test s1 * s2 == Sequence(ScalarSpace(), [6.0])
        @test s1 ^ 2 == Sequence(ScalarSpace(), [4.0])
        @test s1 ^ s2 == Sequence(ScalarSpace(), [8.0]) # 2³
        @test s1 / s2 == Sequence(ScalarSpace(), [2.0/3.0])
        @test s1 \ s2 == Sequence(ScalarSpace(), [1.5]) # inv(2)*3

        @test s1 + 1.0 == 1.0 + s1 == Sequence(ScalarSpace(), [3.0])
        @test s1 - 1.0 == Sequence(ScalarSpace(), [1.0])
        @test 1.0 - s1 == Sequence(ScalarSpace(), [-1.0])
        @test radd!(copy(s1), 1.0) == ladd!(1.0, copy(s1)) == s1 + 1.0
        @test rsub!(copy(s1), 1.0) == s1 - 1.0
        @test lsub!(1.0, copy(s1)) == 1.0 - s1

        # mixing a ScalarSpace sequence with a SequenceSpace sequence: the scalar is treated as
        # the constant (order-0) coefficient
        ta = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        @test s1 + ta == ta + s1 == Sequence(Taylor(2), [3.0, 2.0, 3.0])
    end

    @testset "adding/subtracting a number to a sequence" begin
        @testset "Taylor, Fourier, Chebyshev, TensorSpace" begin
            a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
            @test a + 5.0 == 5.0 + a == Sequence(Taylor(2), [6.0, 2.0, 3.0])
            @test a - 5.0 == Sequence(Taylor(2), [-4.0, 2.0, 3.0])
            @test 5.0 - a == Sequence(Taylor(2), [4.0, -2.0, -3.0])
            @test radd!(copy(a), 5.0) == ladd!(5.0, copy(a)) == a + 5.0
            @test rsub!(copy(a), 5.0) == a - 5.0
            @test lsub!(5.0, copy(a)) == 5.0 - a

            af = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0]) # constant mode is the middle entry
            @test af + 5.0 == Sequence(Fourier(1, 1.0), [1.0, 7.0, 3.0])

            ac = Sequence(Chebyshev(1), [1.0, 2.0])
            @test ac + 5.0 == Sequence(Chebyshev(1), [6.0, 2.0])

            at = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(1.0:6.0))
            @test at + 5.0 == Sequence(Taylor(1) ⊗ Fourier(1, 1.0), [1.0, 2.0, 8.0, 4.0, 5.0, 6.0])
        end

        @testset "SymmetricSpace" begin
            # evensym(Taylor(2)): only even powers (indices 0, 2) are free; adding a constant
            # simply shifts the k=0 coefficient, the space stays evensym(Taylor(2))
            even = Sequence(evensym(Taylor(2)), [1.0, 2.0]) # c₀=1, c₂=2
            r = even + 5.0
            @test space(r) == evensym(Taylor(2))
            @test r == Sequence(evensym(Taylor(2)), [6.0, 2.0])

            # oddsym(Taylor(2)): only the odd power (index 1) is free; adding a constant breaks
            # the odd symmetry, so the result widens to the unrestricted space (all of 0:2)
            odd = Sequence(oddsym(Taylor(2)), [3.0]) # c₁=3
            r2 = odd + 5.0
            @test dimension(space(r2)) == 3
            @test r2 == Sequence(space(r2), [5.0, 3.0, 0.0])

            # in-place `radd!` cannot widen the space: adding a constant to an odd-symmetric
            # sequence in place is ill-defined since k=0 is not one of its own indices.
            @test_throws ArgumentError radd!(copy(odd), 5.0)
        end
    end

    @testset "cartesian sequence ± vector of numbers" begin
        s = Taylor(1)^3
        a = Sequence(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) # 3 components: [1,2], [3,4], [5,6]
        v = [10.0, 20.0, 30.0]
        expected_add = Sequence(s, [11.0, 2.0, 23.0, 4.0, 35.0, 6.0])
        expected_sub = Sequence(s, [-9.0, 2.0, -17.0, 4.0, -25.0, 6.0])
        expected_rsub = Sequence(s, [9.0, -2.0, 17.0, -4.0, 25.0, -6.0]) # v - a

        @test a + v == v + a == radd!(copy(a), v) == ladd!(v, copy(a)) == expected_add
        @test a - v == rsub!(copy(a), v) == expected_sub
        @test v - a == lsub!(v, copy(a)) == expected_rsub

        # nested CartesianProduct containing a CartesianPower: the vector has one entry per
        # deepest (leaf) component
        s2 = (Taylor(1)^2) × Fourier(1, 1.0)
        b = Sequence(s2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        r = b + [100.0, 200.0, 300.0]
        @test r == Sequence(s2, [101.0, 2.0, 203.0, 4.0, 5.0, 306.0, 7.0])

        # `-` with the vector on the right recurses through `_rsub!`; since s2's first
        # component is itself a CartesianSpace (CartesianPower), this exercises the
        # nested-CartesianSpace branch of `_rsub!` (mirroring `_radd!` above but for `-`)
        rs = b - [100.0, 200.0, 300.0]
        @test rs == Sequence(s2, [-99.0, 2.0, -197.0, 4.0, 5.0, -294.0, 7.0])

        @test_throws ArgumentError a + [10.0, 20.0]
        @test_throws ArgumentError radd!(copy(a), [10.0, 20.0])
    end

    @testset "Interval and Complex coefficient promotion" begin
        a = Sequence(Taylor(1), [interval(1.0), interval(2.0)])
        b = Sequence(Taylor(1), [3.0, 4.0])
        c = a + b
        @test eltype(c) == Interval{Float64}
        @test isequal_interval(c[0], interval(4.0)) && isequal_interval(c[1], interval(6.0))
        out = Sequence(Taylor(1), [interval(-9999.0), interval(-9999.0)]) # sentinel to catch missed writes
        add!(out, a, b)
        @test isequal_interval(out[0], interval(4.0)) && isequal_interval(out[1], interval(6.0))

        ac = Sequence(Taylor(1), ComplexF64[1.0+2.0im, 3.0+4.0im])
        bc = Sequence(Taylor(1), [1.0, 1.0])
        cc = ac + bc
        @test eltype(cc) == ComplexF64
        @test cc == Sequence(Taylor(1), ComplexF64[2.0+2.0im, 4.0+4.0im])

        # Complex{Interval{Float64}} promotion
        aci = Sequence(Taylor(1), Complex{Interval{Float64}}[interval(1.0)+interval(1.0)*im, interval(2.0)])
        bci = Sequence(Taylor(1), [1.0, 1.0])
        cci = aci + bci
        @test eltype(cci) == Complex{Interval{Float64}}
        @test isequal_interval(real(cci[0]), interval(2.0)) && isequal_interval(imag(cci[0]), interval(1.0))
        @test isequal_interval(real(cci[1]), interval(3.0)) && isequal_interval(imag(cci[1]), interval(0.0))
    end
end
