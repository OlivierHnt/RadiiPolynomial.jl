@testset "FFT" begin
    # small local helper: minimal AbstractVector with non-1-based axes, used
    # only to exercise the "offset arrays are not supported" guard clauses.
    struct _OffsetVec{T} <: AbstractVector{T}
        data :: Vector{T}
    end
    Base.size(v::_OffsetVec) = size(v.data)
    Base.axes(v::_OffsetVec) = (2:length(v.data)+1,)
    Base.getindex(v::_OffsetVec, i::Int) = v.data[i-1]
    Base.setindex!(v::_OffsetVec, x, i::Int) = (v.data[i-1] = x)

    @testset "fft_size" begin
        # the smallest transform representing the space faithfully, no rounding up
        @testset "Taylor: order+1" begin
            @test fft_size(Taylor(0)) == (1,)
            @test fft_size(Taylor(1)) == (2,)
            @test fft_size(Taylor(4)) == (5,)
            @test fft_size(Taylor(5)) == (6,) # not a power of 2
        end

        @testset "Fourier: 2order+1, independent of frequency" begin
            @test fft_size(Fourier(0, 1.0)) == (1,)
            @test fft_size(Fourier(1, 1.0)) == (3,)
            @test fft_size(Fourier(3, 1.0)) == (7,)
            @test fft_size(Fourier(2, 3.5)) == fft_size(Fourier(2, 1.0)) # frequency does not affect the grid size
        end

        @testset "Chebyshev: 2order, the coefficients being mirrored" begin
            @test fft_size(Chebyshev(0)) == (1,)
            @test fft_size(Chebyshev(1)) == (2,)
            @test fft_size(Chebyshev(4)) == (8,)
            @test fft_size(Chebyshev(5)) == (10,) # not a power of 2
        end

        @testset "TensorSpace: componentwise, one entry per factor" begin
            @test fft_size(Taylor(2) ⊗ Fourier(1, 1.0)) == (3, 3)
            @test fft_size(Taylor(1) ⊗ Taylor(1) ⊗ Chebyshev(3)) == (2, 2, 6)
        end

        @testset "SymmetricSpace: defers to the desymmetrized space" begin
            @test fft_size(evensym(Taylor(3))) == fft_size(Taylor(3)) == (4,)
            @test fft_size(oddsym(Fourier(3, 1.0))) == fft_size(Fourier(3, 1.0)) == (7,)
            @test fft_size(evensym(Chebyshev(3))) == fft_size(Chebyshev(3)) == (6,)
            @test fft_size(d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))) == fft_size(Fourier(2, 1.0) ⊗ Fourier(2, 1.0)) == (5, 5)
        end
    end

    #

    @testset "Taylor" begin
        @testset "exact hand-computed round trip (order 1, N=2 uses only the ±1 roots)" begin
            # a(z) = 1 + 2z fed to the backward (unnormalized inverse) DFT
            # Y[j] = Σ_k C[k]·e^{i2πkj/2}:
            #   Y0 = 1+2 = 3
            #   Y1 = 1-2 = -1
            a = Sequence(Taylor(1), [1.0, 2.0])
            expected_grid = ComplexF64[3, -1]
            @test fft_size(space(a)) == (2,)
            @test to_grid(a) == expected_grid
            @test to_grid!(fill(complex(Inf, Inf), 2), a) == expected_grid
            @test to_coef(expected_grid, Taylor(1)) == Sequence(Taylor(1), ComplexF64[1, 2])
            seeded = Sequence(Taylor(1), ComplexF64[Inf, Inf])
            @test to_coef!(seeded, copy(expected_grid)) == Sequence(Taylor(1), ComplexF64[1, 2]) == seeded
            # the four-point grid of the previous convention is still accepted
            @test to_grid(a, 4) == ComplexF64[3, 1+2im, -1, 1-2im]
        end

        @testset "an oversized destination grid is zero-padded" begin
            a = Sequence(Taylor(1), [1.0, 2.0])
            C = zeros(ComplexF64, 4)
            @test to_grid!(C, a) === C
        end

        @testset "approximate round trip, higher order (N=4: irrational roots, roundoff genuine)" begin
            a = Sequence(Taylor(3), [1.0, -2.0, 0.5, 3.0])
            b = to_coef(to_grid(a), Taylor(3))
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
            @test all(x -> isapprox(x, 0.0; atol=1e-9), imag.(coefficients(b)))
        end

        @testset "round trip at every order, including sizes with a large prime factor" begin
            for n ∈ [0:20; 36; 66; 100] # 37, 67 and 101 are prime
                a = Sequence(Taylor(n), collect(1.0:n+1))
                b = to_coef(to_grid(a), Taylor(n))
                @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-10
            end
        end

        @testset "order-0 (trivial 1-point grid, no bug)" begin
            a = Sequence(Taylor(0), [5.0])
            @test to_grid(a) == ComplexF64[5]
            @test to_coef(ComplexF64[5], Taylor(0)) == Sequence(Taylor(0), ComplexF64[5])
        end

        @testset "complex coefficients" begin
            a = Sequence(Taylor(3), ComplexF64[1.0+1.0im, -2.0, 0.5-3.0im, 3.0])
            b = to_coef(to_grid(a), Taylor(3))
            @test coefficients(b) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval coefficients (enclosure round trip)" begin
            a = Sequence(Taylor(2), interval.([1.0, 2.0, 3.0]))
            b = to_coef(to_grid(a), Taylor(2))
            for k ∈ 0:2
                @test issubset_interval(a[k], real(b[k]))
                @test in_interval(0.0, imag(b[k]))
            end
            # Complex{Interval{Float64}} coefficients
            ac = Sequence(Taylor(2), Complex{Interval{Float64}}[complex(interval(1.0), interval(0.5)), interval(2.0), interval(3.0)])
            bc = to_coef(to_grid(ac), Taylor(2))
            for k ∈ 0:2
                @test issubset_interval(real(ac[k]), real(bc[k]))
                @test issubset_interval(imag(ac[k]), imag(bc[k]))
            end
        end

        @testset "custom oversampled grid size, of any shape" begin
            a = Sequence(Taylor(2), [1.0, -2.0, 3.0])
            for m ∈ (3, 5, 6, 10, 16) # fft_size(Taylor(2)) == 3
                g = to_grid(a, m) # zero-padded
                @test length(g) == m
                b = to_coef(g, Taylor(2))
                @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
            end
        end

        @testset "resampling to a different space via to_coef(::Sequence, ::SequenceSpace)" begin
            a = Sequence(Taylor(1), [1.0, 2.0])
            # zero-padding to a higher order
            @test to_coef(a, Taylor(3)) == Sequence(Taylor(3), ComplexF64[1, 2, 0, 0])
            # truncating to a lower order
            @test to_coef(a, Taylor(0)) == Sequence(Taylor(0), ComplexF64[1])
        end
    end

    #

    @testset "Fourier" begin
        @testset "hand-computed round trip on the four-point grid (only ±1,±i roots)" begin
            # a = c₋₁·e^{-iθ} + c₀ + c₁·e^{iθ} with (c₋₁,c₀,c₁) = (1,2,3).
            # Preprocessing circshifts the zero-frequency mode to position 1:
            # [c₋₁,c₀,c₁,0] -> [c₀,c₁,0,c₋₁] = [2,3,0,1], then the backward DFT gives
            #   Y0 = 2+3+0+1  = 6
            #   Y1 = 2+3i-0-i = 2+2i
            #   Y2 = 2-3+0-1  = -2
            #   Y3 = 2-3i-0+i = 2-2i
            a = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0])
            expected_grid = ComplexF64[6, 2+2im, -2, 2-2im]
            @test fft_size(space(a)) == (3,) # the tightest grid is the three-point one
            @test to_grid(a, 4) == expected_grid
            @test to_grid!(fill(complex(Inf, Inf), 4), a) == expected_grid
            seeded = Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf])
            @test to_coef!(seeded, copy(expected_grid)) == Sequence(Fourier(1, 1.0), ComplexF64[1, 2, 3]) == seeded
        end

        @testset "round trip at every order, including sizes with a large prime factor" begin
            for n ∈ [0:20; 33; 50; 53] # 67, 101 and 107 are prime
                a = Sequence(Fourier(n, 1.0), collect(1.0:2n+1))
                b = to_coef(to_grid(a), Fourier(n, 1.0))
                @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-10
            end
        end

        @testset "approximate round trip, higher order (N=5: irrational roots, roundoff genuine)" begin
            a = Sequence(Fourier(2, 1.0), [1.0, -0.5, 2.0, -0.5, 1.0])
            b = to_coef(to_grid(a), Fourier(2, 1.0))
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "complex coefficients" begin
            a = Sequence(Fourier(2, 1.0), ComplexF64[1.0-1.0im, -0.5, 2.0+0.5im, -0.5, 1.0])
            b = to_coef(to_grid(a), Fourier(2, 1.0))
            @test coefficients(b) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval coefficients (enclosure round trip)" begin
            a = Sequence(Fourier(2, 1.0), interval.([1.0, 2.0, 3.0, 4.0, 5.0]))
            b = to_coef(to_grid(a), Fourier(2, 1.0))
            for k ∈ -2:2
                @test issubset_interval(a[k], real(b[k]))
            end
        end

        @testset "order-0 (trivial 1-point grid, no bug)" begin
            s = Fourier(0, 1.0)
            a = Sequence(s, [7.0])
            @test to_grid(a) == ComplexF64[7]
            b = to_coef(to_grid(a), s)
            @test b == Sequence(s, ComplexF64[7])
            # an oversized grid also works, confirming the default grid size agrees
            b_ok = to_coef(to_grid(a, 2), s)
            @test b_ok == Sequence(s, ComplexF64[7])
            @test to_coef(_ -> 7.0, s) == Sequence(s, ComplexF64[7])
        end

        @testset "a coarse grid leaves the space under-determined" begin
            # one node determines the zero mode and nothing else; the interpolant is
            # the constant going through it, the remaining modes vanishing
            s = Fourier(2, 1.0)
            c = Sequence(s, fill(complex(Inf, Inf), 5))
            to_coef!(c, ComplexF64[7.0])
            @test real(c[0]) ≈ 7.0
            @test all(k -> c[k] == 0, (-2, -1, 1, 2))
            # three nodes determine the modes -1, 0, 1, and the interpolant goes through them
            g = ComplexF64[1, 2, 3]
            b = to_coef(g, s)
            @test all(k -> b[k] == 0, (-2, 2))
            @test all(j -> b(2π*j/3) ≈ g[j+1], 0:2)
        end
    end

    #

    @testset "Chebyshev" begin
        @testset "exact hand-computed round trip (order 1, N=2)" begin
            # a(x) = c0 + c1·T1(x) with (c0,c1) = (1,2).
            # Preprocessing doubles the Nyquist entry: [1,2] -> [1,4], then the
            # backward DFT of size 2 gives Y0 = 1+4 = 5, Y1 = 1-4 = -3.
            a = Sequence(Chebyshev(1), [1.0, 2.0])
            expected_grid = ComplexF64[5, -3]
            @test fft_size(space(a)) == (2,)
            @test to_grid(a) == expected_grid
            @test to_grid!(fill(complex(Inf, Inf), 2), a) == expected_grid
            seeded = Sequence(Chebyshev(1), ComplexF64[Inf, Inf])
            @test to_coef!(seeded, copy(expected_grid)) == Sequence(Chebyshev(1), ComplexF64[1, 2]) == seeded
        end

        @testset "approximate round trip, higher order (N=6: irrational roots, roundoff genuine)" begin
            a = Sequence(Chebyshev(3), [1.0, -2.0, 0.5, 3.0])
            b = to_coef(to_grid(a), Chebyshev(3))
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "round trip at every order, including sizes with a large prime factor" begin
            for n ∈ [0:20; 37; 67; 101] # the transforms have length 2n
                a = Sequence(Chebyshev(n), collect(1.0:n+1))
                b = to_coef(to_grid(a), Chebyshev(n))
                @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-10
            end
        end

        @testset "order-0 (trivial 1-point grid, no bug)" begin
            a = Sequence(Chebyshev(0), [5.0])
            @test to_grid(a) == ComplexF64[5]
            @test to_coef(ComplexF64[5], Chebyshev(0)) == Sequence(Chebyshev(0), ComplexF64[5])
        end

        @testset "complex coefficients" begin
            a = Sequence(Chebyshev(3), ComplexF64[1.0+2.0im, -2.0, 0.5-1.0im, 3.0])
            b = to_coef(to_grid(a), Chebyshev(3))
            @test coefficients(b) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval coefficients (enclosure round trip)" begin
            a = Sequence(Chebyshev(3), interval.([1.0, 2.0, 3.0, 4.0]))
            b = to_coef(to_grid(a), Chebyshev(3))
            for k ∈ 0:3
                @test issubset_interval(a[k], real(b[k]))
            end
        end
    end

    #

    @testset "TensorSpace" begin
        @testset "exact hand-computed trivial round trip (both factors order 0)" begin
            s = Taylor(0) ⊗ Chebyshev(0)
            @test fft_size(s) == (1, 1)
            a = Sequence(s, [7.0])
            g = to_grid(a)
            @test size(g) == (1, 1)
            @test g == ComplexF64[7;;]
            b = to_coef(g, s)
            @test b == Sequence(s, ComplexF64[7])
        end

        @testset "approximate round trip (mixed Taylor/Fourier, roundoff genuine)" begin
            s = Taylor(2) ⊗ Fourier(1, 1.0)
            a = Sequence(s, collect(1.0:dimension(s)))
            g = to_grid(a)
            @test size(g) == grid_size(s) == (3, 3)
            b = to_coef(g, s)
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "complex coefficients" begin
            s = Taylor(1) ⊗ Fourier(1, 1.0)
            a = Sequence(s, ComplexF64.(1.0:dimension(s)) .+ im .* (dimension(s):-1.0:1.0))
            b = to_coef(to_grid(a), s)
            @test coefficients(b) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval coefficients (enclosure round trip)" begin
            s = Taylor(1) ⊗ Fourier(1, 1.0)
            a = Sequence(s, interval.(collect(1.0:dimension(s))))
            b = to_coef(to_grid(a), s)
            for k ∈ indices(s)
                @test issubset_interval(a[k], real(b[k]))
            end
        end
    end

    #

    @testset "SymmetricSpace" begin
        @testset "evensym(Taylor): expands to zero odd-order coefficients then round-trips" begin
            s = evensym(Taylor(3))
            @test indices(s) == 0:2:2
            a = Sequence(s, [1.0, 3.0])
            full = Projection(desymmetrize(s)) * a
            @test coefficients(full) == [1.0, 0.0, 3.0, 0.0] # odd orders forced to 0 by the symmetry
            @test to_grid(a) == to_grid(full) # `to_grid!` on a SymmetricSpace projects first, then delegates
            c = Sequence(s, ComplexF64[Inf, Inf])
            to_coef!(c, to_grid(a))
            @test real.(coefficients(c)) ≈ coefficients(a) atol=1e-9
        end

        @testset "oddsym(Fourier)" begin
            s = oddsym(Fourier(2, 1.0))
            @test indices(s) == 1:2
            a = Sequence(s, [10.0, 20.0])
            c = Sequence(s, ComplexF64[Inf, Inf])
            to_coef!(c, to_grid(a))
            @test real.(coefficients(c)) ≈ coefficients(a) atol=1e-9
        end

        @testset "evensym(Chebyshev)" begin
            s = evensym(Chebyshev(4))
            @test indices(s) == 0:2:4
            a = Sequence(s, [1.0, 2.0, 3.0])
            c = Sequence(s, ComplexF64[Inf, Inf, Inf])
            to_coef!(c, to_grid(a))
            @test real.(coefficients(c)) ≈ coefficients(a) atol=1e-9
        end

        @testset "d4sym(Fourier ⊗ Fourier)" begin
            s = d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))
            @test fft_size(s) == (5, 5)
            a = Sequence(s, collect(1.0:dimension(s)))
            c = Sequence(s, fill(complex(Inf, Inf), dimension(s)))
            to_coef!(c, to_grid(a))
            @test real.(coefficients(c)) ≈ coefficients(a) atol=1e-9
        end
    end

    #

    @testset "function interpolation to_coef(f, space)" begin
        @testset "Taylor (exact, N=4)" begin
            # f(z) = z; nodes are z_j = e^{i2πj/4} ∈ {1,i,-1,-i}, exactly interpolated
            b = to_coef(z -> z, Taylor(1))
            @test b == Sequence(Taylor(1), ComplexF64[0, 1])
        end

        @testset "Fourier (approximate: cos/sin of irrational nodes)" begin
            # 2cos(x) = e^{ix}+e^{-ix} on Fourier(1,1.0): c₋₁=1, c₀=0, c₁=1
            b = to_coef(x -> 2*cos(x), Fourier(1, 1.0))
            @test real.(coefficients(b)) ≈ [1.0, 0.0, 1.0] atol=1e-9
            # frequency parameter is honoured by the node formula 2π/frequency·j/N
            b2 = to_coef(x -> 2*cos(2x), Fourier(1, 2.0))
            @test real.(coefficients(b2)) ≈ [1.0, 0.0, 1.0] atol=1e-9
        end

        @testset "Chebyshev (note: interior modes carry an implicit factor 2)" begin
            # RadiiPolynomial's Chebyshev sequences satisfy f(x) = c0·T0(x) + 2·Σ_{k≥1} ck·Tk(x),
            # matching the halving of the Nyquist mode in `_postprocess_to_coef!`.
            # Hence interpolating f(x) = x = T1(x) gives c1 = 0.5, not 1.
            b1 = to_coef(x -> x, Chebyshev(1))
            @test real.(coefficients(b1)) ≈ [0.0, 0.5] atol=1e-9
            # f(x) = 2x²-1 = T2(x) gives c2 = 0.5
            b2 = to_coef(x -> 2x^2 - 1, Chebyshev(2))
            @test real.(coefficients(b2)) ≈ [0.0, 0.0, 0.5] atol=1e-9
        end

        @testset "TensorSpace (exact, both factors order 1)" begin
            # f(x,y) = xy on Taylor(1) ⊗ Taylor(1): only the (1,1) mode is nonzero
            s = Taylor(1) ⊗ Taylor(1)
            b = to_coef((x, y) -> x*y, s)
            @test b == Sequence(s, ComplexF64[0, 0, 0, 1])
        end

        @testset "the interpolant goes through the nodes, at every order" begin
            f(x) = 1/(2-x)
            for n ∈ 1:12
                b = to_coef(f, Chebyshev(n))
                m = only(grid_size(Chebyshev(n)))
                @test all(k -> b(cospi((k-1)/(m-1))) ≈ f(cospi((k-1)/(m-1))), 1:m)

                b = to_coef(f, Taylor(n))
                m = only(grid_size(Taylor(n)))
                @test all(j -> b(cispi(2j/m)) ≈ f(cispi(2j/m)), 0:m-1)

                g(x) = 1/(2-cos(x))
                b = to_coef(g, Fourier(n, 1.0))
                m = only(grid_size(Fourier(n, 1.0)))
                @test all(j -> b(2π*j/m) ≈ g(2π*j/m), 0:m-1)
            end
        end
    end

    #

    @testset "grid <-> array entry points" begin
        a = Sequence(Taylor(1), [1.0, 2.0])
        grid = to_grid(a)
        grid_copy = copy(grid)

        @testset "to_coef(::AbstractArray, ::SequenceSpace) does not mutate its input" begin
            b = to_coef(grid, Taylor(1))
            @test grid == grid_copy
            @test b == Sequence(Taylor(1), ComplexF64[1, 2])
        end

        @testset "to_coef! fills the sequence and consumes the grid" begin
            c = Sequence(Taylor(1), ComplexF64[Inf, Inf])
            g = copy(grid)
            out = to_coef!(c, g)
            @test out === c
            @test c == Sequence(Taylor(1), ComplexF64[1, 2])
            @test g != grid_copy # transformed in place, no buffer allocated
        end
    end

    #

    @testset "_call_to_coef! internal helper" begin
        a = Sequence(Taylor(1), [1.0, 2.0])
        grid = to_grid(a)

        @test RadiiPolynomial._call_to_coef!(copy(grid), Taylor(1), Float64) == real(to_coef(grid, Taylor(1)))
        @test RadiiPolynomial._call_to_coef!(copy(grid), Taylor(1), Float64) isa Sequence{Taylor,Vector{Float64}}
        @test RadiiPolynomial._call_to_coef!(copy(grid), Taylor(1), ComplexF64) == to_coef(grid, Taylor(1))
        @test RadiiPolynomial._call_to_coef!(copy(grid), Taylor(1), ComplexF64) isa Sequence{Taylor,Vector{ComplexF64}}
    end

    #

    @testset "twiddle tables" begin
        # the `radius` field is the ρ of the a priori bound; it must bound the
        # MODULUS of `mid - exact`, which is not what `radius` of a complex
        # interval returns (that is the larger of the two component radii, too
        # small by up to √2), and would be 0 if read off the midpoints
        for T ∈ (Float64, Float32), len ∈ (4, 64)
            W = RadiiPolynomial._roots_of_unity(T, len)
            @test W.radius > 0
            @test all(w -> sup(abs(w - mid(w))) ≤ W.radius, W.interval)
            @test all(in_interval(real(w), real(W_)) && in_interval(imag(w), imag(W_))
                      for (w, W_) ∈ zip(W.mid, W.interval))
        end
    end

    @testset "arbitrary transform length" begin
        naive_dft(x) = [sum(x[j+1] * cispi(-2*(j*k)/length(x)) for j ∈ 0:length(x)-1) for k ∈ 0:length(x)-1]

        # radix-2 (16), decimation (12 = 2·6, 105 = 3·5·7), direct sum (13),
        # Bluestein (67 > RadiiPolynomial.NAIVE_DFT_MAX)
        @testset "forward transform of length $n" for n ∈ (1, 2, 3, 5, 12, 13, 16, 67, 105)
            x = [ComplexF64(cospi(2k/n), sinpi(3k/(n+1))) for k ∈ 0:n-1]
            @test RadiiPolynomial._fft!(copy(x)) ≈ naive_dft(x) rtol=1e-12 # the reference rounds too
            @test RadiiPolynomial._bfft!(RadiiPolynomial._fft!(copy(x))) ./ n ≈ x rtol=1e-12
        end

        @testset "each dimension is transformed, whatever its length" begin
            x = [ComplexF64(k, j) for k ∈ 1:6, j ∈ 1:5]
            y = RadiiPolynomial._fft!(copy(x))
            @test y ≈ mapslices(naive_dft, mapslices(naive_dft, x; dims = 1); dims = 2) atol=1e-12
            @test RadiiPolynomial._bfft!(copy(y)) ./ length(x) ≈ x atol=1e-12
        end

        @testset "encloses the exact transform of length $n" for n ∈ (5, 12, 13, 67)
            x = round.([ComplexF64(cospi(2k/n), sinpi(3k/n)) for k ∈ 1:n], digits=6)
            exact_dft = setprecision(384) do
                [sum(Complex{BigFloat}(x[j+1]) * cispi(-2*BigFloat(j*k)/n) for j ∈ 0:n-1) for k ∈ 0:n-1]
            end
            y = RadiiPolynomial._fft!([complex(interval(real(z)), interval(imag(z))) for z ∈ x])
            @test all(k -> in_interval(Float64(real(exact_dft[k])), real(y[k])) &&
                           in_interval(Float64(imag(exact_dft[k])), imag(y[k])), 1:n)
        end
    end

    #

    @testset "grid_size" begin
        # Taylor and Fourier sample as many nodes as the transform has points
        @test grid_size(Taylor(2)) == fft_size(Taylor(2)) == (3,)
        @test grid_size(Fourier(2, 1.0)) == fft_size(Fourier(2, 1.0)) == (5,)
        # Chebyshev: the Chebyshev–Lobatto nodes cos(π(k-1)/(m-1)) the mirror folds onto
        @test grid_size(Chebyshev(2)) == (3,) # fft_size 4
        @test grid_size(Chebyshev(5)) == (6,) # fft_size 10
        @test grid_size(Chebyshev(0)) == (1,)
        @test grid_size(Chebyshev(2) ⊗ Fourier(1, 1.0)) == (3, 3)
        @test grid_size(evensym(Chebyshev(4))) == grid_size(Chebyshev(4)) == (5,)
    end

    #

    @testset "Chebyshev half grid (Chebyshev–Lobatto nodes)" begin
        @testset "the grid holds the nodes only, the mirror staying internal" begin
            a = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
            half = to_grid(a) # (3,): nodes 1, 0, -1, from the mirror of length 4
            @test half == to_grid(a, grid_size(space(a)))
            @test half == RadiiPolynomial._to_grid!(zeros(ComplexF64, 4), a)[1:3]
            b = to_coef(half, Chebyshev(2))
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-12
            # a finer grid of nodes interpolates the same sequence
            @test coefficients(to_coef(to_grid(a, 5), Chebyshev(2))) ≈ coefficients(b) atol=1e-12
        end

        @testset "nodes are ordered from x = 1 down to x = -1" begin
            # f(x) = x sampled with the implicit factor 2 convention: c1 = 0.5
            m = only(grid_size(Chebyshev(2))) # 3
            g = [cospi((k-1)/(m-1)) for k ∈ 1:m] # 1, 0, -1
            b = to_coef(g, Chebyshev(2))
            @test real.(coefficients(b)) ≈ [0.0, 0.5, 0.0] atol=1e-12
        end

        @testset "tensor space: folds along every Chebyshev axis" begin
            s = Chebyshev(2) ⊗ Chebyshev(1)
            a = Sequence(s, collect(1.0:dimension(s)))
            half = to_grid(a) # (3, 2), from a (4, 2) mirror
            @test half == grid_size(s) |> m -> to_grid(a, m)
            @test half == RadiiPolynomial._to_grid!(zeros(ComplexF64, 4, 2), a)[1:3, 1:2]
            b = to_coef(half, s)
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "mixed tensor space: every axis counts its own nodes" begin
            s = Chebyshev(2) ⊗ Fourier(1, 1.0)
            a = Sequence(s, collect(1.0:dimension(s)))
            half = to_grid(a, grid_size(s))
            @test size(half) == grid_size(s) == (3, 3)
            b = to_coef(half, s)
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval enclosure round trip through the half grid" begin
            a = Sequence(Chebyshev(3), interval.([1.0, 2.0, 3.0, 4.0]))
            b = to_coef(to_grid(a, grid_size(space(a))), Chebyshev(3))
            for k ∈ 0:3
                @test issubset_interval(a[k], real(b[k]))
            end
        end

        @testset "any number of nodes is accepted, down to the order" begin
            a = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
            for m ∈ 3:8 # Chebyshev(2) needs 3 nodes
                @test size(to_grid(a, m)) == (m,)
                @test real.(coefficients(to_coef(to_grid(a, m), Chebyshev(2)))) ≈ coefficients(a) atol=1e-12
            end
            # two nodes cannot hold the transform of a quadratic
            @test_throws DimensionMismatch to_grid(a, (2,))
            # they do interpolate into it, the quadratic mode being left at zero:
            # f(x) = x through the nodes 1, -1, with the implicit factor 2 on c₁
            @test coefficients(to_coef(ComplexF64[1, -1], Chebyshev(2))) == ComplexF64[0, 0.5, 0]
        end
    end

    #

    @testset "grids of Sequences (parameter families)" begin
        @testset "Chebyshev parameter: round trip and node convention" begin
            s_par, s_inner = Chebyshev(2), Fourier(1, 1.0)
            fs = s_par ⊗ s_inner
            a = Sequence(fs, collect(1.0:dimension(fs)))
            x_grid = to_grid(a, grid_size(s_par))
            @test x_grid isa Vector
            @test size(x_grid) == grid_size(s_par) == (3,)
            @test all(x -> space(x) == s_inner, x_grid)
            # grid elements are the partial evaluations at the Chebyshev–Lobatto nodes
            m = only(grid_size(s_par))
            for k ∈ 1:m
                x_k = cospi((k-1)/(m-1))
                @test real.(coefficients(x_grid[k])) ≈ coefficients(Evaluation(x_k, nothing) * a) atol=1e-9
            end
            b = to_coef(x_grid, s_par)
            @test space(b) == fs
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "Fourier and Taylor parameters" begin
            s7 = Fourier(1, 1.0)
            a7 = Sequence(s7 ⊗ Taylor(1), collect(1.0:dimension(s7 ⊗ Taylor(1))))
            b7 = to_coef(to_grid(a7, grid_size(s7)), s7)
            @test coefficients(b7) ≈ coefficients(a7) atol=1e-9

            s8 = Taylor(1)
            a8 = Sequence(s8 ⊗ Chebyshev(1), collect(1.0:dimension(s8 ⊗ Chebyshev(1))))
            b8 = to_coef(to_grid(a8, grid_size(s8)), s8)
            @test coefficients(b8) ≈ coefficients(a8) atol=1e-9
        end

        @testset "two parameter axes give a Matrix grid" begin
            s2 = Chebyshev(1) ⊗ Chebyshev(2)
            full_space = s2 ⊗ Taylor(1)
            a = Sequence(full_space, collect(1.0:dimension(full_space)))
            g = to_grid(a, grid_size(s2))
            @test g isa Matrix
            @test size(g) == grid_size(s2) == (2, 3)
            b = to_coef(g, s2)
            @test space(b) == full_space
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "tensor inner space flattens" begin
            s_par = Chebyshev(1)
            s_inner = Taylor(1) ⊗ Fourier(1, 1.0)
            fs = s_par ⊗ s_inner
            a = Sequence(fs, collect(1.0:dimension(fs)))
            x_grid = to_grid(a, grid_size(s_par))
            @test all(x -> space(x) == s_inner, x_grid)
            b = to_coef(x_grid, s_par)
            @test space(b) == fs
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval enclosure round trip" begin
            fs = Chebyshev(2) ⊗ Fourier(1, 1.0)
            a = Sequence(fs, interval.(collect(1.0:dimension(fs))))
            b = to_coef(to_grid(a, (3,)), Chebyshev(2))
            for k ∈ indices(fs)
                @test issubset_interval(a[k], real(b[k]))
            end
        end

        @testset "resampling on a finer grid (more points than the order needs)" begin
            # sampling a low-order family on a finer grid is exact (zero-padding)
            s_par = Chebyshev(2)
            a = Sequence(s_par ⊗ Fourier(1, 1.0), collect(1.0:9))
            fine = to_grid(a, (9,))
            @test size(fine) == (9,)
            m = 9
            for k ∈ (1, 5, 9)
                x_k = cospi((k-1)/(m-1))
                @test real.(coefficients(fine[k])) ≈ coefficients(Evaluation(x_k, nothing) * a) atol=1e-9
            end
            # interpolating back at the original order is exact (content is order 2)
            b = to_coef(fine, s_par)
            @test space(b) == space(a)
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
            # interpolating at the finer order: the extra coefficients vanish
            b8 = to_coef(fine, Chebyshev(8))
            @test real.(coefficients(project(b8, space(a)))) ≈ coefficients(a) atol=1e-9

            # full-space scalar resampling
            c = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
            g = to_grid(c, (5,))
            @test size(g) == (5,)
            @test real.(coefficients(project(to_coef(g, Chebyshev(4)), Chebyshev(2)))) ≈ coefficients(c) atol=1e-9

            # a Fourier parameter resamples too, on a grid of any size
            af = Sequence(Fourier(1, 1.0) ⊗ Taylor(1), collect(1.0:6))
            for m ∈ (3, 6, 8)
                gf = to_grid(af, (m,))
                @test size(gf) == (m,)
                @test coefficients(to_coef(gf, Fourier(1, 1.0))) ≈ coefficients(af) atol=1e-9
            end
            @test_throws DimensionMismatch to_grid(af, (2,)) # too coarse for order 1

            # a grid too coarse for the family is rejected
            a8 = Sequence(Chebyshev(8) ⊗ Taylor(1), collect(1.0:18))
            @test_throws DimensionMismatch to_grid(a8, (2,))
        end

        @testset "symmetric inner space: reduced coefficients round trip" begin
            s_par = Chebyshev(2)
            inner_sym = evensym(Fourier(2, 1.0))
            prod_sym = s_par ⊗ inner_sym
            a = project(Sequence(s_par ⊗ Fourier(2, 1.0), collect(1.0:15)), prod_sym)
            x_grid = to_grid(a, grid_size(s_par))
            # grid elements carry the restricted symmetry group
            @test all(x -> space(x) == inner_sym, x_grid)
            # grid elements are the partial evaluations of the desymmetrized family
            full = Projection(desymmetrize(prod_sym)) * a
            m = only(grid_size(s_par))
            for k ∈ 1:m
                x_k = cospi((k-1)/(m-1))
                @test real.(coefficients(Projection(Fourier(2, 1.0)) * x_grid[k])) ≈ coefficients(Evaluation(x_k, nothing) * full) atol=1e-9
            end
            b = to_coef(x_grid, s_par)
            @test space(b) == prod_sym
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "abstractly typed grid containers" begin
            # a grid assembled as `Matrix{Sequence}` / `Vector{Sequence}` (the natural way to
            # preallocate a continuation grid) has `eltype(eltype(x_grid)) === Any`, so the
            # coefficient type must be derived from the elements themselves
            s_par = Chebyshev(2)
            inner = Fourier(1, 1.0)
            a = Sequence(s_par ⊗ inner, collect(1.0:9))
            m = grid_size(s_par)

            ref = to_grid(a, m)
            x_grid = Vector{Sequence}(undef, length(ref)) # abstract eltype
            copyto!(x_grid, ref)
            b = to_coef(x_grid, s_par)
            @test space(b) == s_par ⊗ inner
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-12

            # in-place `to_grid!` into an abstractly typed container
            buf = Vector{Sequence}(undef, length(ref))
            for i ∈ eachindex(buf)
                buf[i] = Sequence(inner, zeros(ComplexF64, dimension(inner)))
            end
            to_grid!(buf, a)
            @test all(i -> coefficients(buf[i]) ≈ coefficients(ref[i]), eachindex(buf))

            # two parameter axes, operator grid: the abstract container must agree with the
            # concrete one it was copied from
            s_par2 = Chebyshev(1) ⊗ Chebyshev(1)
            ops = [LinearOperator(inner, inner, rand(dimension(inner), dimension(inner)))
                   for _ ∈ CartesianIndices(grid_size(s_par2))]
            B_concrete = to_coef(ops, s_par2)
            A_grid = Matrix{LinearOperator}(undef, size(ops)...) # abstract eltype
            copyto!(A_grid, ops)
            B_abstract = to_coef(A_grid, s_par2)
            @test domain(B_abstract) == domain(B_concrete)
            @test codomain(B_abstract) == codomain(B_concrete)
            @test coefficients(B_abstract) == coefficients(B_concrete)
        end

        @testset "symmetric inner space: interval enclosure round trip" begin
            s_par = Chebyshev(1)
            prod_sym = s_par ⊗ evensym(Fourier(1, 1.0))
            a = project(Sequence(s_par ⊗ Fourier(1, 1.0), interval.(collect(1.0:6))), interval(prod_sym))
            b = to_coef(to_grid(a, grid_size(s_par)), s_par)
            for n ∈ eachindex(coefficients(a))
                @test issubset_interval(coefficients(a)[n], real(coefficients(b)[n]))
            end
        end

        @testset "error paths" begin
            # the grid must have at most one axis per factor, and at least one
            @test_throws ArgumentError to_grid(Sequence(Fourier(1, 1.0), ones(3)), (4, 4))
            @test_throws ArgumentError to_grid(Sequence(Fourier(1, 1.0) ⊗ Taylor(1), ones(6)), ())
            # grid elements on mismatched spaces
            @test_throws ArgumentError to_coef([Sequence(Taylor(1), [1.0, 2.0]), Sequence(Taylor(2), [1.0, 2.0, 3.0])], Chebyshev(1))
            # grid dimension must match the number of factors of `s`
            @test_throws ArgumentError to_coef(fill(Sequence(Taylor(1), [1.0, 2.0]), 3, 3), Chebyshev(1))
            # the symmetry group must not mix the grid and inner directions
            d4 = d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0)) # swaps the two factors
            @test_throws ArgumentError to_grid(Sequence(d4, ones(dimension(d4))), (4,))
        end
    end

    #

    @testset "grids of Sequences on a CartesianSpace" begin
        @testset "the components share the nodes, hence a single grid size" begin
            @test grid_size(Chebyshev(2)^3) == grid_size(Chebyshev(2)) == (3,)
            @test fft_size(Chebyshev(2)^3) == fft_size(Chebyshev(2)) == (4,)
            # the finest component sets the size, the others being oversampled
            @test grid_size(Chebyshev(2) × Chebyshev(4)) == grid_size(Chebyshev(4)) == (5,)
            @test fft_size(Chebyshev(2) × Chebyshev(4)) == (8,)
            @test grid_size((Chebyshev(2) ⊗ Fourier(1, 1.0)) × (Chebyshev(4) ⊗ Fourier(2, 1.0))) == (5, 5)
            # nested cartesian products and symmetries defer to the space underneath
            @test grid_size((Taylor(1)^2 × Taylor(3))^2) == grid_size(Taylor(3)) == (4,)
            @test grid_size(evensym(Fourier(2, 1.0))^2) == grid_size(Fourier(2, 1.0)) == (5,)
        end

        @testset "a scalar family: the grid holds the values of every component" begin
            s = Chebyshev(2)^3
            a = Sequence(s, collect(1.0:9))
            x_grid = to_grid(a)
            @test x_grid isa Vector
            @test size(x_grid) == grid_size(s) == (3,)
            @test all(x -> space(x) == ScalarSpace()^3, x_grid)
            # each component is sampled as it would be on its own
            for i ∈ 1:3
                @test [coefficients(x)[i] for x ∈ x_grid] == to_grid(component(a, i))
            end
            b = to_coef(x_grid, Chebyshev(2))
            @test space(b) == s
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "a product of families: the grid holds Sequences on the inner spaces" begin
            s_par = Chebyshev(2)
            fs = (s_par ⊗ Fourier(1, 1.0)) × (s_par ⊗ Taylor(1))
            a = Sequence(fs, collect(1.0:dimension(fs)))
            x_grid = to_grid(a, grid_size(s_par))
            @test all(x -> space(x) == Fourier(1, 1.0) × Taylor(1), x_grid)
            # grid elements are the partial evaluations at the Chebyshev–Lobatto nodes
            m = only(grid_size(s_par))
            for k ∈ 1:m
                x_k = cospi((k-1)/(m-1))
                @test real.(coefficients(x_grid[k])) ≈ coefficients(Evaluation(x_k, nothing) * a) atol=1e-9
            end
            b = to_coef(x_grid, s_par)
            @test space(b) == fs
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "cartesian products nest, entirely discretized components included" begin
            s_par = Chebyshev(1)
            fs = (s_par ⊗ Fourier(1, 1.0))^2 × s_par
            a = Sequence(fs, collect(1.0:dimension(fs)))
            x_grid = to_grid(a, grid_size(s_par))
            @test all(x -> space(x) == Fourier(1, 1.0)^2 × ScalarSpace(), x_grid)
            b = to_coef(x_grid, s_par)
            @test space(b) == fs
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "components of different orders: the coarser one is oversampled" begin
            s = Chebyshev(2) × Chebyshev(4)
            a = Sequence(s, collect(1.0:8))
            x_grid = to_grid(a) # 5 nodes, set by the second component
            @test size(x_grid) == (5,)
            b = to_coef(x_grid, Chebyshev(4))
            @test space(b) == Chebyshev(4)^2 # every component is interpolated onto Chebyshev(4)
            @test real.(coefficients(project(b, s))) ≈ coefficients(a) atol=1e-9
        end

        @testset "symmetric inner space: the components keep their symmetry" begin
            s_par = Chebyshev(2)
            inner_sym = evensym(Fourier(2, 1.0))
            fs = (s_par ⊗ inner_sym)^2
            a = project(Sequence((s_par ⊗ Fourier(2, 1.0))^2, collect(1.0:30)), fs)
            x_grid = to_grid(a, grid_size(s_par))
            @test all(x -> space(x) == inner_sym^2, x_grid)
            b = to_coef(x_grid, s_par)
            @test space(b) == fs
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "in-place entry points" begin
            s_par = Chebyshev(2)
            fs = (s_par ⊗ Fourier(1, 1.0)) × s_par
            a = Sequence(fs, collect(1.0:dimension(fs)))
            ref = to_grid(a, grid_size(s_par))

            x_grid = [Sequence(Fourier(1, 1.0) × ScalarSpace(), fill(complex(Inf, Inf), 4)) for _ ∈ 1:3]
            @test to_grid!(x_grid, a) === x_grid
            @test all(i -> coefficients(x_grid[i]) ≈ coefficients(ref[i]), eachindex(x_grid))

            c = Sequence(fs, fill(complex(Inf, Inf), dimension(fs)))
            @test to_coef!(c, ref) === c
            @test real.(coefficients(c)) ≈ coefficients(a) atol=1e-9
        end

        @testset "resampling via to_coef(::Sequence, ::SequenceSpace)" begin
            a = Sequence(Chebyshev(2)^3, collect(1.0:9))
            b = to_coef(a, Chebyshev(4)) # zero-padding to a higher order
            @test space(b) == Chebyshev(4)^3
            @test real.(coefficients(project(b, space(a)))) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval coefficients (enclosure round trip)" begin
            fs = (Chebyshev(2) ⊗ Fourier(1, 1.0)) × Chebyshev(2)
            a = Sequence(fs, interval.(collect(1.0:dimension(fs))))
            b = to_coef(to_grid(a, (3,)), Chebyshev(2))
            @test space(b) == fs
            for n ∈ eachindex(coefficients(a))
                @test issubset_interval(coefficients(a)[n], real(coefficients(b)[n]))
            end
        end

        @testset "error paths" begin
            # the components must be sampled on the same nodes
            @test_throws MethodError to_grid(Sequence(Chebyshev(2) × Fourier(1, 1.0), ones(6)))
            @test_throws ArgumentError to_grid(Sequence(Fourier(1, 1.0) × Fourier(1, 2.0), ones(6)), (3,))
            @test_throws MethodError to_grid(Sequence(Chebyshev(2) × (Chebyshev(2) ⊗ Taylor(1)), ones(9)))
            # a component with no factor to discretize
            @test_throws MethodError to_grid(Sequence(Chebyshev(2) × ScalarSpace(), ones(4)))
            @test_throws MethodError to_grid(Sequence(Chebyshev(2) × ScalarSpace(), ones(4)), (3,))
            # grid elements on mismatched spaces
            @test_throws ArgumentError to_grid!(to_grid(Sequence(Chebyshev(2)^3, ones(9))), Sequence(Chebyshev(2)^2, ones(6)))
            # a grid too coarse for the family
            @test_throws DimensionMismatch to_grid(Sequence(Chebyshev(2)^2, ones(6)), (2,))
        end
    end

    #

    @testset "unsupported vector spaces" begin
        # `to_grid`/`to_coef` are only defined for `Sequence{<:SequenceSpace}`
        # and `Sequence{<:CartesianSpace}`; `ScalarSpace` is neither
        @test_throws MethodError to_grid(Sequence(ScalarSpace(), [1.0]))
        # a cartesian space of scalars is matched, but has nothing to discretize
        @test_throws MethodError to_grid(Sequence(ScalarSpace()^2, [1.0, 2.0]))
    end

    #

    @testset "error paths" begin
        @testset "offset arrays are rejected" begin
            a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
            @test_throws ArgumentError to_grid!(_OffsetVec(zeros(ComplexF64, 8)), a)

            c = Sequence(Taylor(2), zeros(ComplexF64, 3))
            @test_throws ArgumentError to_coef!(c, _OffsetVec(to_grid(a)))
        end

        @testset "to_coef! accepts any grid size" begin
            c = Sequence(Taylor(2), zeros(ComplexF64, 3))
            @test to_coef!(c, zeros(ComplexF64, 6)) === c # finer: truncated to the space
            @test to_coef!(c, zeros(ComplexF64, 2)) === c # coarser: the space is under-determined
        end

        @testset "to_grid! with an incompatible grid size throws DimensionMismatch" begin
            a = Sequence(Taylor(2), [1.0, 2.0, 3.0]) # 3 coefficients, needs n ≥ 3
            @test_throws DimensionMismatch to_grid!(fill(complex(Inf, Inf), 2), a)
        end
    end
end
