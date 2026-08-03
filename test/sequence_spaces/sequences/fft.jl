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
        @testset "Taylor: nextpow(2, 2order+1)" begin
            @test fft_size(Taylor(0)) == (1,)
            @test fft_size(Taylor(1)) == (4,)
            @test fft_size(Taylor(2)) == (8,)
            @test fft_size(Taylor(3)) == (8,)
            @test fft_size(Taylor(4)) == (16,)
        end

        @testset "Fourier: nextpow(2, 2order+1), independent of frequency" begin
            @test fft_size(Fourier(0, 1.0)) == (1,)
            @test fft_size(Fourier(1, 1.0)) == (4,)
            @test fft_size(Fourier(2, 1.0)) == (8,)
            @test fft_size(Fourier(3, 1.0)) == (8,)
            @test fft_size(Fourier(2, 3.5)) == fft_size(Fourier(2, 1.0)) # frequency does not affect the grid size
        end

        @testset "Chebyshev: nextpow(2, 2order + !ispow2(order))" begin
            @test fft_size(Chebyshev(0)) == (1,) # 2·0+1 (0 is not a power of 2)
            @test fft_size(Chebyshev(1)) == (2,) # 2·1+0 (1 = 2^0 is a power of 2)
            @test fft_size(Chebyshev(2)) == (4,) # 2·2+0
            @test fft_size(Chebyshev(3)) == (8,) # 2·3+1 = 7 -> nextpow2 = 8
            @test fft_size(Chebyshev(4)) == (8,) # 2·4+0
            @test fft_size(Chebyshev(5)) == (16,) # 2·5+1 = 11 -> nextpow2 = 16
        end

        @testset "TensorSpace: componentwise, one entry per factor" begin
            @test fft_size(Taylor(2) ⊗ Fourier(1, 1.0)) == (8, 4)
            @test fft_size(Taylor(1) ⊗ Taylor(1) ⊗ Chebyshev(3)) == (4, 4, 8)
        end

        @testset "SymmetricSpace: defers to the desymmetrized space" begin
            @test fft_size(evensym(Taylor(3))) == fft_size(Taylor(3)) == (8,)
            @test fft_size(oddsym(Fourier(3, 1.0))) == fft_size(Fourier(3, 1.0)) == (8,)
            @test fft_size(evensym(Chebyshev(3))) == fft_size(Chebyshev(3)) == (8,)
            @test fft_size(d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))) == fft_size(Fourier(2, 1.0) ⊗ Fourier(2, 1.0)) == (8, 8)
        end
    end

    #

    @testset "Taylor" begin
        @testset "exact hand-computed round trip (order 1, N=4 uses only ±1,±i roots)" begin
            # a(z) = 1 + 2z, padded to [1,2,0,0] and fed to the backward
            # (unnormalized inverse) DFT Y[j] = Σ_k C[k]·e^{i2πkj/4}:
            #   Y0 = 1+2       = 3
            #   Y1 = 1+2i      = 1+2i
            #   Y2 = 1-2       = -1
            #   Y3 = 1-2i      = 1-2i
            a = Sequence(Taylor(1), [1.0, 2.0])
            expected_grid = ComplexF64[3, 1+2im, -1, 1-2im]
            @test fft_size(space(a)) == (4,)
            @test to_grid(a) == expected_grid
            @test to_grid!(fill(complex(Inf, Inf), 4), a) == expected_grid
            @test to_seq(expected_grid, Taylor(1)) == Sequence(Taylor(1), ComplexF64[1, 2])
            @test to_seq!(copy(expected_grid), Taylor(1)) == Sequence(Taylor(1), ComplexF64[1, 2])
            seeded = Sequence(Taylor(1), ComplexF64[Inf, Inf])
            @test to_seq!(seeded, copy(expected_grid)) == Sequence(Taylor(1), ComplexF64[1, 2]) == seeded
        end

        @testset "aliasing" begin
            a = Sequence(Taylor(1), [1.0, 2.0])
            C = zeros(ComplexF64, 4)
            @test to_grid!(C, a) === C
        end

        @testset "approximate round trip, higher order (N=8: irrational roots, roundoff genuine)" begin
            a = Sequence(Taylor(3), [1.0, -2.0, 0.5, 3.0])
            b = to_seq(to_grid(a), Taylor(3))
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
            @test all(x -> isapprox(x, 0.0; atol=1e-9), imag.(coefficients(b)))
        end

        @testset "order-0 (trivial 1-point grid, no bug)" begin
            a = Sequence(Taylor(0), [5.0])
            @test to_grid(a) == ComplexF64[5]
            @test to_seq(ComplexF64[5], Taylor(0)) == Sequence(Taylor(0), ComplexF64[5])
        end

        @testset "complex coefficients" begin
            a = Sequence(Taylor(3), ComplexF64[1.0+1.0im, -2.0, 0.5-3.0im, 3.0])
            b = to_seq(to_grid(a), Taylor(3))
            @test coefficients(b) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval coefficients (enclosure round trip)" begin
            a = Sequence(Taylor(2), interval.([1.0, 2.0, 3.0]))
            b = to_seq(to_grid(a), Taylor(2))
            for k ∈ 0:2
                @test issubset_interval(a[k], real(b[k]))
                @test in_interval(0.0, imag(b[k]))
            end
            # Complex{Interval{Float64}} coefficients
            ac = Sequence(Taylor(2), Complex{Interval{Float64}}[complex(interval(1.0), interval(0.5)), interval(2.0), interval(3.0)])
            bc = to_seq(to_grid(ac), Taylor(2))
            for k ∈ 0:2
                @test issubset_interval(real(ac[k]), real(bc[k]))
                @test issubset_interval(imag(ac[k]), imag(bc[k]))
            end
        end

        @testset "custom oversampled grid size" begin
            a = Sequence(Taylor(2), [1.0, -2.0, 3.0])
            g16 = to_grid(a, 16) # 16 > fft_size(Taylor(2)) == 8, zero-padded
            @test length(g16) == 16
            b = to_seq(g16, Taylor(2))
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "resampling to a different space via to_seq(::Sequence, ::SequenceSpace)" begin
            a = Sequence(Taylor(1), [1.0, 2.0])
            # zero-padding to a higher order
            @test to_seq(a, Taylor(3)) == Sequence(Taylor(3), ComplexF64[1, 2, 0, 0])
            # truncating to a lower order
            @test to_seq(a, Taylor(0)) == Sequence(Taylor(0), ComplexF64[1])
        end
    end

    #

    @testset "Fourier" begin
        @testset "exact hand-computed round trip (order 1, N=4 uses only ±1,±i roots)" begin
            # a = c₋₁·e^{-iθ} + c₀ + c₁·e^{iθ} with (c₋₁,c₀,c₁) = (1,2,3).
            # Preprocessing circshifts the zero-frequency mode to position 1:
            # [c₋₁,c₀,c₁,0] -> [c₀,c₁,0,c₋₁] = [2,3,0,1], then the backward DFT gives
            #   Y0 = 2+3+0+1  = 6
            #   Y1 = 2+3i-0-i = 2+2i
            #   Y2 = 2-3+0-1  = -2
            #   Y3 = 2-3i-0+i = 2-2i
            a = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0])
            expected_grid = ComplexF64[6, 2+2im, -2, 2-2im]
            @test fft_size(space(a)) == (4,)
            @test to_grid(a) == expected_grid
            @test to_grid!(fill(complex(Inf, Inf), 4), a) == expected_grid
            seeded = Sequence(Fourier(1, 1.0), ComplexF64[Inf, Inf, Inf])
            @test to_seq!(seeded, copy(expected_grid)) == Sequence(Fourier(1, 1.0), ComplexF64[1, 2, 3]) == seeded
        end

        @testset "approximate round trip, higher order (N=8: irrational roots, roundoff genuine)" begin
            a = Sequence(Fourier(2, 1.0), [1.0, -0.5, 2.0, -0.5, 1.0])
            b = to_seq(to_grid(a), Fourier(2, 1.0))
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "complex coefficients" begin
            a = Sequence(Fourier(2, 1.0), ComplexF64[1.0-1.0im, -0.5, 2.0+0.5im, -0.5, 1.0])
            b = to_seq(to_grid(a), Fourier(2, 1.0))
            @test coefficients(b) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval coefficients (enclosure round trip)" begin
            a = Sequence(Fourier(2, 1.0), interval.([1.0, 2.0, 3.0, 4.0, 5.0]))
            b = to_seq(to_grid(a), Fourier(2, 1.0))
            for k ∈ -2:2
                @test issubset_interval(a[k], real(b[k]))
            end
        end

        @testset "order-0 (trivial 1-point grid, no bug)" begin
            s = Fourier(0, 1.0)
            a = Sequence(s, [7.0])
            @test to_grid(a) == ComplexF64[7]
            b = to_seq(to_grid(a), s)
            @test b == Sequence(s, ComplexF64[7])
            # an oversized grid also works, confirming the default grid size agrees
            b_ok = to_seq(to_grid(a, 2), s)
            @test b_ok == Sequence(s, ComplexF64[7])
            @test to_seq(_ -> 7.0, s) == Sequence(s, ComplexF64[7])
        end

        @testset "1-point grid with order(space) > 0 determines only the zero mode" begin
            # `to_grid` rejects a grid this small for order > 0 (dft dimension 5 > 1),
            # so this exercises `to_seq!` directly on a hand-built size-1 grid.
            s = Fourier(2, 1.0)
            c = Sequence(s, fill(complex(Inf, Inf), 5))
            to_seq!(c, ComplexF64[7.0])
            @test real(c[0]) ≈ 7.0
            for k ∈ (-2, -1, 1, 2)
                @test c[k] == 0
            end
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
            @test to_seq!(seeded, copy(expected_grid)) == Sequence(Chebyshev(1), ComplexF64[1, 2]) == seeded
        end

        @testset "approximate round trip, higher order (N=8: irrational roots, roundoff genuine)" begin
            a = Sequence(Chebyshev(3), [1.0, -2.0, 0.5, 3.0])
            b = to_seq(to_grid(a), Chebyshev(3))
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "order-0 (trivial 1-point grid, no bug)" begin
            a = Sequence(Chebyshev(0), [5.0])
            @test to_grid(a) == ComplexF64[5]
            @test to_seq(ComplexF64[5], Chebyshev(0)) == Sequence(Chebyshev(0), ComplexF64[5])
        end

        @testset "complex coefficients" begin
            a = Sequence(Chebyshev(3), ComplexF64[1.0+2.0im, -2.0, 0.5-1.0im, 3.0])
            b = to_seq(to_grid(a), Chebyshev(3))
            @test coefficients(b) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval coefficients (enclosure round trip)" begin
            a = Sequence(Chebyshev(3), interval.([1.0, 2.0, 3.0, 4.0]))
            b = to_seq(to_grid(a), Chebyshev(3))
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
            b = to_seq(g, s)
            @test b == Sequence(s, ComplexF64[7])
        end

        @testset "approximate round trip (mixed Taylor/Fourier, roundoff genuine)" begin
            s = Taylor(2) ⊗ Fourier(1, 1.0)
            a = Sequence(s, collect(1.0:dimension(s)))
            g = to_grid(a)
            @test size(g) == fft_size(s) == (8, 4)
            b = to_seq(g, s)
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "complex coefficients" begin
            s = Taylor(1) ⊗ Fourier(1, 1.0)
            a = Sequence(s, ComplexF64.(1.0:dimension(s)) .+ im .* (dimension(s):-1.0:1.0))
            b = to_seq(to_grid(a), s)
            @test coefficients(b) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval coefficients (enclosure round trip)" begin
            s = Taylor(1) ⊗ Fourier(1, 1.0)
            a = Sequence(s, interval.(collect(1.0:dimension(s))))
            b = to_seq(to_grid(a), s)
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
            to_seq!(c, to_grid(a))
            @test real.(coefficients(c)) ≈ coefficients(a) atol=1e-9
        end

        @testset "oddsym(Fourier)" begin
            s = oddsym(Fourier(2, 1.0))
            @test indices(s) == 1:2
            a = Sequence(s, [10.0, 20.0])
            c = Sequence(s, ComplexF64[Inf, Inf])
            to_seq!(c, to_grid(a))
            @test real.(coefficients(c)) ≈ coefficients(a) atol=1e-9
        end

        @testset "evensym(Chebyshev)" begin
            s = evensym(Chebyshev(4))
            @test indices(s) == 0:2:4
            a = Sequence(s, [1.0, 2.0, 3.0])
            c = Sequence(s, ComplexF64[Inf, Inf, Inf])
            to_seq!(c, to_grid(a))
            @test real.(coefficients(c)) ≈ coefficients(a) atol=1e-9
        end

        @testset "d4sym(Fourier ⊗ Fourier)" begin
            s = d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))
            @test fft_size(s) == (8, 8)
            a = Sequence(s, collect(1.0:dimension(s)))
            c = Sequence(s, fill(complex(Inf, Inf), dimension(s)))
            to_seq!(c, to_grid(a))
            @test real.(coefficients(c)) ≈ coefficients(a) atol=1e-9
        end
    end

    #

    @testset "function interpolation to_seq(f, space)" begin
        @testset "Taylor (exact, N=4)" begin
            # f(z) = z; nodes are z_j = e^{i2πj/4} ∈ {1,i,-1,-i}, exactly interpolated
            b = to_seq(z -> z, Taylor(1))
            @test b == Sequence(Taylor(1), ComplexF64[0, 1])
        end

        @testset "Fourier (approximate: cos/sin of irrational nodes)" begin
            # 2cos(x) = e^{ix}+e^{-ix} on Fourier(1,1.0): c₋₁=1, c₀=0, c₁=1
            b = to_seq(x -> 2*cos(x), Fourier(1, 1.0))
            @test real.(coefficients(b)) ≈ [1.0, 0.0, 1.0] atol=1e-9
            # frequency parameter is honoured by the node formula 2π/frequency·j/N
            b2 = to_seq(x -> 2*cos(2x), Fourier(1, 2.0))
            @test real.(coefficients(b2)) ≈ [1.0, 0.0, 1.0] atol=1e-9
        end

        @testset "Chebyshev (note: interior modes carry an implicit factor 2)" begin
            # RadiiPolynomial's Chebyshev sequences satisfy f(x) = c0·T0(x) + 2·Σ_{k≥1} ck·Tk(x),
            # matching the halving of the Nyquist mode in `_postprocess_to_seq!`.
            # Hence interpolating f(x) = x = T1(x) gives c1 = 0.5, not 1.
            b1 = to_seq(x -> x, Chebyshev(1))
            @test real.(coefficients(b1)) ≈ [0.0, 0.5] atol=1e-9
            # f(x) = 2x²-1 = T2(x) gives c2 = 0.5
            b2 = to_seq(x -> 2x^2 - 1, Chebyshev(2))
            @test real.(coefficients(b2)) ≈ [0.0, 0.0, 0.5] atol=1e-9
        end

        @testset "TensorSpace (exact, both factors order 1)" begin
            # f(x,y) = xy on Taylor(1) ⊗ Taylor(1): only the (1,1) mode is nonzero
            s = Taylor(1) ⊗ Taylor(1)
            b = to_seq((x, y) -> x*y, s)
            @test b == Sequence(s, ComplexF64[0, 0, 0, 1])
        end
    end

    #

    @testset "grid <-> array entry points" begin
        a = Sequence(Taylor(1), [1.0, 2.0])
        grid = to_grid(a)
        grid_copy = copy(grid)

        @testset "to_seq(::AbstractArray, ::SequenceSpace) does not mutate its input" begin
            b = to_seq(grid, Taylor(1))
            @test grid == grid_copy
            @test b == Sequence(Taylor(1), ComplexF64[1, 2])
        end

        @testset "to_seq!(::AbstractArray, ::SequenceSpace) mutates its input in-place" begin
            g = copy(grid)
            b = to_seq!(g, Taylor(1))
            @test g != grid_copy # destroyed by the in-place forward FFT
            @test b == Sequence(Taylor(1), ComplexF64[1, 2])
        end

        @testset "to_seq!(::Sequence, ::AbstractArray) mutates the sequence, not the space" begin
            c = Sequence(Taylor(1), ComplexF64[Inf, Inf])
            g = copy(grid)
            out = to_seq!(c, g)
            @test out === c
            @test c == Sequence(Taylor(1), ComplexF64[1, 2])
        end
    end

    #

    @testset "_call_to_seq! internal helper" begin
        a = Sequence(Taylor(1), [1.0, 2.0])
        grid = to_grid(a)

        @test RadiiPolynomial._call_to_seq!(copy(grid), Taylor(1), Float64) == real(to_seq(grid, Taylor(1)))
        @test RadiiPolynomial._call_to_seq!(copy(grid), Taylor(1), Float64) isa Sequence{Taylor,Vector{Float64}}
        @test RadiiPolynomial._call_to_seq!(copy(grid), Taylor(1), ComplexF64) == to_seq(grid, Taylor(1))
        @test RadiiPolynomial._call_to_seq!(copy(grid), Taylor(1), ComplexF64) isa Sequence{Taylor,Vector{ComplexF64}}
    end

    #

    @testset "twiddle tables and the a priori error bound" begin
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

        # `:apriori_bound` must enclose the exact transform, not merely look plausible
        n = 32
        x = round.([ComplexF64(cospi(2k/n), sinpi(3k/n)) for k ∈ 1:n], digits=6)
        exact = setprecision(384) do
            [sum(Complex{BigFloat}(x[j+1]) * cispi(-2*BigFloat(j*k)/n) for j ∈ 0:n-1) for k ∈ 0:n-1]
        end
        for algo ∈ (:interval, :apriori_bound)
            set_fft_algorithm(algo)
            y = RadiiPolynomial._fft_pow2!([complex(interval(real(z)), interval(imag(z))) for z ∈ x])
            @test all(k -> in_interval(Float64(real(exact[k])), real(y[k])) &&
                           in_interval(Float64(imag(exact[k])), imag(y[k])), 1:n)
            # the same holds in `BigFloat`, whose tables are built for the precision in use
            setprecision(128) do
                z = RadiiPolynomial._fft_pow2!([complex(interval(BigFloat, real(w)), interval(BigFloat, imag(w))) for w ∈ x])
                @test all(k -> in_interval(BigFloat(real(exact[k])), real(z[k])) &&
                               in_interval(BigFloat(imag(exact[k])), imag(z[k])), 1:n)
                @test precision(inf(real(z[1]))) == 128
            end
        end
        set_fft_algorithm(:interval)
    end

    @testset "grid_size" begin
        @test grid_size(Taylor(2)) == fft_size(Taylor(2)) == (8,)
        @test grid_size(Fourier(2, 1.0)) == fft_size(Fourier(2, 1.0)) == (8,)
        # Chebyshev: half grid of Chebyshev–Lobatto nodes cos(π(k-1)/(m-1))
        @test grid_size(Chebyshev(2)) == (3,) # fft_size 4
        @test grid_size(Chebyshev(4)) == (5,) # fft_size 8
        @test grid_size(Chebyshev(0)) == (1,)
        @test grid_size(Chebyshev(2) ⊗ Fourier(1, 1.0)) == (3, 4)
        @test grid_size(evensym(Chebyshev(4))) == grid_size(Chebyshev(4)) == (5,)
    end

    #

    @testset "Chebyshev half grid (Chebyshev–Lobatto nodes)" begin
        @testset "2^k+1 grid points fold onto the half grid; to_seq accepts it" begin
            a = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
            full = to_grid(a) # size 4
            half = to_grid(a, grid_size(space(a))) # (3,): nodes 1, 0, -1
            @test half == full[1:3]
            b = to_seq(half, Chebyshev(2))
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-12
            # the full mirrored grid is still accepted
            @test coefficients(to_seq(full, Chebyshev(2))) ≈ coefficients(b) atol=1e-12
        end

        @testset "nodes are ordered from x = 1 down to x = -1" begin
            # f(x) = x sampled with the implicit factor 2 convention: c1 = 0.5
            m = only(grid_size(Chebyshev(2))) # 3
            g = [cospi((k-1)/(m-1)) for k ∈ 1:m] # 1, 0, -1
            b = to_seq(g, Chebyshev(2))
            @test real.(coefficients(b)) ≈ [0.0, 0.5, 0.0] atol=1e-12
        end

        @testset "tensor space: folds along every Chebyshev axis" begin
            s = Chebyshev(2) ⊗ Chebyshev(1)
            a = Sequence(s, collect(1.0:dimension(s)))
            full = to_grid(a) # (4, 2)
            half = to_grid(a, grid_size(s)) # (3, 2)
            @test half == full[1:3, 1:2]
            b = to_seq(half, s)
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "mixed tensor space: non-Chebyshev axes keep the full grid" begin
            s = Chebyshev(2) ⊗ Fourier(1, 1.0)
            a = Sequence(s, collect(1.0:dimension(s)))
            half = to_grid(a, grid_size(s))
            @test size(half) == grid_size(s) == (3, 4)
            b = to_seq(half, s)
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval enclosure round trip through the half grid" begin
            a = Sequence(Chebyshev(3), interval.([1.0, 2.0, 3.0, 4.0]))
            b = to_seq(to_grid(a, grid_size(space(a))), Chebyshev(3))
            for k ∈ 0:3
                @test issubset_interval(a[k], real(b[k]))
            end
        end

        @testset "invalid grid sizes are rejected" begin
            # 6 is neither a power of 2 (full grid) nor of the form 2^k+1 (half grid)
            @test_throws ArgumentError to_seq(zeros(6), Chebyshev(2))
            @test_throws DimensionMismatch to_grid(Sequence(Chebyshev(2), [1.0, 2.0, 3.0]), (6,))
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
            b = to_seq(x_grid, s_par)
            @test space(b) == fs
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "Fourier and Taylor parameters" begin
            s7 = Fourier(1, 1.0)
            a7 = Sequence(s7 ⊗ Taylor(1), collect(1.0:dimension(s7 ⊗ Taylor(1))))
            b7 = to_seq(to_grid(a7, grid_size(s7)), s7)
            @test coefficients(b7) ≈ coefficients(a7) atol=1e-9

            s8 = Taylor(1)
            a8 = Sequence(s8 ⊗ Chebyshev(1), collect(1.0:dimension(s8 ⊗ Chebyshev(1))))
            b8 = to_seq(to_grid(a8, grid_size(s8)), s8)
            @test coefficients(b8) ≈ coefficients(a8) atol=1e-9
        end

        @testset "two parameter axes give a Matrix grid" begin
            s2 = Chebyshev(1) ⊗ Chebyshev(2)
            full_space = s2 ⊗ Taylor(1)
            a = Sequence(full_space, collect(1.0:dimension(full_space)))
            g = to_grid(a, grid_size(s2))
            @test g isa Matrix
            @test size(g) == grid_size(s2) == (2, 3)
            b = to_seq(g, s2)
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
            b = to_seq(x_grid, s_par)
            @test space(b) == fs
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
        end

        @testset "interval enclosure round trip" begin
            fs = Chebyshev(2) ⊗ Fourier(1, 1.0)
            a = Sequence(fs, interval.(collect(1.0:dimension(fs))))
            b = to_seq(to_grid(a, (3,)), Chebyshev(2))
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
            b = to_seq(fine, s_par)
            @test space(b) == space(a)
            @test real.(coefficients(b)) ≈ coefficients(a) atol=1e-9
            # interpolating at the finer order: the extra coefficients vanish
            b8 = to_seq(fine, Chebyshev(8))
            @test real.(coefficients(project(b8, space(a)))) ≈ coefficients(a) atol=1e-9

            # full-space scalar resampling
            c = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
            g = to_grid(c, (5,))
            @test size(g) == (5,)
            @test real.(coefficients(project(to_seq(g, Chebyshev(4)), Chebyshev(2)))) ≈ coefficients(c) atol=1e-9

            # a Fourier parameter resamples too (power-of-2 grid)
            af = Sequence(Fourier(1, 1.0) ⊗ Taylor(1), collect(1.0:6))
            gf = to_grid(af, (8,))
            @test size(gf) == (8,)
            @test coefficients(to_seq(gf, Fourier(1, 1.0))) ≈ coefficients(af) atol=1e-9
            @test_throws DimensionMismatch to_grid(af, (6,)) # 6 is not a valid grid size

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
            b = to_seq(x_grid, s_par)
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
            b = to_seq(x_grid, s_par)
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
            B_concrete = to_seq(ops, s_par2)
            A_grid = Matrix{LinearOperator}(undef, size(ops)...) # abstract eltype
            copyto!(A_grid, ops)
            B_abstract = to_seq(A_grid, s_par2)
            @test domain(B_abstract) == domain(B_concrete)
            @test codomain(B_abstract) == codomain(B_concrete)
            @test coefficients(B_abstract) == coefficients(B_concrete)
        end

        @testset "symmetric inner space: interval enclosure round trip" begin
            s_par = Chebyshev(1)
            prod_sym = s_par ⊗ evensym(Fourier(1, 1.0))
            a = project(Sequence(s_par ⊗ Fourier(1, 1.0), interval.(collect(1.0:6))), interval(prod_sym))
            b = to_seq(to_grid(a, grid_size(s_par)), s_par)
            for n ∈ eachindex(coefficients(a))
                @test issubset_interval(coefficients(a)[n], real(coefficients(b)[n]))
            end
        end

        @testset "error paths" begin
            # the grid must have at most one axis per factor, and at least one
            @test_throws ArgumentError to_grid(Sequence(Fourier(1, 1.0), ones(3)), (4, 4))
            @test_throws ArgumentError to_grid(Sequence(Fourier(1, 1.0) ⊗ Taylor(1), ones(6)), ())
            # grid elements on mismatched spaces
            @test_throws ArgumentError to_seq([Sequence(Taylor(1), [1.0, 2.0]), Sequence(Taylor(2), [1.0, 2.0, 3.0])], Chebyshev(1))
            # grid dimension must match the number of factors of `s`
            @test_throws ArgumentError to_seq(fill(Sequence(Taylor(1), [1.0, 2.0]), 3, 3), Chebyshev(1))
            # the symmetry group must not mix the grid and inner directions
            d4 = d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0)) # swaps the two factors
            @test_throws ArgumentError to_grid(Sequence(d4, ones(dimension(d4))), (4,))
        end
    end

    #

    @testset "unsupported vector spaces" begin
        # `to_grid`/`to_seq` are only defined for `Sequence{<:SequenceSpace}`;
        # `ScalarSpace` and `CartesianSpace` are not `SequenceSpace`s.
        @test_throws MethodError to_grid(Sequence(ScalarSpace(), [1.0]))
        @test_throws MethodError to_grid(Sequence(Taylor(2) × Taylor(2), ones(6)))
    end

    #

    @testset "error paths" begin
        @testset "offset arrays are rejected" begin
            a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
            @test_throws ArgumentError to_grid!(_OffsetVec(zeros(ComplexF64, 8)), a)

            c = Sequence(Taylor(2), zeros(ComplexF64, 3))
            @test_throws ArgumentError to_seq!(c, _OffsetVec(to_grid(a)))
        end

        @testset "to_seq! requires all grid sizes to be a power of 2" begin
            c = Sequence(Taylor(2), zeros(ComplexF64, 3))
            @test_throws ArgumentError to_seq!(c, zeros(ComplexF64, 6))
        end

        @testset "to_grid! with an incompatible grid size throws DimensionMismatch" begin
            a = Sequence(Taylor(2), [1.0, 2.0, 3.0]) # dft dimension 5, needs n ≥ 5
            @test_throws DimensionMismatch to_grid!(fill(complex(Inf, Inf), 2), a) # n = 2 < 5: incompatible size
        end
    end
end
