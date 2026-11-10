# a vector type with non-1-based axes
struct _FFTOffsetVec{T} <: AbstractVector{T}
    data :: Vector{T}
end
Base.size(v::_FFTOffsetVec) = size(v.data)
Base.axes(v::_FFTOffsetVec) = (2:length(v.data)+1,)
Base.getindex(v::_FFTOffsetVec, i::Int) = v.data[i-1]
Base.setindex!(v::_FFTOffsetVec, x, i::Int) = (v.data[i-1] = x)

# `b` recovers `a`: up to roundoff for floats, as an enclosure for intervals
function _fft_test_recovers(a, b; atol = 1e-9)
    x, y = coefficients(a), coefficients(b)
    if eltype(x) <: Union{Interval,Complex{<:Interval}}
        @test all(issubset_interval.(real.(x), real.(y))) && all(issubset_interval.(imag.(x), imag.(y)))
    else
        @test y ≈ x atol=atol
    end
end

# sampling `a` along the factors `s_par` gives an array of Sequences on `inner`, which interpolates back to `a`
function _fft_test_family(a, s_par, inner, m = grid_size(s_par))
    x_grid = to_grid(a, m)
    @test x_grid isa Array{<:Sequence,length(m)}
    @test size(x_grid) == m
    @test all(x -> space(x) == inner, x_grid)
    b = to_coef(x_grid, s_par)
    @test space(b) == space(a)
    _fft_test_recovers(a, b)
    return x_grid
end

# the elements of a grid along a Chebyshev factor are the partial evaluations at the Chebyshev-Lobatto nodes
function _fft_test_nodes(x_grid, a)
    for k ∈ eachindex(x_grid)
        x_k = cospi((k-1)/(length(x_grid)-1))
        @test coefficients(x_grid[k]) ≈ coefficients(Evaluation(x_k, nothing) * a) atol=1e-9
    end
end

@testset "FFT" begin
    @testset "fft_size and grid_size" begin
        F = Fourier(2, 1.0)
        # (space, fft_size, grid_size): a Chebyshev transform has length 2order, its mirror folding onto order+1 nodes
        for (s, n, m) ∈ (
                (Taylor(0), (1,), (1,)), (Taylor(1), (2,), (2,)), (Taylor(4), (5,), (5,)), (Taylor(5), (6,), (6,)),
                (Fourier(0, 1.0), (1,), (1,)), (Fourier(1, 1.0), (3,), (3,)), (Fourier(3, 1.0), (7,), (7,)), (Fourier(2, 3.5), (5,), (5,)),
                (Chebyshev(0), (1,), (1,)), (Chebyshev(1), (2,), (2,)), (Chebyshev(4), (8,), (5,)), (Chebyshev(5), (10,), (6,)),
                (Taylor(2) ⊗ Fourier(1, 1.0), (3, 3), (3, 3)), (Taylor(1) ⊗ Taylor(1) ⊗ Chebyshev(3), (2, 2, 6), (2, 2, 4)),
                (Taylor(0) ⊗ Chebyshev(0), (1, 1), (1, 1)), (Chebyshev(2) ⊗ Fourier(1, 1.0), (4, 3), (3, 3)),
                # a symmetric space defers to the desymmetrized space
                (evensym(Taylor(3)), (4,), (4,)), (oddsym(Fourier(3, 1.0)), (7,), (7,)), (evensym(Chebyshev(3)), (6,), (4,)),
                (d4sym(F ⊗ F), (5, 5), (5, 5)), (evensym(F)^2, (5,), (5,)),
                # the components of a cartesian space share the nodes of the finest one
                (Chebyshev(2)^3, (4,), (3,)), (Chebyshev(2) × Chebyshev(4), (8,), (5,)), ((Taylor(1)^2 × Taylor(3))^2, (4,), (4,)),
                ((Chebyshev(2) ⊗ Fourier(1, 1.0)) × (Chebyshev(4) ⊗ F), (8, 5), (5, 5)),
            )
            @test fft_size(s) == n
            @test grid_size(s) == m
        end
    end

    @testset "hand-computed grid on $s" for (s, c, n, grid) ∈ (
            # 1 + 2z at z = 1, -1
            (Taylor(1), [1.0, 2.0], 2, ComplexF64[3, -1]),
            # e^{-iθ} + 2 + 3e^{iθ} at θ = 0, π/2, π, 3π/2
            (Fourier(1, 1.0), [1.0, 2.0, 3.0], 4, ComplexF64[6, 2+2im, -2, 2-2im]),
            # 1 + 2(2T₁(x)) at x = 1, -1, the modes k ≥ 1 carrying an implicit factor 2
            (Chebyshev(1), [1.0, 2.0], 2, ComplexF64[5, -3]),
        )
        a = Sequence(s, c)
        @test to_grid(a, n) == grid
        @test to_grid!(fill(complex(Inf, Inf), n), a) == grid
        @test to_coef(grid, s) == a
        seeded = fill(complex(Inf, Inf), s)
        @test to_coef!(seeded, copy(grid)) === seeded
        @test seeded == a
    end

    @testset "round trip on $s" for s ∈ (Taylor(3), Fourier(2, 1.0), Chebyshev(3), Taylor(2) ⊗ Fourier(1, 1.0),
            Chebyshev(2) ⊗ Chebyshev(1), Chebyshev(2) ⊗ Fourier(1, 1.0))
        x = collect(1.0:dimension(s))
        for c ∈ (x, complex.(x, reverse(x)), interval.(x), complex.(interval.(x), interval.(reverse(x))))
            a = Sequence(s, c)
            g = to_grid(a)
            @test size(g) == grid_size(s)
            _fft_test_recovers(a, to_coef(g, s))
        end
    end

    @testset "round trip at every order, including sizes with a large prime factor" begin
        for s ∈ [Taylor.([0:20; 36; 66; 100]); Fourier.([0:20; 33; 50; 53], 1.0); Chebyshev.([0:20; 37; 67; 101])]
            a = Sequence(s, collect(1.0:dimension(s)))
            _fft_test_recovers(a, to_coef(to_grid(a), s); atol = 1e-10)
        end
    end

    @testset "order 0 is exact on $s" for s ∈ (Taylor(0), Fourier(0, 1.0), Chebyshev(0), Taylor(0) ⊗ Chebyshev(0))
        a = Sequence(s, [7.0])
        @test to_grid(a) == fill(complex(7.0), grid_size(s))
        @test to_coef(to_grid(a), s) == a
    end

    @testset "grids finer or coarser than the space" begin
        for (s, ms) ∈ ((Taylor(2), (3, 5, 6, 10, 16)), (Chebyshev(2), 3:8)), m ∈ ms
            a = Sequence(s, [1.0, -2.0, 3.0])
            g = to_grid(a, m)
            @test size(g) == (m,)
            _fft_test_recovers(a, to_coef(g, s); atol = 1e-12)
        end
        @test to_grid(Sequence(Taylor(1), [1.0, 2.0]), 4) == ComplexF64[3, 1+2im, -1, 1-2im]
        @test to_coef(to_grid(Sequence(Fourier(0, 1.0), [7.0]), 2), Fourier(0, 1.0)) == Sequence(Fourier(0, 1.0), [7.0])

        # a grid too coarse to hold the transform is rejected
        @test_throws DimensionMismatch to_grid(Sequence(Chebyshev(2), [1.0, 2.0, 3.0]), (2,))
        @test_throws DimensionMismatch to_grid!(fill(complex(Inf, Inf), 2), Sequence(Taylor(2), [1.0, 2.0, 3.0]))

        # a coarse grid determines the lowest modes only, the others vanishing
        s = Fourier(2, 1.0)
        c = fill(complex(Inf, Inf), s)
        to_coef!(c, ComplexF64[7.0])
        @test real(c[0]) ≈ 7.0
        @test all(k -> c[k] == 0, (-2, -1, 1, 2))
        g = ComplexF64[1, 2, 3]
        b = to_coef(g, s)
        @test all(k -> b[k] == 0, (-2, 2))
        @test all(j -> b(2π*j/3) ≈ g[j+1], 0:2)
        # f(x) = x through the nodes 1, -1
        @test coefficients(to_coef(ComplexF64[1, -1], Chebyshev(2))) == ComplexF64[0, 0.5, 0]

        c = zeros(ComplexF64, Taylor(2))
        @test to_coef!(c, zeros(ComplexF64, 6)) === c
        @test to_coef!(c, zeros(ComplexF64, 2)) === c
        C = zeros(ComplexF64, 4)
        @test to_grid!(C, Sequence(Taylor(1), [1.0, 2.0])) === C
    end

    @testset "resampling via to_coef(::Sequence, ::SequenceSpace)" begin
        a = Sequence(Taylor(1), [1.0, 2.0])
        @test to_coef(a, Taylor(3)) == Sequence(Taylor(3), ComplexF64[1, 2, 0, 0])
        @test to_coef(a, Taylor(0)) == Sequence(Taylor(0), ComplexF64[1])
        a = Sequence(Chebyshev(2)^3, collect(1.0:9))
        b = to_coef(a, Chebyshev(4))
        @test space(b) == Chebyshev(4)^3
        _fft_test_recovers(a, project(b, space(a)))
    end

    @testset "SymmetricSpace $s" for s ∈ (evensym(Taylor(3)), oddsym(Fourier(2, 1.0)), evensym(Chebyshev(4)),
            d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0)))
        a = Sequence(s, collect(1.0:dimension(s)))
        @test to_grid(a) == to_grid(Projection(desymmetrize(s)) * a)
        _fft_test_recovers(a, to_coef!(fill(complex(Inf, Inf), s), to_grid(a)))
    end

    @testset "function interpolation to_coef(f, space)" begin
        @test to_coef(z -> z, Taylor(1)) == Sequence(Taylor(1), ComplexF64[0, 1])
        @test to_coef((x, y) -> x*y, Taylor(1) ⊗ Taylor(1)) == Sequence(Taylor(1) ⊗ Taylor(1), ComplexF64[0, 0, 0, 1])
        @test to_coef(_ -> 7.0, Fourier(0, 1.0)) == Sequence(Fourier(0, 1.0), ComplexF64[7])
        # 2cos(ωx) = e^{-iωx} + e^{iωx}, the nodes following the frequency
        @test coefficients(to_coef(x -> 2*cos(x), Fourier(1, 1.0))) ≈ [1.0, 0.0, 1.0] atol=1e-9
        @test coefficients(to_coef(x -> 2*cos(2x), Fourier(1, 2.0))) ≈ [1.0, 0.0, 1.0] atol=1e-9
        # f(x) = c₀ + 2Σ cₖTₖ(x): T₁ and T₂ have coefficient 0.5
        @test coefficients(to_coef(x -> x, Chebyshev(1))) ≈ [0.0, 0.5] atol=1e-9
        @test coefficients(to_coef(x -> 2x^2 - 1, Chebyshev(2))) ≈ [0.0, 0.0, 0.5] atol=1e-9
        # nodes are ordered from x = 1 down to x = -1
        @test coefficients(to_coef([1.0, 0.0, -1.0], Chebyshev(2))) ≈ [0.0, 0.5, 0.0] atol=1e-12

        # the interpolant goes through the nodes
        f(x) = 1/(2-x)
        g(x) = 1/(2-cos(x))
        for n ∈ 1:12
            b, m = to_coef(f, Chebyshev(n)), only(grid_size(Chebyshev(n)))
            @test all(k -> b(cospi((k-1)/(m-1))) ≈ f(cospi((k-1)/(m-1))), 1:m)
            b, m = to_coef(f, Taylor(n)), only(grid_size(Taylor(n)))
            @test all(j -> b(cispi(2j/m)) ≈ f(cispi(2j/m)), 0:m-1)
            b, m = to_coef(g, Fourier(n, 1.0)), only(grid_size(Fourier(n, 1.0)))
            @test all(j -> b(2π*j/m) ≈ g(2π*j/m), 0:m-1)
        end
    end

    @testset "to_coef leaves the grid untouched, to_coef! transforms it in place" begin
        expected = Sequence(Taylor(1), ComplexF64[1, 2])
        grid = to_grid(Sequence(Taylor(1), [1.0, 2.0]))
        g = copy(grid)
        @test to_coef(g, Taylor(1)) == expected
        @test g == grid
        @test to_coef!(fill(complex(Inf, Inf), Taylor(1)), g) == expected
        @test g != grid

        for T ∈ (Float64, ComplexF64)
            c = RadiiPolynomial._call_to_coef!(copy(grid), Taylor(1), T)
            @test c isa Sequence{Taylor,Vector{T}}
            @test c == expected
        end
    end

    @testset "twiddle tables" begin
        # the radius bounds the modulus of the error between the stored midpoint and the exact root
        # of unity, which the radius of a complex interval underestimates by up to √2
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

        # radix-2 (16), decimation (12 = 2·6, 105 = 3·5·7), a direct sum (13) and Bluestein's algorithm (67)
        @testset "forward transform of length $n" for n ∈ (1, 2, 3, 5, 12, 13, 16, 67, 105)
            x = [ComplexF64(cospi(2k/n), sinpi(3k/(n+1))) for k ∈ 0:n-1]
            @test RadiiPolynomial._fft!(copy(x)) ≈ naive_dft(x) rtol=1e-12
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

    @testset "Chebyshev grids hold the nodes only, the mirror staying internal" begin
        a = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
        @test to_grid(a) == to_grid(a, grid_size(space(a))) == RadiiPolynomial._to_grid!(zeros(ComplexF64, 4), a)[1:3]
        s = Chebyshev(2) ⊗ Chebyshev(1)
        b = Sequence(s, collect(1.0:dimension(s)))
        @test to_grid(b) == to_grid(b, grid_size(s)) == RadiiPolynomial._to_grid!(zeros(ComplexF64, 4, 2), b)[1:3, 1:2]
    end

    @testset "grids of Sequences" begin
        F, Fₑ = Fourier(1, 1.0), evensym(Fourier(2, 1.0))
        𝒞₁, 𝒞₂ = Chebyshev(1), Chebyshev(2)
        seq(s) = Sequence(s, collect(1.0:dimension(s)))

        @testset "sampling $s along $s_par" for (s, s_par, inner) ∈ (
                (𝒞₂ ⊗ F, 𝒞₂, F),
                (F ⊗ Taylor(1), F, Taylor(1)),
                (Taylor(1) ⊗ 𝒞₁, Taylor(1), 𝒞₁),
                (𝒞₁ ⊗ 𝒞₂ ⊗ Taylor(1), 𝒞₁ ⊗ 𝒞₂, Taylor(1)),
                (𝒞₁ ⊗ Taylor(1) ⊗ F, 𝒞₁, Taylor(1) ⊗ F),
                # the components of a cartesian space are sampled together
                (𝒞₂^3, 𝒞₂, ScalarSpace()^3),
                ((𝒞₂ ⊗ F) × (𝒞₂ ⊗ Taylor(1)), 𝒞₂, F × Taylor(1)),
                ((𝒞₁ ⊗ F)^2 × 𝒞₁, 𝒞₁, F^2 × ScalarSpace()),
            )
            _fft_test_family(seq(s), s_par, inner)
            _fft_test_family(interval.(seq(s)), s_par, inner)
        end

        @testset "a symmetric inner space keeps its symmetry" begin
            full = 𝒞₂ ⊗ Fourier(2, 1.0)
            a = project(seq(full), 𝒞₂ ⊗ Fₑ)
            x_grid = _fft_test_family(a, 𝒞₂, Fₑ)
            _fft_test_nodes(map(x -> Projection(Fourier(2, 1.0)) * x, x_grid), Projection(full) * a)
            _fft_test_family(project(seq(full^2), (𝒞₂ ⊗ Fₑ)^2), 𝒞₂, Fₑ^2)
            _fft_test_family(project(interval.(seq(𝒞₁ ⊗ Fourier(2, 1.0))), interval(𝒞₁ ⊗ Fₑ)), 𝒞₁, Fₑ)
        end

        @testset "Chebyshev-Lobatto nodes" begin
            a = seq(𝒞₂ ⊗ F)
            _fft_test_nodes(to_grid(a, (3,)), a)
            # a finer grid samples the same family
            fine = _fft_test_family(a, 𝒞₂, F, (9,))
            _fft_test_nodes(fine, a)
            _fft_test_recovers(a, project(to_coef(fine, Chebyshev(8)), space(a)))
            b = seq((𝒞₂ ⊗ F) × (𝒞₂ ⊗ Taylor(1)))
            _fft_test_nodes(to_grid(b, (3,)), b)

            # each component is sampled as it would be on its own
            c = seq(𝒞₂^3)
            @test all(i -> [coefficients(x)[i] for x ∈ to_grid(c)] == to_grid(component(c, i)), 1:3)
        end

        @testset "grids of any size" begin
            c = Sequence(𝒞₂, [1.0, 2.0, 3.0])
            _fft_test_recovers(c, project(to_coef(to_grid(c, (5,)), Chebyshev(4)), 𝒞₂))
            for m ∈ (3, 6, 8)
                _fft_test_family(seq(F ⊗ Taylor(1)), F, Taylor(1), (m,))
            end
            # the finest component sets the size, the others being oversampled and interpolated onto its space
            a = seq(𝒞₂ × Chebyshev(4))
            x_grid = to_grid(a)
            @test size(x_grid) == (5,)
            b = to_coef(x_grid, Chebyshev(4))
            @test space(b) == Chebyshev(4)^2
            _fft_test_recovers(a, project(b, space(a)))
        end

        @testset "in-place entry points" begin
            fs = (𝒞₂ ⊗ F) × 𝒞₂
            a = seq(fs)
            ref = to_grid(a, (3,))
            x_grid = [fill(complex(Inf, Inf), F × ScalarSpace()) for _ ∈ 1:3]
            @test to_grid!(x_grid, a) === x_grid
            @test all(i -> coefficients(x_grid[i]) ≈ coefficients(ref[i]), eachindex(x_grid))
            c = fill(complex(Inf, Inf), fs)
            @test to_coef!(c, ref) === c
            _fft_test_recovers(a, c)
        end

        @testset "abstractly typed grid containers" begin
            # the element type of a `Vector{Sequence}` says nothing of the coefficient type
            a = seq(𝒞₂ ⊗ F)
            ref = to_grid(a, (3,))
            b = to_coef(copyto!(Vector{Sequence}(undef, 3), ref), 𝒞₂)
            @test space(b) == 𝒞₂ ⊗ F
            _fft_test_recovers(a, b; atol = 1e-12)

            buf = Vector{Sequence}(undef, 3)
            for i ∈ eachindex(buf)
                buf[i] = zeros(ComplexF64, F)
            end
            to_grid!(buf, a)
            @test all(i -> coefficients(buf[i]) ≈ coefficients(ref[i]), eachindex(buf))

            ops = [LinearOperator(F, F, reshape(1.0:9.0, 3, 3) .+ sum(Tuple(k))) for k ∈ CartesianIndices(grid_size(𝒞₁ ⊗ 𝒞₁))]
            B_concrete = to_coef(ops, 𝒞₁ ⊗ 𝒞₁)
            B_abstract = to_coef(copyto!(Matrix{LinearOperator}(undef, size(ops)...), ops), 𝒞₁ ⊗ 𝒞₁)
            @test domain(B_abstract) == domain(B_concrete)
            @test codomain(B_abstract) == codomain(B_concrete)
            @test coefficients(B_abstract) == coefficients(B_concrete)
        end

        @testset "error paths" begin
            # the grid has at least one axis and at most one per factor
            @test_throws ArgumentError to_grid(Sequence(F, ones(3)), (4, 4))
            @test_throws ArgumentError to_grid(Sequence(F ⊗ Taylor(1), ones(6)), ())
            @test_throws ArgumentError to_coef(fill(Sequence(Taylor(1), [1.0, 2.0]), 3, 3), 𝒞₁)
            # grid elements on mismatched spaces
            @test_throws ArgumentError to_coef([Sequence(Taylor(1), [1.0, 2.0]), Sequence(Taylor(2), [1.0, 2.0, 3.0])], 𝒞₁)
            @test_throws ArgumentError to_grid!(to_grid(Sequence(𝒞₂^3, ones(9))), Sequence(𝒞₂^2, ones(6)))
            # the symmetry group must not mix the grid and inner directions
            d4 = d4sym(F ⊗ F)
            @test_throws ArgumentError to_grid(Sequence(d4, ones(dimension(d4))), (4,))
            # a grid too coarse for the family
            @test_throws DimensionMismatch to_grid(Sequence(F ⊗ Taylor(1), ones(6)), (2,))
            @test_throws DimensionMismatch to_grid(Sequence(Chebyshev(8) ⊗ Taylor(1), ones(18)), (2,))
            @test_throws DimensionMismatch to_grid(Sequence(𝒞₂^2, ones(6)), (2,))
            # the components of a cartesian space must be sampled on the same nodes
            @test_throws MethodError to_grid(Sequence(𝒞₂ × F, ones(6)))
            @test_throws ArgumentError to_grid(Sequence(F × Fourier(1, 2.0), ones(6)), (3,))
            @test_throws MethodError to_grid(Sequence(𝒞₂ × (𝒞₂ ⊗ Taylor(1)), ones(9)))
            # nothing to discretize
            @test_throws MethodError to_grid(Sequence(𝒞₂ × ScalarSpace(), ones(4)))
            @test_throws MethodError to_grid(Sequence(𝒞₂ × ScalarSpace(), ones(4)), (3,))
            @test_throws MethodError to_grid(Sequence(ScalarSpace(), [1.0]))
            @test_throws MethodError to_grid(Sequence(ScalarSpace()^2, [1.0, 2.0]))
        end
    end

    @testset "offset arrays are rejected" begin
        a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        @test_throws ArgumentError to_grid!(_FFTOffsetVec(zeros(ComplexF64, 8)), a)
        @test_throws ArgumentError to_coef!(zeros(ComplexF64, Taylor(2)), _FFTOffsetVec(to_grid(a)))
    end
end
