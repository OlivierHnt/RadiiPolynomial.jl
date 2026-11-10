# The action of an operator agrees across its entry points: the functor, `mul!`, the materialized
# matrix and the in-place method `inplace!(c, a)` when given. Every output starts from `Inf`
# coefficients so that a coefficient left unwritten is caught.
function test_action(𝒪, a, expected; inplace! = nothing, cmp = ==)
    codom, T = space(expected), eltype(expected)
    @testset let 𝒪 = 𝒪, dom = space(a)
        @test cmp(𝒪(a), expected)
        @test cmp(mul!(fill(T(Inf), codom), 𝒪, a), expected)
        @test cmp(project(𝒪, dom, codom, T)(a), expected)
        isnothing(inplace!) || @test cmp(inplace!(fill(T(Inf), codom), a), expected)
    end
    return nothing
end

# constructors, accessor and guards of an operator carrying one value per dimension
function test_valued_operator(𝒪)
    @test 𝒪(2.0) isa 𝒪{Float64}
    @test value(𝒪(2.0)) == 2.0
    @test 𝒪(2.0, 3.0) == 𝒪((2.0, 3.0))
    @test value(𝒪(2.0, 3.0)) == (2.0, 3.0)
    @test_throws ArgumentError 𝒪(())
    @test_throws ArgumentError 𝒪()
    return nothing
end

# constructors, order accessor and algebra shared by `Derivative` and `Integral`
function _calculus_test_algebra(𝒪)
    @test order(𝒪(1)) == 1
    @test order(𝒪(1, 2)) == (1, 2) == order(𝒪((1, 2)))
    @test 𝒪(1, 2) == 𝒪((1, 2))
    @test 𝒪(1) * 𝒪(1) == 𝒪(1)^2 == 𝒪(2)
    @test 𝒪(2) * 𝒪(3) == 𝒪(5)
    @test 𝒪((1, 2)) * 𝒪((3, 1)) == 𝒪((4, 3))
    @test 𝒪(2)^3 == 𝒪(6)
    @test 𝒪(1)^0 == 𝒪(0)
    @test 𝒪((1, 2))^3 == 𝒪((3, 6))
    @test 𝒪((1, 2))^(2, 3) == 𝒪((2, 6))
    @test_throws DomainError 𝒪(-1)
    @test_throws DomainError 𝒪((1, -1))
    @test_throws ArgumentError 𝒪(())
    @test_throws ArgumentError 𝒪()
end

# Measured column norms of `op` on Chebyshev: for each j, the norm of the image of e_j restricted to
# `head`, its remainder and its total, in `X_codom` and relative to the norm of e_j in `X_dom`
function _calculus_chebyshev_columns(op, head, X_dom, X_codom; J = 80)
    return map(0:J-2) do j
        e = zeros(Chebyshev(J))
        e[j] = 1.0
        img = op(e)
        h, t, w = norm(project(img, head), X_codom), norm(img, X_codom), norm(e, X_dom)
        (h / w, (t - h) / w, t / w)
    end
end

# The four error factors on Chebyshev(N) are restricted column norms: the head columns j ≤ N give the
# finite factor, the tail columns give the cross (head rows) and tail (other rows) factors
function _calculus_test_chebyshev_factors((finite, cross, tail, total), X, N, cols)
    s = Chebyshev(N)
    @test finite(X, s, 1) ≈ maximum(c[1] for c ∈ cols[1:N+1])   rtol=1e-12
    @test cross(X, s, 1)  ≈ maximum(c[1] for c ∈ cols[N+2:end]) rtol=1e-12
    @test tail(X, s, 1)   ≈ maximum(c[2] for c ∈ cols[N+2:end]) rtol=1e-12
    @test total(X, s, 1)  ≈ maximum(c[3] for c ∈ cols)          rtol=1e-12
end

# an image error combines the input errors (0.5, 0.25, 0.75) through the four factors
function _calculus_test_triangular(r, (finite, cross, tail, total), X, s)
    κf, κc, κt, κo = finite(X, s, 1), cross(X, s, 1), tail(X, s, 1), total(X, s, 1)
    @test finite_error(r) == κf * 0.5 + κc * 0.25
    @test tail_error(r) == κt * 0.25
    @test total_error(r) == min(κo * 0.75, κf * 0.5 + κc * 0.25 + κt * 0.25)
end
