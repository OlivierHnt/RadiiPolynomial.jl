"""
    VerificationFailure <: Exception

Supertype of the exceptions thrown when the input is valid but a rigorous bound
could not be established. Catch it to tell a failed verification apart from
invalid input (`ArgumentError`, `DomainError`, ...).

See also: [`AnalyticityError`](@ref), [`ProofError`](@ref).
"""
abstract type VerificationFailure <: Exception end

Base.showerror(io::IO, e::VerificationFailure) = print(io, nameof(typeof(e)), ": ", e.msg)

"""
    AnalyticityError(msg)

The exception thrown when a function is not verified to be analytic where its
bound needs it: the image of a contour, or the disc of the input error around it,
meets one of the poles or the branch cut of a `Nonlinearity`.
"""
struct AnalyticityError <: VerificationFailure
    msg :: String
end

"""
    ProofError(msg)

The exception thrown when a Newton–Kantorovich argument fails: the bounds `Y`,
`Z₁`, `Z₂` admit no radius of existence, e.g. because the approximation is not
accurate enough.
"""
struct ProofError <: VerificationFailure
    msg :: String
end
