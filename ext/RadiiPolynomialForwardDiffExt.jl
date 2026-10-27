module RadiiPolynomialForwardDiffExt

using RadiiPolynomial, ForwardDiff
using ForwardDiff: Dual, valtype
import RadiiPolynomial: _twiddle_table, _chirp_table

# the FFT tables are constants, so they carry no derivative: build them at the
# value type and lift them to the dual type

_twiddle_table(::Type{Complex{D}}, len::Integer) where {D<:Dual} =
    Complex{D}.(_twiddle_table(Complex{valtype(D)}, len))

function _chirp_table(::Type{Complex{D}}, n::Integer) where {D<:Dual}
    w, kernel = _chirp_table(Complex{valtype(D)}, n)
    return Complex{D}.(w), Complex{D}.(kernel)
end

end
