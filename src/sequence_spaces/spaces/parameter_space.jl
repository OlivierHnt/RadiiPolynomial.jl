"""
    ParameterSpace <: SingleSpace

Space of parameters.

Fields:
- `dimension :: Int`
"""
struct ParameterSpace <: SingleSpace
    dimension :: Int
end

##

dimension(s::ParameterSpace) = s.dimension
startindex(s::ParameterSpace) = 1
endindex(s::ParameterSpace) = s.dimension
allindices(s::ParameterSpace) = Base.OneTo(endindex(s))
isindexof(::Colon, space::ParameterSpace) = true
isindexof(i, space::ParameterSpace) = i ∈ allindices(space)
_findindex(i, space::ParameterSpace) = i

##

Base.issubset(s₁::ParameterSpace, s₂::ParameterSpace) = s₁.dimension == s₂.dimension
Base.:(==)(s₁::ParameterSpace, s₂::ParameterSpace) = s₁.dimension == s₂.dimension

## show

Base.show(io::IO, space::ParameterSpace) = print(io, pretty_string(space))

pretty_string(s::ParameterSpace) = "𝔽"*superscriptify(s.dimension)
