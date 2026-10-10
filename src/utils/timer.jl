"""
    SectionTimer(names)

Accumulates wall-clock time and call counts for a fixed list of named sections. Overhead is
two `time_ns()` calls per measured section, so it is safe to leave on in hot loops.

```julia
t = SectionTimer((:solve, :update))
@timed_section t :solve begin ... end
```
"""
struct SectionTimer
    names::Vector{Symbol}
    seconds::Vector{Float64}
    calls::Vector{Int}
end

SectionTimer(names) = SectionTimer(collect(Symbol, names), zeros(length(names)), zeros(Int, length(names)))

@inline function _section_index(t::SectionTimer, name::Symbol)
    # linear scan is fine: timers only have a handful of sections
    @inbounds for k in eachindex(t.names)
        t.names[k] === name && return k
    end
    throw(ArgumentError("unknown timer section :$name"))
end

@inline function record!(t::SectionTimer, name::Symbol, ns::UInt64)
    k = _section_index(t, name)
    @inbounds t.seconds[k] += ns * 1e-9
    @inbounds t.calls[k] += 1
    return nothing
end

"Total seconds spent in section `name`."
seconds(t::SectionTimer, name::Symbol) = t.seconds[_section_index(t, name)]

function Base.empty!(t::SectionTimer)
    fill!(t.seconds, 0.0)
    fill!(t.calls, 0)
    return t
end

"Add the totals of `src` into `dst` (same section names)."
function accumulate!(dst::SectionTimer, src::SectionTimer)
    dst.names == src.names || throw(ArgumentError("timer sections differ"))
    dst.seconds .+= src.seconds
    dst.calls .+= src.calls
    return dst
end

"""
    @timed_section timer name expr

Evaluate `expr` and add its wall-clock time to section `name` of `timer`.
"""
macro timed_section(timer, name, expr)
    # `local` keeps t0/val hygienic; the block evaluates to the value of `expr`
    return quote
        local t0 = time_ns()
        local val = $(esc(expr))
        record!($(esc(timer)), $(esc(name)), time_ns() - t0)
        val
    end
end
