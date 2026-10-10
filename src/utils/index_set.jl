"""
    IndexSet(capacity)

Unordered set of integers in `1:capacity` with O(1) `push!`, `delete!` and `in`, and
cache-friendly iteration over its elements (`items(s)`). Used for the working set W and the
nullspace set N of the QPAS, which change by a single index per iteration.
"""
struct IndexSet
    elements::Vector{Int}   # the members, in arbitrary order
    position::Vector{Int}   # position[j] = index of j in `elements`, or 0 if j ∉ set
end

IndexSet(capacity::Integer) = IndexSet(Int[], zeros(Int, capacity))

Base.length(s::IndexSet) = length(s.elements)
Base.isempty(s::IndexSet) = isempty(s.elements)
Base.in(j::Integer, s::IndexSet) = @inbounds s.position[j] != 0
Base.iterate(s::IndexSet, state...) = iterate(s.elements, state...)
Base.eltype(::Type{IndexSet}) = Int
capacity(s::IndexSet) = length(s.position)

"Read-only view of the members (arbitrary order)."
items(s::IndexSet) = s.elements

function Base.push!(s::IndexSet, j::Integer)
    @boundscheck checkbounds(s.position, j)
    @inbounds if s.position[j] == 0
        push!(s.elements, j)
        s.position[j] = length(s.elements)
    end
    return s
end

function Base.delete!(s::IndexSet, j::Integer)
    @boundscheck checkbounds(s.position, j)
    @inbounds begin
        pos = s.position[j]
        pos == 0 && return s
        last = s.elements[end]
        s.elements[pos] = last
        s.position[last] = pos
        pop!(s.elements)
        s.position[j] = 0
    end
    return s
end

function Base.empty!(s::IndexSet)
    @inbounds for j in s.elements
        s.position[j] = 0
    end
    empty!(s.elements)
    return s
end
