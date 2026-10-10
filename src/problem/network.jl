"""
    FlowNetwork(n_nodes, tail, head; source = 1, sink = n_nodes)

Directed acyclic graph on nodes `1:n_nodes` whose nodes are numbered in topological order
(`tail[e] < head[e]` for every edge `e`). Each pilot's schedule is a `source → sink` path
(thesis §5.1.1). Outgoing edges are stored in CSR form for the pricing dynamic program.
"""
struct FlowNetwork
    n_nodes::Int
    tail::Vector{Int}
    head::Vector{Int}
    source::Int
    sink::Int
    out_start::Vector{Int}   # edges leaving node v: out_edges[out_start[v] : out_start[v+1]-1]
    out_edges::Vector{Int}
end

function FlowNetwork(n_nodes::Integer, tail::Vector{Int}, head::Vector{Int};
                     source::Integer = 1, sink::Integer = n_nodes)
    length(tail) == length(head) || throw(DimensionMismatch("tail and head must have equal length"))
    1 <= source < sink <= n_nodes || throw(ArgumentError("need 1 ≤ source < sink ≤ n_nodes"))
    for e in eachindex(tail)
        1 <= tail[e] < head[e] <= n_nodes ||
            throw(ArgumentError("edge $e = ($(tail[e]) → $(head[e])) violates topological numbering"))
    end
    # CSR adjacency by tail node (counting sort keeps edges of a node in input order)
    # count out-degrees, shifted by one so the prefix sum below yields start offsets
    out_start = zeros(Int, n_nodes + 1)
    for t in tail
        out_start[t + 1] += 1
    end
    # prefix sum: out_start[v] = 1 + Σ_{u<v} outdeg(u)
    out_start[1] = 1
    for v in 1:n_nodes
        out_start[v + 1] += out_start[v]
    end
    # place each edge at the next free slot of its tail node
    fill_pos = copy(out_start)
    out_edges = Vector{Int}(undef, length(tail))
    for e in eachindex(tail)
        out_edges[fill_pos[tail[e]]] = e
        fill_pos[tail[e]] += 1
    end
    return FlowNetwork(Int(n_nodes), tail, head, Int(source), Int(sink), out_start, out_edges)
end

n_nodes(net::FlowNetwork) = net.n_nodes
n_edges(net::FlowNetwork) = length(net.tail)

"Edges leaving node `v`."
@inline out_edges(net::FlowNetwork, v::Integer) =
    @inbounds view(net.out_edges, net.out_start[v]:(net.out_start[v + 1] - 1))

"""
    incidence_matrix(net) -> SparseMatrixCSC

Node-edge flow matrix `F` (thesis eq. 5.1b): `F[tail(e), e] = +1`, `F[head(e), e] = -1`.
"""
function incidence_matrix(net::FlowNetwork)
    E = n_edges(net)
    # two nonzeros per edge column: +1 at the tail, −1 at the head
    rows = vcat(net.tail, net.head)
    cols = vcat(1:E, 1:E)
    vals = vcat(ones(E), -ones(E))
    return sparse(rows, cols, vals, net.n_nodes, E)
end

"Supply vector `b` of the flow conservation constraints: `+1` at the source, `-1` at the sink."
function supply_vector(net::FlowNetwork)
    b = zeros(net.n_nodes)
    b[net.source] = 1.0
    b[net.sink] = -1.0
    return b
end

Base.show(io::IO, net::FlowNetwork) =
    print(io, "FlowNetwork(", net.n_nodes, " nodes, ", n_edges(net), " edges)")
