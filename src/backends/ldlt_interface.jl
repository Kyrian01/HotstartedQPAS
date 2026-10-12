"""
    LDLtInterface

Common interface of the updatable LDLᵀ factorizations of the effective reduced Hessian. The
QPAS (`src/qpas/`) only talks to a factor through these functions, so the linear-algebra
backend is a configuration choice (`QPASOptions(factorization = …)`):

| backend   | type                         | storage                                                  |
|:----------|:-----------------------------|:---------------------------------------------------------|
| `:cholmod`| `Cholmod.LDLtFactor`         | sparse simplicial LDLᵀ of fixed size `n_max × n_max`, identity outside `N` |
| `:dense`  | `DenseLDLt.DenseLDLtFactor`  | dense LDLᵀ of the active block only, `|N| × |N|` (`|N| ≤ m`) |

Both represent a symmetric `n × n` matrix `M` (`n = n_max`) that equals the identity on every
inactive index. The operations, with `k` an index in `1:n`:

- `rowadd!(F, k, rows, vals, nnz)`: row/column `k` (an identity row) becomes the given sparse
  vector, which may only couple `k` to active indices; `k` becomes active
- `rowdel!(F, k)`: row/column `k` becomes the identity again; `k` becomes inactive
- `updown!(F, update, rows, vals, nnz) -> Bool`: `M ± vvᵀ` with `v` supported on active indices;
  `false` if a downdate lost positive definiteness (the factor must then be rebuilt)
- `solve!(F)`, `solve!(F, pattern, npattern)`: solve with `rhs_buffer(F)`
- `pivot(F, k)`, `min_scaled_pivot(F, indices, scale)`: the diagonal `D`
- `reset!(F)`: back to the identity
- `factor_stats(F, active)`, `last_reach(F)`, `backend_stats(F)`, `factor_matrices(F)`:
  statistics and debugging
- `needs_periodic_refactor(F)`: whether updates leave structural garbage that only a rebuild
  removes (CHOLMOD) or not (dense)

`docs/dense_ldlt_backend.md` derives the update rules of the dense backend.
"""
module LDLtInterface

export UpdatableLDLt

"Supertype of the updatable LDLᵀ factorizations (see the module documentation)."
abstract type UpdatableLDLt end

function rowadd! end
function rowdel! end
function updown! end
function solve! end
function rhs_buffer end
function pivot end
function min_scaled_pivot end
function reset! end
function dimension end
function factor_matrices end
function factor_stats end
function last_reach end

"Short name of the backend (`\"cholmod\"`, `\"dense\"`)."
function backend_name end

"""
    needs_periodic_refactor(F) -> Bool

`true` if row deletions and rank-one updates leave numerically zero entries in the stored
structure of the factor, so that a periodic rebuild pays off (`QPASOptions.refactor_interval`).
"""
function needs_periodic_refactor end

"""
    backend_stats(F) -> NamedTuple

Backend-specific counters (e.g. slot capacity and deletion positions of the dense backend).
"""
function backend_stats end

end # module LDLtInterface
