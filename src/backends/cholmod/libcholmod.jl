# Raw bindings to the `cholmod_l_*` API (SuiteSparse_long indices).
#
# This is the only file in the package that calls into CHOLMOD. Each binding is a 1:1 `ccall`;
# argument checking, error handling and memory ownership are handled by the callers in
# `common.jl` and `ldlt_factor.jl`. All indices passed here are 0-based.

const CommonPtr = Ptr{Cvoid}

cholmod_l_start(common::CommonPtr) =
    ccall((:cholmod_l_start, libcholmod), Cint, (CommonPtr,), common)

cholmod_l_finish(common::CommonPtr) =
    ccall((:cholmod_l_finish, libcholmod), Cint, (CommonPtr,), common)

function cholmod_l_version(version::Vector{Cint})
    return ccall((:cholmod_l_version, libcholmod), Cint, (Ptr{Cint},), version)
end

# ── factor ────────────────────────────────────────────────────────────────────

cholmod_l_allocate_factor(n::Integer, common::CommonPtr) =
    ccall((:cholmod_l_allocate_factor, libcholmod), Ptr{CholmodFactor},
          (Csize_t, CommonPtr), n, common)

function cholmod_l_change_factor(to_xtype::Cint, to_ll::Cint, to_super::Cint, to_packed::Cint,
                                 to_monotonic::Cint, L::Ptr{CholmodFactor}, common::CommonPtr)
    return ccall((:cholmod_l_change_factor, libcholmod), Cint,
                 (Cint, Cint, Cint, Cint, Cint, Ptr{CholmodFactor}, CommonPtr),
                 to_xtype, to_ll, to_super, to_packed, to_monotonic, L, common)
end

cholmod_l_free_factor(handle::Base.RefValue{Ptr{CholmodFactor}}, common::CommonPtr) =
    ccall((:cholmod_l_free_factor, libcholmod), Cint,
          (Ptr{Ptr{CholmodFactor}}, CommonPtr), handle, common)

# ── dense ─────────────────────────────────────────────────────────────────────

cholmod_l_allocate_dense(nrow::Integer, ncol::Integer, d::Integer, xdtype::Cint, common::CommonPtr) =
    ccall((:cholmod_l_allocate_dense, libcholmod), Ptr{CholmodDense},
          (Csize_t, Csize_t, Csize_t, Cint, CommonPtr), nrow, ncol, d, xdtype, common)

cholmod_l_free_dense(handle::Base.RefValue{Ptr{CholmodDense}}, common::CommonPtr) =
    ccall((:cholmod_l_free_dense, libcholmod), Cint,
          (Ptr{Ptr{CholmodDense}}, CommonPtr), handle, common)

# ── sparse ────────────────────────────────────────────────────────────────────

function cholmod_l_allocate_sparse(nrow::Integer, ncol::Integer, nzmax::Integer, sorted::Cint,
                                   packed::Cint, stype::Cint, xdtype::Cint, common::CommonPtr)
    return ccall((:cholmod_l_allocate_sparse, libcholmod), Ptr{CholmodSparse},
                 (Csize_t, Csize_t, Csize_t, Cint, Cint, Cint, Cint, CommonPtr),
                 nrow, ncol, nzmax, sorted, packed, stype, xdtype, common)
end

cholmod_l_free_sparse(handle::Base.RefValue{Ptr{CholmodSparse}}, common::CommonPtr) =
    ccall((:cholmod_l_free_sparse, libcholmod), Cint,
          (Ptr{Ptr{CholmodSparse}}, CommonPtr), handle, common)

# ── modify module ─────────────────────────────────────────────────────────────

cholmod_l_rowadd(k::Integer, R::Ptr{CholmodSparse}, L::Ptr{CholmodFactor}, common::CommonPtr) =
    ccall((:cholmod_l_rowadd, libcholmod), Cint,
          (Csize_t, Ptr{CholmodSparse}, Ptr{CholmodFactor}, CommonPtr), k, R, L, common)

cholmod_l_rowdel(k::Integer, R::Ptr{CholmodSparse}, L::Ptr{CholmodFactor}, common::CommonPtr) =
    ccall((:cholmod_l_rowdel, libcholmod), Cint,
          (Csize_t, Ptr{CholmodSparse}, Ptr{CholmodFactor}, CommonPtr), k, R, L, common)

# LDLᵀ ← LDLᵀ + CCᵀ (update = TRUE) or LDLᵀ − CCᵀ (update = FALSE), C sparse n × k.
cholmod_l_updown(update::Cint, C::Ptr{CholmodSparse}, L::Ptr{CholmodFactor}, common::CommonPtr) =
    ccall((:cholmod_l_updown, libcholmod), Cint,
          (Cint, Ptr{CholmodSparse}, Ptr{CholmodFactor}, CommonPtr), update, C, L, common)

# ── solve ─────────────────────────────────────────────────────────────────────

# `Bset == C_NULL`: dense solve. Otherwise only B(Bset) is used and X is computed on
# Xset = reach(Bset), which is returned in `Xset` (pass a handle holding C_NULL to ignore it).
function cholmod_l_solve2(sys::Cint, L::Ptr{CholmodFactor}, B::Ptr{CholmodDense},
                          Bset::Ptr{CholmodSparse},
                          X::Base.RefValue{Ptr{CholmodDense}},
                          Xset::Base.RefValue{Ptr{CholmodSparse}},
                          Y::Base.RefValue{Ptr{CholmodDense}},
                          E::Base.RefValue{Ptr{CholmodDense}}, common::CommonPtr)
    return ccall((:cholmod_l_solve2, libcholmod), Cint,
                 (Cint, Ptr{CholmodFactor}, Ptr{CholmodDense}, Ptr{CholmodSparse},
                  Ptr{Ptr{CholmodDense}}, Ptr{Ptr{CholmodSparse}},
                  Ptr{Ptr{CholmodDense}}, Ptr{Ptr{CholmodDense}}, CommonPtr),
                 sys, L, B, Bset, X, Xset, Y, E, common)
end
