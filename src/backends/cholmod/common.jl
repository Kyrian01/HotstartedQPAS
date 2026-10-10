# Owned `cholmod_common` workspace.
#
# The struct is ~2.7 kB and its layout changes between CHOLMOD versions, so it is kept as an
# opaque byte buffer whose size and field offsets come from the generated bindings for the
# bundled library. `CholmodCommon` has no finalizer: its owner (an `LDLtFactor`) frees every
# CHOLMOD object first and then calls `finish!`, which avoids finalizer-ordering hazards.

const _LSS = SparseArrays.LibSuiteSparse
const COMMON_SIZE = sizeof(_LSS.cholmod_common_struct)

_common_offset(name::Symbol) =
    fieldoffset(_LSS.cholmod_common_struct,
                findfirst(==(name), fieldnames(_LSS.cholmod_common_struct)))

const COMMON_STATUS_OFFSET = _common_offset(:status)
const COMMON_PRINT_OFFSET = _common_offset(:print)

"Exception raised when a CHOLMOD call reports failure."
struct CholmodError <: Exception
    func::String
    status::Cint
end

Base.showerror(io::IO, e::CholmodError) =
    print(io, "CholmodError: ", e.func, " failed (Common->status = ", e.status, ")")

mutable struct CholmodCommon
    buffer::Vector{UInt8}
    finished::Bool
end

function CholmodCommon(; print_level::Integer = 0)
    common = CholmodCommon(zeros(UInt8, COMMON_SIZE), false)
    GC.@preserve common begin
        ptr = common_ptr(common)
        cholmod_l_start(ptr) == CHOLMOD_TRUE || throw(CholmodError("cholmod_l_start", Cint(-1)))
        unsafe_store!(Ptr{Cint}(ptr + COMMON_PRINT_OFFSET), Cint(print_level))
    end
    return common
end

@inline common_ptr(c::CholmodCommon) = Ptr{Cvoid}(pointer(c.buffer))

status(c::CholmodCommon) =
    GC.@preserve c unsafe_load(Ptr{Cint}(common_ptr(c) + COMMON_STATUS_OFFSET))

function finish!(c::CholmodCommon)
    c.finished && return nothing
    GC.@preserve c cholmod_l_finish(common_ptr(c))
    c.finished = true
    return nothing
end

"Throw a [`CholmodError`](@ref) unless `ret` is CHOLMOD's TRUE."
@inline function check_call(ret::Cint, c::CholmodCommon, func::String)
    ret == CHOLMOD_TRUE || throw(CholmodError(func, status(c)))
    return nothing
end

"""
    cholmod_version() -> VersionNumber

Version of the CHOLMOD library that is loaded.
"""
function cholmod_version()
    v = zeros(Cint, 3)
    cholmod_l_version(v)
    return VersionNumber(v[1], v[2], v[3])
end
