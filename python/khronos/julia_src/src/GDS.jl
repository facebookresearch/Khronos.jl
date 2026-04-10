# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Pure-Julia GDSII binary file reader.
# Parses .gds files and returns 2D polygon vertices suitable for
# extrusion into PolygonalPrism shapes via GeometryPrimitives.jl.

# ────────────────────────────────────────────────────────────
# GDSII Record Type Constants
# ────────────────────────────────────────────────────────────

const GDS_HEADER    = 0x00
const GDS_BGNLIB    = 0x01
const GDS_LIBNAME   = 0x02
const GDS_UNITS     = 0x03
const GDS_ENDLIB    = 0x04
const GDS_BGNSTR    = 0x05
const GDS_STRNAME   = 0x06
const GDS_ENDSTR    = 0x07
const GDS_BOUNDARY  = 0x08
const GDS_PATH      = 0x09
const GDS_SREF      = 0x0A
const GDS_AREF      = 0x0B
const GDS_TEXT      = 0x0C
const GDS_LAYER     = 0x0D
const GDS_DATATYPE  = 0x0E
const GDS_WIDTH     = 0x0F
const GDS_XY        = 0x10
const GDS_ENDEL     = 0x11
const GDS_SNAME     = 0x12
const GDS_COLROW    = 0x13
const GDS_NODE      = 0x15
const GDS_TEXTTYPE  = 0x16
const GDS_STRANS    = 0x1A
const GDS_MAG       = 0x1B
const GDS_ANGLE     = 0x1C
const GDS_PATHTYPE  = 0x21
const GDS_BOXTYPE   = 0x2E
const GDS_BOX       = 0x2D
const GDS_BGNEXTN   = 0x30
const GDS_ENDEXTN   = 0x31
const GDS_PROPATTR  = 0x2B
const GDS_PROPVALUE = 0x2C

# Data type codes
const GDS_DT_NODATA    = 0x00
const GDS_DT_BITARRAY  = 0x01
const GDS_DT_INT16     = 0x02
const GDS_DT_INT32     = 0x03
const GDS_DT_REAL4     = 0x04
const GDS_DT_REAL8     = 0x05
const GDS_DT_STRING    = 0x06

# ────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────

struct GDSPolygon
    vertices::Vector{Tuple{Float64,Float64}}
    layer::Int
    datatype::Int
end

struct GDSReference
    cell_name::String
    origin::Tuple{Float64,Float64}
    rotation::Float64         # radians
    magnification::Float64
    x_reflection::Bool
    is_array::Bool
    columns::Int
    rows::Int
    col_vec::Tuple{Float64,Float64}
    row_vec::Tuple{Float64,Float64}
end

struct GDSCell
    name::String
    polygons::Vector{GDSPolygon}
    references::Vector{GDSReference}
end

struct GDSLibrary
    name::String
    unit::Float64       # user unit in meters
    precision::Float64  # database unit in meters
    cells::Dict{String,GDSCell}
end

# ────────────────────────────────────────────────────────────
# GDSII 8-byte Real Decoding (excess-64 base-16)
# ────────────────────────────────────────────────────────────

"""
    decode_gdsii_real(io::IO) -> Float64

Read 8 bytes from `io` and decode the GDSII excess-64 base-16 floating point format.

The format is:
- Bit 63: sign (1 = negative)
- Bits 62-56: exponent + 64 (excess-64, base-16)
- Bits 55-0: 56-bit mantissa (fractional)

Result = (-1)^sign × mantissa × 16^(exponent - 64)
"""
function decode_gdsii_real(io::IO)::Float64
    raw = ntoh(read(io, UInt64))
    sign_bit = (raw >> 63) & 1
    exponent = Int((raw >> 56) & 0x7F) - 64
    mantissa = Float64(raw & 0x00FFFFFFFFFFFFFF) / exp2(56)
    value = mantissa * exp2(4 * exponent)  # 16^exp = 2^(4*exp)
    return sign_bit == 1 ? -value : value
end

"""
    decode_gdsii_real(bytes::AbstractVector{UInt8}) -> Float64

Decode 8 bytes (big-endian) as a GDSII excess-64 real.
"""
function decode_gdsii_real(bytes::AbstractVector{UInt8})::Float64
    raw = UInt64(0)
    for i in 1:8
        raw = (raw << 8) | UInt64(bytes[i])
    end
    sign_bit = (raw >> 63) & 1
    exponent = Int((raw >> 56) & 0x7F) - 64
    mantissa = Float64(raw & 0x00FFFFFFFFFFFFFF) / exp2(56)
    value = mantissa * exp2(4 * exponent)
    return sign_bit == 1 ? -value : value
end

# ────────────────────────────────────────────────────────────
# PATH → Polygon Expansion
# ────────────────────────────────────────────────────────────

"""
    expand_path(centerline, half_width, pathtype, bgnextn, endextn) -> Vector{Tuple{Float64,Float64}}

Convert a PATH centerline with constant width to a closed polygon outline.

# Arguments
- `centerline`: vector of (x,y) centerline points
- `half_width`: half the path width
- `pathtype`: 0=flush, 1=round (→flush), 2=half-width extension, 4=explicit extension
- `bgnextn`: begin extension distance (for pathtype 4)
- `endextn`: end extension distance (for pathtype 4)
"""
function expand_path(
    centerline::Vector{Tuple{Float64,Float64}},
    half_width::Float64,
    pathtype::Int,
    bgnextn::Float64,
    endextn::Float64,
)::Vector{Tuple{Float64,Float64}}
    N = length(centerline)
    N < 2 && return Tuple{Float64,Float64}[]

    # Compute tangents and normals at each point
    tangents = Vector{Tuple{Float64,Float64}}(undef, N)
    normals = Vector{Tuple{Float64,Float64}}(undef, N)

    for i in 1:N
        if i == 1
            dx = centerline[2][1] - centerline[1][1]
            dy = centerline[2][2] - centerline[1][2]
        elseif i == N
            dx = centerline[N][1] - centerline[N-1][1]
            dy = centerline[N][2] - centerline[N-1][2]
        else
            dx = centerline[i+1][1] - centerline[i-1][1]
            dy = centerline[i+1][2] - centerline[i-1][2]
        end
        len = sqrt(dx * dx + dy * dy)
        if len < 1e-30
            tangents[i] = (1.0, 0.0)
            normals[i] = (0.0, 1.0)
        else
            tangents[i] = (dx / len, dy / len)
            normals[i] = (-dy / len, dx / len)
        end
    end

    # Build right and left edge points
    right_pts = Vector{Tuple{Float64,Float64}}(undef, N)
    left_pts = Vector{Tuple{Float64,Float64}}(undef, N)

    for i in 1:N
        cx, cy = centerline[i]
        nx, ny = normals[i]
        right_pts[i] = (cx - nx * half_width, cy - ny * half_width)
        left_pts[i] = (cx + nx * half_width, cy + ny * half_width)
    end

    # Use miter joins at interior vertices for better accuracy
    for i in 2:N-1
        # Compute per-segment tangent/normal pairs
        dx0 = centerline[i][1] - centerline[i-1][1]
        dy0 = centerline[i][2] - centerline[i-1][2]
        len0 = sqrt(dx0^2 + dy0^2)
        if len0 < 1e-30; continue; end
        t0x, t0y = dx0 / len0, dy0 / len0
        n0x, n0y = -t0y, t0x

        dx1 = centerline[i+1][1] - centerline[i][1]
        dy1 = centerline[i+1][2] - centerline[i][2]
        len1 = sqrt(dx1^2 + dy1^2)
        if len1 < 1e-30; continue; end
        t1x, t1y = dx1 / len1, dy1 / len1
        n1x, n1y = -t1y, t1x

        # Right edge: intersect incoming and outgoing offset lines
        _miter_join!(right_pts, i, centerline[i], n0x, n0y, t0x, t0y,
                     n1x, n1y, t1x, t1y, -half_width)
        _miter_join!(left_pts, i, centerline[i], n0x, n0y, t0x, t0y,
                     n1x, n1y, t1x, t1y, half_width)
    end

    # Apply end caps
    result = Tuple{Float64,Float64}[]

    # Start cap
    tx, ty = tangents[1]
    if pathtype == 2
        ext = half_width
        push!(result, (right_pts[1][1] - tx * ext, right_pts[1][2] - ty * ext))
        push!(result, (left_pts[1][1] - tx * ext, left_pts[1][2] - ty * ext))
    elseif pathtype == 4
        push!(result, (right_pts[1][1] - tx * bgnextn, right_pts[1][2] - ty * bgnextn))
        push!(result, (left_pts[1][1] - tx * bgnextn, left_pts[1][2] - ty * bgnextn))
    else  # flush (0) or round (1, approximated as flush)
        push!(result, right_pts[1])
        push!(result, left_pts[1])
    end

    # Left edge (forward)
    for i in 2:N-1
        push!(result, left_pts[i])
    end

    # End cap
    tx, ty = tangents[N]
    if pathtype == 2
        ext = half_width
        push!(result, (left_pts[N][1] + tx * ext, left_pts[N][2] + ty * ext))
        push!(result, (right_pts[N][1] + tx * ext, right_pts[N][2] + ty * ext))
    elseif pathtype == 4
        push!(result, (left_pts[N][1] + tx * endextn, left_pts[N][2] + ty * endextn))
        push!(result, (right_pts[N][1] + tx * endextn, right_pts[N][2] + ty * endextn))
    else
        push!(result, left_pts[N])
        push!(result, right_pts[N])
    end

    # Right edge (backward)
    for i in N-1:-1:2
        push!(result, right_pts[i])
    end

    return result
end

"""
Compute miter join intersection for a path edge at an interior vertex.
"""
function _miter_join!(
    pts::Vector{Tuple{Float64,Float64}},
    idx::Int,
    center::Tuple{Float64,Float64},
    n0x::Float64, n0y::Float64,
    t0x::Float64, t0y::Float64,
    n1x::Float64, n1y::Float64,
    t1x::Float64, t1y::Float64,
    offset::Float64,  # +half_width for left, -half_width for right
)
    # Points on the two offset lines
    p0x = center[1] + n0x * offset
    p0y = center[2] + n0y * offset
    p1x = center[1] + n1x * offset
    p1y = center[2] + n1y * offset

    # Line intersection: p0 + u*t0 = p1 + v*t1
    denom = t0x * t1y - t0y * t1x
    if abs(denom) < 1e-15
        # Parallel segments — just use the average
        pts[idx] = ((p0x + p1x) / 2, (p0y + p1y) / 2)
        return
    end
    dpx = p1x - p0x
    dpy = p1y - p0y
    u = (dpx * t1y - dpy * t1x) / denom

    # Clamp miter to prevent spikes at very acute angles
    max_u = abs(offset) * 4.0
    u = clamp(u, -max_u, max_u)

    pts[idx] = (p0x + u * t0x, p0y + u * t0y)
end

# ────────────────────────────────────────────────────────────
# SREF/AREF Transform
# ────────────────────────────────────────────────────────────

"""
    transform_point(x, y, mag, x_reflect, rotation, origin) -> (Float64, Float64)

Apply GDSII SREF transformation to a point.
Order: magnification → x_reflection → rotation → translation.
"""
function transform_gds_point(
    x::Float64, y::Float64,
    mag::Float64,
    x_reflect::Bool,
    rotation::Float64,
    origin::Tuple{Float64,Float64},
)::Tuple{Float64,Float64}
    # Scale
    x *= mag
    y *= mag
    # Reflect across x-axis
    if x_reflect
        y = -y
    end
    # Rotate
    ca = cos(rotation)
    sa = sin(rotation)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    # Translate
    return (xr + origin[1], yr + origin[2])
end

# ────────────────────────────────────────────────────────────
# Main GDS Reader
# ────────────────────────────────────────────────────────────

"""
    read_gds(filename::AbstractString) -> GDSLibrary

Read a GDSII binary file and return a `GDSLibrary` containing all cells,
polygons, paths (expanded to polygons), and structure references.

Coordinates are returned in GDS user units (typically microns).
"""
function read_gds(filename::AbstractString)::GDSLibrary
    open(filename, "r") do io
        _read_gds_stream(io)
    end
end

function _read_gds_stream(io::IO)::GDSLibrary
    lib_name = ""
    db_in_user = 1.0
    db_in_meters = 1.0
    factor = 1.0  # converts database integer → user units

    cells = Dict{String,GDSCell}()

    # Current cell being parsed
    cur_cell_name = ""
    cur_polygons = GDSPolygon[]
    cur_references = GDSReference[]
    in_cell = false

    # Current element state
    elem_type = :none  # :boundary, :path, :sref, :aref, :text, :node, :box
    elem_layer = 0
    elem_datatype = 0
    elem_points = Tuple{Float64,Float64}[]
    elem_width = 0.0
    elem_pathtype = 0
    elem_bgnextn = 0.0
    elem_endextn = 0.0
    elem_sname = ""
    elem_strans_reflect = false
    elem_mag = 1.0
    elem_angle = 0.0  # radians
    elem_colrow = (1, 1)

    while !eof(io)
        # Read record header (4 bytes)
        header_bytes = read(io, 4)
        length(header_bytes) < 4 && break

        record_length = (UInt16(header_bytes[1]) << 8) | UInt16(header_bytes[2])
        record_type = header_bytes[3]
        data_type = header_bytes[4]

        # Read payload
        payload_length = Int(record_length) - 4
        if payload_length < 0
            break
        end

        payload = payload_length > 0 ? read(io, payload_length) : UInt8[]
        if length(payload) < payload_length
            break
        end

        # Dispatch on record type
        if record_type == GDS_HEADER
            # Skip

        elseif record_type == GDS_BGNLIB
            # Skip

        elseif record_type == GDS_LIBNAME
            lib_name = _decode_string(payload)

        elseif record_type == GDS_UNITS
            # Two 8-byte GDSII reals: db_in_user, db_in_meters
            db_in_user = decode_gdsii_real(@view payload[1:8])
            db_in_meters = decode_gdsii_real(@view payload[9:16])
            factor = db_in_user

        elseif record_type == GDS_ENDLIB
            # All done
            break

        elseif record_type == GDS_BGNSTR
            cur_cell_name = ""
            cur_polygons = GDSPolygon[]
            cur_references = GDSReference[]
            in_cell = true

        elseif record_type == GDS_STRNAME
            cur_cell_name = _decode_string(payload)

        elseif record_type == GDS_ENDSTR
            if in_cell && cur_cell_name != ""
                cells[cur_cell_name] = GDSCell(cur_cell_name, cur_polygons, cur_references)
            end
            in_cell = false

        elseif record_type == GDS_BOUNDARY || record_type == GDS_BOX
            elem_type = :boundary
            elem_layer = 0
            elem_datatype = 0
            elem_points = Tuple{Float64,Float64}[]

        elseif record_type == GDS_PATH
            elem_type = :path
            elem_layer = 0
            elem_datatype = 0
            elem_points = Tuple{Float64,Float64}[]
            elem_width = 0.0
            elem_pathtype = 0
            elem_bgnextn = 0.0
            elem_endextn = 0.0

        elseif record_type == GDS_SREF
            elem_type = :sref
            elem_sname = ""
            elem_points = Tuple{Float64,Float64}[]
            elem_strans_reflect = false
            elem_mag = 1.0
            elem_angle = 0.0

        elseif record_type == GDS_AREF
            elem_type = :aref
            elem_sname = ""
            elem_points = Tuple{Float64,Float64}[]
            elem_strans_reflect = false
            elem_mag = 1.0
            elem_angle = 0.0
            elem_colrow = (1, 1)

        elseif record_type == GDS_TEXT
            elem_type = :text

        elseif record_type == GDS_NODE
            elem_type = :node

        elseif record_type == GDS_LAYER
            elem_layer = _read_int16(payload)

        elseif record_type == GDS_DATATYPE || record_type == GDS_BOXTYPE || record_type == GDS_TEXTTYPE
            elem_datatype = _read_int16(payload)

        elseif record_type == GDS_WIDTH
            w = _read_int32(payload)
            elem_width = factor * abs(Float64(w))

        elseif record_type == GDS_XY
            elem_points = _read_xy(payload, factor)

        elseif record_type == GDS_PATHTYPE
            elem_pathtype = _read_int16(payload)

        elseif record_type == GDS_BGNEXTN
            elem_bgnextn = factor * Float64(_read_int32(payload))

        elseif record_type == GDS_ENDEXTN
            elem_endextn = factor * Float64(_read_int32(payload))

        elseif record_type == GDS_SNAME
            elem_sname = _decode_string(payload)

        elseif record_type == GDS_COLROW
            cols = _read_int16(payload, 1)
            rows = _read_int16(payload, 3)
            elem_colrow = (cols, rows)

        elseif record_type == GDS_STRANS
            bits = _read_uint16(payload)
            elem_strans_reflect = (bits >> 15) & 1 == 1

        elseif record_type == GDS_MAG
            elem_mag = decode_gdsii_real(@view payload[1:8])

        elseif record_type == GDS_ANGLE
            elem_angle = deg2rad(decode_gdsii_real(@view payload[1:8]))

        elseif record_type == GDS_ENDEL
            if in_cell
                _finalize_element!(
                    elem_type, cur_polygons, cur_references,
                    elem_layer, elem_datatype, elem_points,
                    elem_width, elem_pathtype, elem_bgnextn, elem_endextn,
                    elem_sname, elem_strans_reflect, elem_mag, elem_angle,
                    elem_colrow, factor,
                )
            end
            elem_type = :none

        # Ignore all other record types (PROPATTR, PROPVALUE, etc.)
        end
    end

    return GDSLibrary(lib_name, db_in_user * db_in_meters / db_in_user,
                      db_in_meters, cells)
end

# ────────────────────────────────────────────────────────────
# Element Finalization
# ────────────────────────────────────────────────────────────

function _finalize_element!(
    elem_type::Symbol,
    polygons::Vector{GDSPolygon},
    references::Vector{GDSReference},
    layer::Int, datatype::Int,
    points::Vector{Tuple{Float64,Float64}},
    width::Float64, pathtype::Int,
    bgnextn::Float64, endextn::Float64,
    sname::String,
    x_reflect::Bool, mag::Float64, angle::Float64,
    colrow::Tuple{Int,Int},
    factor::Float64,
)
    if elem_type == :boundary
        if length(points) >= 3
            # Remove duplicate closing point (GDSII convention)
            verts = copy(points)
            if length(verts) > 1 && _points_equal(verts[1], verts[end])
                pop!(verts)
            end
            push!(polygons, GDSPolygon(verts, layer, datatype))
        end

    elseif elem_type == :path
        if length(points) >= 2 && width > 0.0
            half_w = width / 2.0
            expanded = expand_path(points, half_w, pathtype, bgnextn, endextn)
            if length(expanded) >= 3
                push!(polygons, GDSPolygon(expanded, layer, datatype))
            end
        elseif length(points) >= 2 && width == 0.0
            # Zero-width path: treat as a polyline (no expansion possible)
            # Skip — zero-width paths are non-physical
        end

    elseif elem_type == :sref
        if !isempty(sname) && !isempty(points)
            origin = points[1]
            push!(references, GDSReference(
                sname, origin, angle, mag, x_reflect,
                false, 1, 1, (0.0, 0.0), (0.0, 0.0),
            ))
        end

    elseif elem_type == :aref
        if !isempty(sname) && length(points) >= 3
            origin = points[1]
            cols, rows = colrow

            # Compute spacing vectors from the three XY points
            col_vec = (
                (points[2][1] - origin[1]) / max(cols, 1),
                (points[2][2] - origin[2]) / max(cols, 1),
            )
            row_vec = (
                (points[3][1] - origin[1]) / max(rows, 1),
                (points[3][2] - origin[2]) / max(rows, 1),
            )

            push!(references, GDSReference(
                sname, origin, angle, mag, x_reflect,
                true, cols, rows, col_vec, row_vec,
            ))
        end
    end
    # :text, :node → ignored
end

# ────────────────────────────────────────────────────────────
# Cell Flattening
# ────────────────────────────────────────────────────────────

"""
    flatten_cell(lib::GDSLibrary, cell_name::String; layers=nothing) -> Vector{GDSPolygon}

Recursively flatten a cell, resolving all SREF and AREF references.
Returns a flat list of polygons with all transforms applied.

# Arguments
- `lib`: the parsed GDS library
- `cell_name`: name of the cell to flatten
- `layers`: optional filter — set of `(layer, datatype)` tuples, or a single
  `(layer, datatype)` tuple, or `nothing` for all layers
"""
function flatten_cell(
    lib::GDSLibrary,
    cell_name::String;
    layers=nothing,
)::Vector{GDSPolygon}
    result = GDSPolygon[]
    _flatten_recursive!(result, lib, cell_name, 1.0, false, 0.0, (0.0, 0.0), 0)
    if !isnothing(layers)
        layer_set = _to_layer_set(layers)
        filter!(p -> (p.layer, p.datatype) in layer_set, result)
    end
    return result
end

"""
    flatten_cell(lib::GDSLibrary, cell_name::String, layer::Int, datatype::Int=0) -> Vector{GDSPolygon}

Convenience method: flatten a cell and filter to a single layer/datatype.
"""
function flatten_cell(
    lib::GDSLibrary,
    cell_name::String,
    layer::Int,
    datatype::Int=0,
)::Vector{GDSPolygon}
    flatten_cell(lib, cell_name; layers=Set([(layer, datatype)]))
end

function _to_layer_set(layers)
    if layers isa Set
        return layers
    elseif layers isa Tuple{Int,Int}
        return Set([layers])
    elseif layers isa AbstractVector
        return Set(Tuple{Int,Int}[l for l in layers])
    else
        return Set(layers)
    end
end

function _flatten_recursive!(
    result::Vector{GDSPolygon},
    lib::GDSLibrary,
    cell_name::String,
    mag::Float64,
    x_reflect::Bool,
    rotation::Float64,
    origin::Tuple{Float64,Float64},
    depth::Int,
)
    depth > 128 && error("GDS cell hierarchy exceeds 128 levels — possible circular reference")

    if !haskey(lib.cells, cell_name)
        @warn "GDS reference to unknown cell: $cell_name"
        return
    end
    cell = lib.cells[cell_name]

    # Add all polygons from this cell, applying the current transform
    for poly in cell.polygons
        transformed_verts = Tuple{Float64,Float64}[]
        for (x, y) in poly.vertices
            push!(transformed_verts, transform_gds_point(x, y, mag, x_reflect, rotation, origin))
        end
        push!(result, GDSPolygon(transformed_verts, poly.layer, poly.datatype))
    end

    # Recurse into references
    for ref in cell.references
        if ref.is_array
            for ci in 0:ref.columns-1
                for ri in 0:ref.rows-1
                    # Array offset in the referenced cell's coordinate frame
                    ox = ci * ref.col_vec[1] + ri * ref.row_vec[1]
                    oy = ci * ref.col_vec[2] + ri * ref.row_vec[2]
                    # Apply the reference's own transform to get the array instance origin
                    ref_origin = transform_gds_point(
                        ox + ref.origin[1], oy + ref.origin[2],
                        1.0, false, 0.0, (0.0, 0.0),  # no double-transform of the outer context
                    )
                    # Compose the reference transform with the current transform
                    new_mag = mag * ref.mag
                    new_reflect = xor(x_reflect, ref.x_reflection)
                    new_rot = rotation + (x_reflect ? -ref.rotation : ref.rotation)
                    # The new origin: transform the ref_origin through the outer context
                    new_origin = transform_gds_point(
                        ref_origin[1], ref_origin[2],
                        mag, x_reflect, rotation, origin,
                    )
                    _flatten_recursive!(result, lib, ref.cell_name,
                                       new_mag, new_reflect, new_rot, new_origin, depth + 1)
                end
            end
        else
            new_mag = mag * ref.magnification
            new_reflect = xor(x_reflect, ref.x_reflection)
            new_rot = rotation + (x_reflect ? -ref.rotation : ref.rotation)
            new_origin = transform_gds_point(
                ref.origin[1], ref.origin[2],
                mag, x_reflect, rotation, origin,
            )
            _flatten_recursive!(result, lib, ref.cell_name,
                               new_mag, new_reflect, new_rot, new_origin, depth + 1)
        end
    end
end

# ────────────────────────────────────────────────────────────
# Khronos Integration: GDS → Object
# ────────────────────────────────────────────────────────────

"""
    gds_polygons_to_objects(
        polygons::Vector{GDSPolygon},
        z_min::Real, z_max::Real,
        material::Material;
        axis::Int=3,
    ) -> Vector{Object}

Convert a list of `GDSPolygon`s into Khronos `Object`s by extruding each
polygon along `axis` from `z_min` to `z_max`.

Each polygon becomes one or more `Prism` objects (from GeometryPrimitives.jl).
Non-convex polygons are decomposed into triangles via ear clipping.
"""
function gds_polygons_to_objects(
    polygons::Vector{GDSPolygon},
    z_min::Real, z_max::Real,
    material::Material;
    axis::Int=3,
)::Vector{Object}
    objects = Object[]
    for poly in polygons
        _polygon_to_objects!(objects, poly.vertices, z_min, z_max, material, axis)
    end
    return objects
end

"""
    gds_to_objects(
        lib::GDSLibrary,
        cell_name::String,
        layer_map::Dict;
        axis::Int=3,
    ) -> Vector{Object}

High-level convenience function: flatten a GDS cell and create `Object`s
for each layer according to `layer_map`.

`layer_map` maps `(layer, datatype)` tuples to `(z_min, z_max, Material)` tuples.

# Example
```julia
lib = read_gds("coupler.gds")
objects = gds_to_objects(lib, "COUPLER", Dict(
    (0, 0) => (-4.0, 0.0, Material{Float64}(ε=1.45^2)),
    (1, 0) => (0.0, 0.22, Material{Float64}(ε=3.48^2)),
); axis=3)
```
"""
function gds_to_objects(
    lib::GDSLibrary,
    cell_name::String,
    layer_map::Dict;
    axis::Int=3,
)::Vector{Object}
    objects = Object[]
    for ((layer, datatype), (z_min, z_max, mat)) in layer_map
        polys = flatten_cell(lib, cell_name, layer, datatype)
        append!(objects, gds_polygons_to_objects(polys, z_min, z_max, mat; axis=axis))
    end
    return objects
end

"""
    _polygon_to_objects!(objects, vertices, z_min, z_max, material, axis)

Convert a single polygon into one or more `Object`s appended to `objects`.
Convex polygons produce a single `Prism`; non-convex polygons are decomposed
into triangles via ear clipping.
"""
function _polygon_to_objects!(
    objects::Vector{Object},
    vertices::Vector{Tuple{Float64,Float64}},
    z_min::Real, z_max::Real,
    material::Material,
    axis::Int,
)
    K = length(vertices)
    K < 3 && return
    height = Float64(z_max - z_min)
    height <= 0 && return

    if _is_convex(vertices)
        obj = _make_prism(vertices, z_min, z_max, material, axis)
        !isnothing(obj) && push!(objects, obj)
    else
        # Ear-clipping triangulation for non-convex polygons
        triangles = _ear_clip_triangulate(vertices)
        for tri in triangles
            obj = _make_prism(tri, z_min, z_max, material, axis)
            !isnothing(obj) && push!(objects, obj)
        end
    end
end

"""Check if a polygon (given as list of (x,y) tuples) is convex."""
function _is_convex(verts::Vector{Tuple{Float64,Float64}})::Bool
    n = length(verts)
    n < 3 && return false
    sign = 0
    for i in 1:n
        j = mod1(i + 1, n)
        k = mod1(i + 2, n)
        dx1 = verts[j][1] - verts[i][1]
        dy1 = verts[j][2] - verts[i][2]
        dx2 = verts[k][1] - verts[j][1]
        dy2 = verts[k][2] - verts[j][2]
        cross = dx1 * dy2 - dy1 * dx2
        if abs(cross) > 1e-15
            if sign == 0
                sign = cross > 0 ? 1 : -1
            elseif (cross > 0 ? 1 : -1) != sign
                return false
            end
        end
    end
    return true
end

"""Ear-clipping triangulation for simple (non-self-intersecting) polygons."""
function _ear_clip_triangulate(
    vertices::Vector{Tuple{Float64,Float64}},
)::Vector{Vector{Tuple{Float64,Float64}}}
    triangles = Vector{Tuple{Float64,Float64}}[]
    n = length(vertices)
    n < 3 && return triangles

    # Work with indices into a mutable list
    indices = collect(1:n)
    verts = copy(vertices)

    # Ensure CCW winding
    if _signed_area(verts) < 0
        reverse!(indices)
    end

    max_iter = n * n  # safety limit
    iter = 0
    while length(indices) > 3
        iter += 1
        iter > max_iter && break

        found_ear = false
        ni = length(indices)
        for i in 1:ni
            prev_i = mod1(i - 1, ni)
            next_i = mod1(i + 1, ni)
            a = verts[indices[prev_i]]
            b = verts[indices[i]]
            c = verts[indices[next_i]]

            # Check if this is a convex vertex (CCW: cross product > 0)
            cross = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
            cross <= 1e-15 && continue

            # Check no other vertex is inside this triangle
            is_ear = true
            for j in 1:ni
                j == prev_i && continue
                j == i && continue
                j == next_i && continue
                p = verts[indices[j]]
                if _point_in_triangle(p, a, b, c)
                    is_ear = false
                    break
                end
            end

            if is_ear
                push!(triangles, [a, b, c])
                deleteat!(indices, i)
                found_ear = true
                break
            end
        end

        # If no ear found, force remove a vertex (degenerate polygon)
        if !found_ear
            ni = length(indices)
            ni <= 3 && break
            prev_i = mod1(1, ni)
            next_i = mod1(3, ni)
            push!(triangles, [verts[indices[1]], verts[indices[2]], verts[indices[3]]])
            deleteat!(indices, 2)
        end
    end

    # Last triangle
    if length(indices) == 3
        push!(triangles, [verts[indices[1]], verts[indices[2]], verts[indices[3]]])
    end

    return triangles
end

function _signed_area(verts::Vector{Tuple{Float64,Float64}})::Float64
    n = length(verts)
    area = 0.0
    for i in 1:n
        j = mod1(i + 1, n)
        area += verts[i][1] * verts[j][2] - verts[j][1] * verts[i][2]
    end
    return area / 2.0
end

function _point_in_triangle(
    p::Tuple{Float64,Float64},
    a::Tuple{Float64,Float64},
    b::Tuple{Float64,Float64},
    c::Tuple{Float64,Float64},
)::Bool
    d1 = (p[1] - b[1]) * (a[2] - b[2]) - (a[1] - b[1]) * (p[2] - b[2])
    d2 = (p[1] - c[1]) * (b[2] - c[2]) - (b[1] - c[1]) * (p[2] - c[2])
    d3 = (p[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (p[2] - a[2])
    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0)
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0)
    return !(has_neg && has_pos)
end

"""Create a single Prism Object from convex vertices."""
function _make_prism(
    vertices::Vector{Tuple{Float64,Float64}},
    z_min::Real, z_max::Real,
    material::Material,
    axis::Int,
)::Union{Object,Nothing}
    K = length(vertices)
    K < 3 && return nothing

    # Skip degenerate polygons (near-zero area)
    area = _signed_area(vertices)
    abs(area) < 1e-20 && return nothing

    cx = sum(v[1] for v in vertices) / K
    cy = sum(v[2] for v in vertices) / K
    cz = (z_min + z_max) / 2.0
    height = Float64(z_max - z_min)

    # Build vertices relative to centroid, ensuring CCW winding
    # GeometryPrimitives requires CCW (positive signed area)
    ordered = area > 0 ? vertices : reverse(vertices)

    vert_data = Float64[]
    for (x, y) in ordered
        push!(vert_data, x - cx)
        push!(vert_data, y - cy)
    end

    verts_matrix = SMatrix{2,K,Float64}(vert_data...)

    local poly
    try
        poly = Polygon(verts_matrix)
    catch
        return nothing  # skip polygons that GeometryPrimitives rejects
    end

    I3 = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1)

    if axis == 3
        center = SVector(cx, cy, cz)
    elseif axis == 2
        center = SVector(cx, cz, cy)
    elseif axis == 1
        center = SVector(cz, cx, cy)
    else
        error("axis must be 1, 2, or 3")
    end

    prism = Prism(center, poly, height, I3)
    return Object(shape=prism, material=material)
end

# ────────────────────────────────────────────────────────────
# Utility: Layer inspection
# ────────────────────────────────────────────────────────────

"""
    get_layers(cell::GDSCell) -> Set{Tuple{Int,Int}}

Return the set of (layer, datatype) pairs present in a cell's polygons.
"""
function get_layers(cell::GDSCell)::Set{Tuple{Int,Int}}
    result = Set{Tuple{Int,Int}}()
    for poly in cell.polygons
        push!(result, (poly.layer, poly.datatype))
    end
    return result
end

"""
    get_cell_names(lib::GDSLibrary) -> Vector{String}

Return the names of all cells in the library.
"""
function get_cell_names(lib::GDSLibrary)::Vector{String}
    collect(keys(lib.cells))
end

# ────────────────────────────────────────────────────────────
# Internal Helpers
# ────────────────────────────────────────────────────────────

function _decode_string(payload::AbstractVector{UInt8})::String
    # Strip trailing null bytes and whitespace
    s = String(copy(payload))
    return rstrip(s, ['\0', ' '])
end

function _read_int16(payload::AbstractVector{UInt8}, offset::Int=1)::Int
    Int(reinterpret(Int16, [payload[offset+1], payload[offset]])[1])
end

function _read_uint16(payload::AbstractVector{UInt8}, offset::Int=1)::UInt16
    (UInt16(payload[offset]) << 8) | UInt16(payload[offset+1])
end

function _read_int32(payload::AbstractVector{UInt8}, offset::Int=1)::Int32
    b1, b2, b3, b4 = payload[offset], payload[offset+1], payload[offset+2], payload[offset+3]
    return reinterpret(Int32, [(b4), (b3), (b2), (b1)])[1]
end

function _read_xy(payload::AbstractVector{UInt8}, factor::Float64)::Vector{Tuple{Float64,Float64}}
    n_coords = div(length(payload), 4)
    n_points = div(n_coords, 2)
    points = Vector{Tuple{Float64,Float64}}(undef, n_points)
    for i in 1:n_points
        x_offset = (i - 1) * 8 + 1
        y_offset = x_offset + 4
        x = factor * Float64(_read_int32(payload, x_offset))
        y = factor * Float64(_read_int32(payload, y_offset))
        points[i] = (x, y)
    end
    return points
end

function _points_equal(a::Tuple{Float64,Float64}, b::Tuple{Float64,Float64})::Bool
    abs(a[1] - b[1]) < 1e-15 && abs(a[2] - b[2]) < 1e-15
end

# ────────────────────────────────────────────────────────────
# Exports
# ────────────────────────────────────────────────────────────

export read_gds, flatten_cell, gds_to_objects, gds_polygons_to_objects
export GDSLibrary, GDSCell, GDSPolygon, GDSReference
export get_layers, get_cell_names, decode_gdsii_real
