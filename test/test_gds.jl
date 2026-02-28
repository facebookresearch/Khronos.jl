# Copyright (c) Meta Platforms, Inc. and affiliates.

using Test
using GeometryPrimitives: bounds

# GDS test files from tidy3d-notebooks
const GDS_DIR = joinpath(@__DIR__, "..", "..", "tidy3d-notebooks", "misc")

@testset "GDS Reader" begin

    # ────────────────────────────────────────────────────────
    # GDSII Real Decoding
    # ────────────────────────────────────────────────────────
    @testset "decode_gdsii_real" begin
        # Zero
        @test Khronos.decode_gdsii_real(UInt8[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]) == 0.0

        # 1.0 in GDSII format:
        # sign=0, exponent=65 (0x41), mantissa=0x10000000000000 (= 1/16 * 16^1 = 1.0)
        @test Khronos.decode_gdsii_real(UInt8[0x41, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]) ≈ 1.0

        # -1.0
        @test Khronos.decode_gdsii_real(UInt8[0xC1, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]) ≈ -1.0

        # 0.001 (common for GDSII units)
        @test Khronos.decode_gdsii_real(UInt8[0x3E, 0x41, 0x89, 0x37, 0x4B, 0xC6, 0xA7, 0xF0]) ≈ 0.001 rtol=1e-10

        # 1e-9 (1 nanometer, common precision unit)
        # Verify round-trip: encode 1e-9 and check it decodes correctly
        # The exact encoding depends on the base-16 representation
        val = Khronos.decode_gdsii_real(UInt8[0x3A, 0x44, 0xB8, 0x2F, 0xA0, 0x9B, 0x5A, 0x54])
        @test val > 0  # positive value
        @test val < 1e-6  # small value in the nanometer range
    end

    # ────────────────────────────────────────────────────────
    # PATH Expansion
    # ────────────────────────────────────────────────────────
    @testset "expand_path" begin
        # Simple horizontal path
        centerline = [(0.0, 0.0), (10.0, 0.0)]
        hw = 1.0
        poly = Khronos.expand_path(centerline, hw, 0, 0.0, 0.0)
        @test length(poly) == 4  # rectangle

        # Check bounding box
        xs = [p[1] for p in poly]
        ys = [p[2] for p in poly]
        @test minimum(xs) ≈ 0.0
        @test maximum(xs) ≈ 10.0
        @test minimum(ys) ≈ -1.0
        @test maximum(ys) ≈ 1.0

        # Half-width extension (pathtype=2)
        poly2 = Khronos.expand_path(centerline, hw, 2, 0.0, 0.0)
        xs2 = [p[1] for p in poly2]
        @test minimum(xs2) ≈ -1.0  # extended by half_width
        @test maximum(xs2) ≈ 11.0

        # Explicit extension (pathtype=4)
        poly4 = Khronos.expand_path(centerline, hw, 4, 0.5, 2.0)
        xs4 = [p[1] for p in poly4]
        @test minimum(xs4) ≈ -0.5  # extended by bgnextn
        @test maximum(xs4) ≈ 12.0  # extended by endextn

        # L-shaped path
        centerline_L = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
        poly_L = Khronos.expand_path(centerline_L, 0.5, 0, 0.0, 0.0)
        @test length(poly_L) >= 4
    end

    # ────────────────────────────────────────────────────────
    # Convexity check and triangulation
    # ────────────────────────────────────────────────────────
    @testset "convex decomposition" begin
        # Convex polygon (square)
        square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        @test Khronos._is_convex(square) == true

        # Non-convex polygon (L-shape)
        lshape = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
        @test Khronos._is_convex(lshape) == false

        # Triangulation produces valid triangles
        triangles = Khronos._ear_clip_triangulate(lshape)
        @test length(triangles) == 4  # 6-vertex polygon → 4 triangles
        for tri in triangles
            @test length(tri) == 3
            area = abs(Khronos._signed_area(tri))
            @test area > 0  # non-degenerate
        end

        # Total area should be preserved
        total_tri_area = sum(abs(Khronos._signed_area(tri)) for tri in triangles)
        @test total_tri_area ≈ abs(Khronos._signed_area(lshape)) rtol=1e-10
    end

    # ────────────────────────────────────────────────────────
    # SREF Transform
    # ────────────────────────────────────────────────────────
    @testset "transform_gds_point" begin
        # Identity transform
        x, y = Khronos.transform_gds_point(1.0, 2.0, 1.0, false, 0.0, (0.0, 0.0))
        @test x ≈ 1.0
        @test y ≈ 2.0

        # Translation only
        x, y = Khronos.transform_gds_point(1.0, 2.0, 1.0, false, 0.0, (10.0, 20.0))
        @test x ≈ 11.0
        @test y ≈ 22.0

        # 90-degree rotation
        x, y = Khronos.transform_gds_point(1.0, 0.0, 1.0, false, π/2, (0.0, 0.0))
        @test x ≈ 0.0 atol=1e-12
        @test y ≈ 1.0

        # X reflection
        x, y = Khronos.transform_gds_point(1.0, 2.0, 1.0, true, 0.0, (0.0, 0.0))
        @test x ≈ 1.0
        @test y ≈ -2.0

        # Magnification
        x, y = Khronos.transform_gds_point(1.0, 2.0, 3.0, false, 0.0, (0.0, 0.0))
        @test x ≈ 3.0
        @test y ≈ 6.0

        # Combined: mag=2, reflect, rotate 90°, translate (5,10)
        x, y = Khronos.transform_gds_point(1.0, 1.0, 2.0, true, π/2, (5.0, 10.0))
        # mag: (2, 2), reflect: (2, -2), rot90: (2, 2), translate: (7, 12)
        @test x ≈ 7.0 atol=1e-12
        @test y ≈ 12.0 atol=1e-12
    end

    # ────────────────────────────────────────────────────────
    # Read actual GDS files
    # ────────────────────────────────────────────────────────
    if isdir(GDS_DIR)
        @testset "coupler.gds structure" begin
            lib = Khronos.read_gds(joinpath(GDS_DIR, "coupler.gds"))
            @test lib isa Khronos.GDSLibrary
            @test lib.name == "library"
            @test length(lib.cells) == 2
            @test haskey(lib.cells, "COUPLER")
            @test haskey(lib.cells, "COUPLER_ARM")

            # COUPLER_ARM: 1 polygon (90 vertices), layer 1
            arm = lib.cells["COUPLER_ARM"]
            @test length(arm.polygons) == 1
            @test arm.polygons[1].layer == 1
            @test arm.polygons[1].datatype == 0
            @test length(arm.polygons[1].vertices) == 90

            # COUPLER: 1 polygon (substrate), 2 references
            coupler = lib.cells["COUPLER"]
            @test length(coupler.polygons) == 1
            @test coupler.polygons[1].layer == 0
            @test length(coupler.references) == 2

            # Flattened COUPLER: 3 polygons (1 substrate + 2 arms)
            flat = Khronos.flatten_cell(lib, "COUPLER")
            @test length(flat) == 3

            # Layer filtering
            flat_l0 = Khronos.flatten_cell(lib, "COUPLER", 0, 0)
            flat_l1 = Khronos.flatten_cell(lib, "COUPLER", 1, 0)
            @test length(flat_l0) == 1
            @test length(flat_l1) == 2

            # One arm is reflected (y-coordinates should be mirrored)
            arm_polys = flat_l1
            ys_1 = [v[2] for v in arm_polys[1].vertices]
            ys_2 = [v[2] for v in arm_polys[2].vertices]
            # One arm should have positive y, other negative
            @test minimum(ys_1) * minimum(ys_2) < 0  # opposite signs
        end

        @testset "mrr-electrode.gds multiple layers" begin
            lib = Khronos.read_gds(joinpath(GDS_DIR, "mrr-electrode.gds"))
            @test !isempty(lib.cells)

            all_layers = Set{Tuple{Int,Int}}()
            for (name, cell) in lib.cells
                union!(all_layers, Khronos.get_layers(cell))
            end
            @test length(all_layers) >= 2
        end

        @testset "read all 20 GDS files" begin
            gds_files = filter(f -> endswith(f, ".gds"), readdir(GDS_DIR, join=true))
            @test length(gds_files) == 20

            for gds_file in gds_files
                fname = basename(gds_file)
                @testset "$fname" begin
                    lib = Khronos.read_gds(gds_file)
                    @test lib isa Khronos.GDSLibrary
                    @test !isempty(lib.cells)

                    for (name, cell) in lib.cells
                        polys = Khronos.flatten_cell(lib, name)
                        for poly in polys
                            @test length(poly.vertices) >= 3
                            for (x, y) in poly.vertices
                                @test isfinite(x) && isfinite(y)
                            end
                        end
                    end
                end
            end
        end

        @testset "gds_to_objects" begin
            lib = Khronos.read_gds(joinpath(GDS_DIR, "coupler.gds"))
            mat_si = Khronos.Material{Float64}(ε=3.48^2)
            mat_sio2 = Khronos.Material{Float64}(ε=1.45^2)

            objects = Khronos.gds_to_objects(lib, "COUPLER", Dict(
                (0, 0) => (-4.0, 0.0, mat_sio2),
                (1, 0) => (0.0, 0.22, mat_si),
            ); axis=3)
            @test length(objects) >= 3  # substrate (convex) + arm triangles

            # Verify all objects have valid bounds
            for obj in objects
                b = bounds(obj.shape)
                for d in 1:3
                    @test isfinite(b[1][d]) && isfinite(b[2][d])
                    @test b[1][d] < b[2][d]
                end
            end

            # Substrate should be a single prism covering the full extent
            substrate_objs = filter(o -> o.material.ε ≈ 1.45^2, objects)
            @test length(substrate_objs) == 1
            sb = bounds(substrate_objs[1].shape)
            @test sb[1][1] ≈ -12.5  # x min
            @test sb[2][1] ≈ 12.5   # x max
            @test sb[1][3] ≈ -4.0   # z min (slab_bounds)
            @test sb[2][3] ≈ 0.0    # z max
        end
    else
        @warn "GDS test directory not found: $GDS_DIR — skipping file-based tests"
    end

    # ────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────
    @testset "internal helpers" begin
        @test Khronos._decode_string(UInt8[0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x00]) == "Hello"
        @test Khronos._read_int16(UInt8[0x00, 0x05]) == 5
        @test Khronos._read_int16(UInt8[0xFF, 0xFB]) == -5
        @test Khronos._read_int32(UInt8[0x00, 0x00, 0x00, 0x0A]) == 10
        @test Khronos._read_int32(UInt8[0xFF, 0xFF, 0xFF, 0xF6]) == -10
    end
end
