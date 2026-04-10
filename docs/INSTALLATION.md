# Khronos: Installation & Sysimage Guide

## Python Quick Start

```bash
pip install khronos
```

That's it. On first use, Khronos will automatically:
1. Download and install Julia (via juliacall/JuliaUp)
2. Resolve Julia package dependencies
3. Build a system image in the background for fast future startups

```python
import khronos.meep as mp

sim = mp.Simulation(
    cell_size=mp.Vector3(16, 8, 0),
    resolution=10,
    boundary_layers=[mp.PML(1.0)],
    sources=[mp.Source(mp.GaussianSource(frequency=0.15, fwidth=0.1),
                       component=mp.Ez, center=mp.Vector3(-5, 0))],
    geometry=[mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
                       material=mp.Medium(epsilon=12))],
)
sim.run(until=200)
```

## Julia Quick Start

```bash
git clone https://github.com/your-org/Khronos.jl
cd Khronos.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

```julia
import Khronos
using GeometryPrimitives

Khronos.choose_backend(Khronos.CUDADevice(), Float32)

sim = Khronos.Simulation(
    cell_size = [12.0f0, 6.0f0, 0.0f0],
    cell_center = [0.0f0, 0.0f0, 0.0f0],
    resolution = 20,
    geometry = [
        Khronos.Object(
            shape = Cuboid([0.0, 0.0, 0.0], [10.0, 0.5, 1e-6]),
            material = Khronos.Material(ε = 12.0),
        ),
    ],
    sources = [
        Khronos.UniformSource(
            time_profile = Khronos.GaussianPulseSource(fcen = 0.15, fwidth = 0.1),
            component = Khronos.Ez(),
            center = [-4.0, 0.0, 0.0],
            size = [0.0, 2.0, 0.0],
        ),
    ],
    monitors = [
        Khronos.FluxMonitor(center = [4.0, 0.0, 0.0], size = [0.0, 2.0, 0.0],
                             frequencies = [0.1, 0.15, 0.2]),
    ],
    boundaries = [[1.0f0, 1.0f0], [1.0f0, 1.0f0], [0.0f0, 0.0f0]],
)

Khronos.run(sim, until = 200.0)
flux = Khronos.get_flux(sim.monitors[1].monitor_data)
```

See [JULIA_API.md](JULIA_API.md) and [PYTHON_API.md](PYTHON_API.md) for full API references.

## How It Works

### Package Structure

When installed via pip, the package bundles:
- Python wrapper code (`khronos/meep/`)
- Julia source code (`khronos/julia_src/src/`, `khronos/julia_src/Project.toml`)
- Precompile workload (`khronos/julia_src/precompile/workload.jl`)

### Startup Sequence

On the first call to any function that needs Julia (e.g., `sim.run()`):

1. **Julia bootstrap** — `juliacall` checks if Julia is installed. If not, it downloads and installs it via JuliaUp. This is a one-time ~200MB download.

2. **Dependency resolution** — `Pkg.instantiate()` resolves and precompiles all Julia dependencies (GeometryPrimitives, KernelAbstractions, CUDA.jl, etc.). One-time, takes 2–5 minutes.

3. **Module loading** — `import Khronos` loads the FDTD solver. Without a sysimage, Julia JIT-compiles code on first use, which adds latency.

4. **Auto sysimage build** — After successful initialization, a background process starts building a PackageCompiler sysimage. This takes 5–15 minutes but does **not** block your current session.

On subsequent imports, the sysimage is detected and loaded automatically, reducing startup from minutes to seconds.

### Startup Timeline

| Session | What happens | Startup time |
|---------|-------------|-------------|
| 1st ever | Julia install + deps + JIT (sysimage building in background) | ~5 min |
| 2nd (if sysimage ready) | Loads sysimage | ~10 sec |
| 3rd+ | Loads sysimage | ~10 sec |

## System Image

The system image (sysimage) is a pre-compiled snapshot of Julia code that eliminates JIT compilation overhead. It is stored at:

```
~/.khronos/sysimage/khronos_sysimage.so      # Linux
~/.khronos/sysimage/khronos_sysimage.dylib    # macOS
```

### Auto-Build (Default)

For end users (`pip install khronos`), the sysimage is built automatically in the background on first use. No action required.

The build log is written to `~/.khronos/sysimage/build.log`.

### Manual Build

```bash
khronos-build-sysimage             # build (or rebuild)
khronos-build-sysimage --info      # check status
khronos-build-sysimage --remove    # delete sysimage
khronos-build-sysimage --output /path/to/sysimage.so
khronos-build-sysimage --julia /usr/local/bin/julia
```

Or from Python:

```python
import khronos
khronos.build_sysimage()       # build
khronos.sysimage_info()        # check status
```

### Julia Sysimage (without Python)

```julia
using Pkg; Pkg.activate("/path/to/Khronos.jl")
using PackageCompiler
create_sysimage(
    [:Khronos, :GeometryPrimitives];
    sysimage_path = "khronos_sysimage.so",
    precompile_execution_file = "python/khronos/julia_src/precompile/workload.jl",
    project = "/path/to/Khronos.jl",
)
```

```bash
julia --sysimage=khronos_sysimage.so --project=/path/to/Khronos.jl my_script.jl
```

### When to Rebuild

Rebuild the sysimage after updating the `khronos` pip package, Julia itself, or Julia dependencies:

```bash
khronos-build-sysimage --remove && khronos-build-sysimage
```

## Development Mode

```bash
cd Khronos.jl/python
pip install -e .
```

In development mode:
- The bridge detects the git checkout and uses Julia source files directly from the repo
- **Auto sysimage build is disabled** — sysimage would go stale with code changes
- Julia's built-in precompilation cache still works, so incremental changes are fast
- You can still manually build a sysimage for benchmarking: `khronos-build-sysimage`

### Building a Distributable Package

```bash
./scripts/sync_julia_src.sh        # copy Julia source into Python package tree
cd python && python -m build       # build wheel/sdist
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KHRONOS_JULIA_PROJECT` | Override Julia project path | Auto-detected |
| `KHRONOS_SYSIMAGE` | Override sysimage file path | `~/.khronos/sysimage/khronos_sysimage.so` |
| `KHRONOS_NO_AUTO_SYSIMAGE` | Set to `1` to disable auto sysimage build | Not set (auto-build enabled) |

## Troubleshooting

### Sysimage build failed

```bash
cat ~/.khronos/sysimage/build.log
```

### Stale sysimage after update

```bash
khronos-build-sysimage --remove && khronos-build-sysimage
```

### Build stuck (lock file)

```bash
rm ~/.khronos/sysimage/.build_in_progress
khronos-build-sysimage
```

The lock file auto-expires after 1 hour.

### Force no sysimage

```bash
KHRONOS_SYSIMAGE=/dev/null python my_script.py
```

### Julia not found

```bash
pip install juliacall
python -c "import juliacall"   # triggers Julia installation
```
