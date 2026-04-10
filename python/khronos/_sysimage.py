# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Build and manage Khronos.jl PackageCompiler system images.

Usage:
    khronos-build-sysimage          # CLI entry point
    python -m khronos._sysimage     # alternative

    from khronos._sysimage import build_sysimage
    build_sysimage()                # programmatic
"""

import os
import platform
import subprocess
import shutil


def get_sysimage_dir():
    """Return the default sysimage directory (~/.khronos/sysimage/)."""
    return os.path.join(os.path.expanduser("~"), ".khronos", "sysimage")


def get_sysimage_path():
    """Return the default sysimage file path."""
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    return os.path.join(get_sysimage_dir(), f"khronos_sysimage{ext}")


def build_sysimage(output_path=None, julia_executable=None, verbose=True):
    """Build a PackageCompiler sysimage for Khronos.jl.

    Parameters
    ----------
    output_path : str, optional
        Where to write the sysimage. Default: ~/.khronos/sysimage/khronos_sysimage.so
    julia_executable : str, optional
        Path to julia binary. Default: auto-detect from juliacall/juliapkg.
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        Path to the built sysimage.
    """
    from ._bridge import _find_julia_project

    julia_project, _ = _find_julia_project()
    if output_path is None:
        output_path = get_sysimage_path()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Find the precompile workload
    workload_path = os.path.join(julia_project, "precompile", "workload.jl")
    if not os.path.isfile(workload_path):
        # Try the bundled location relative to this module
        workload_path = os.path.join(
            os.path.dirname(__file__), "julia_src", "precompile", "workload.jl"
        )
    if not os.path.isfile(workload_path):
        raise FileNotFoundError(
            f"Precompile workload not found. Looked in:\n"
            f"  {os.path.join(julia_project, 'precompile', 'workload.jl')}\n"
            f"  {workload_path}"
        )

    # Find Julia executable
    if julia_executable is None:
        julia_executable = _find_julia()

    if verbose:
        print(f"Building Khronos sysimage...")
        print(f"  Julia project: {julia_project}")
        print(f"  Julia binary:  {julia_executable}")
        print(f"  Workload:      {workload_path}")
        print(f"  Output:        {output_path}")
        print()
        print("This may take 5-15 minutes on first build.")
        print()

    # Escape paths for Julia string literals
    julia_project_esc = julia_project.replace("\\", "\\\\").replace('"', '\\"')
    output_path_esc = output_path.replace("\\", "\\\\").replace('"', '\\"')
    workload_path_esc = workload_path.replace("\\", "\\\\").replace('"', '\\"')

    # Build the sysimage via a Julia script
    build_script = f"""
using Pkg
Pkg.activate("{julia_project_esc}")
Pkg.instantiate()

using PackageCompiler

create_sysimage(
    [:Khronos, :GeometryPrimitives];
    sysimage_path="{output_path_esc}",
    precompile_execution_file="{workload_path_esc}",
    project="{julia_project_esc}",
)
"""

    result = subprocess.run(
        [julia_executable, f"--project={julia_project}", "-e", build_script],
        capture_output=not verbose,
        text=True,
    )

    if result.returncode != 0:
        if not verbose:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        # Clean up lock file on failure
        lock_file = os.path.join(os.path.dirname(output_path), ".build_in_progress")
        if os.path.isfile(lock_file):
            os.remove(lock_file)
        raise RuntimeError(
            f"Sysimage build failed with exit code {result.returncode}"
        )

    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nSysimage built successfully!")
        print(f"  Path: {output_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"\nKhronos will automatically use this sysimage on next import.")

    # Clean up build lock file if it exists
    lock_file = os.path.join(os.path.dirname(output_path), ".build_in_progress")
    if os.path.isfile(lock_file):
        os.remove(lock_file)

    return output_path


def remove_sysimage():
    """Remove the default sysimage."""
    path = get_sysimage_path()
    if os.path.isfile(path):
        os.remove(path)
        print(f"Removed sysimage: {path}")
    else:
        print(f"No sysimage found at: {path}")


def sysimage_info():
    """Print information about the current sysimage."""
    path = get_sysimage_path()
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        mtime = os.path.getmtime(path)
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f"Sysimage found:")
        print(f"  Path:    {path}")
        print(f"  Size:    {size_mb:.1f} MB")
        print(f"  Built:   {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"No sysimage found at default location: {path}")
        print(f"Run 'khronos-build-sysimage' to build one.")


def _find_julia():
    """Find the Julia executable, preferring juliapkg's managed install."""
    # Try juliapkg first (what juliacall uses)
    try:
        import juliapkg
        return juliapkg.executable()
    except (ImportError, Exception):
        pass

    # Try PATH
    julia_path = shutil.which("julia")
    if julia_path:
        return julia_path

    raise RuntimeError(
        "Cannot find Julia executable. Install juliacall first:\n"
        "  pip install juliacall\n"
        "  python -c 'import juliacall'  # triggers Julia installation"
    )


def build_sysimage_cli():
    """CLI entry point for khronos-build-sysimage."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Build a PackageCompiler sysimage for Khronos.jl"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=f"Output path (default: {get_sysimage_path()})"
    )
    parser.add_argument(
        "--julia",
        default=None,
        help="Path to Julia executable (default: auto-detect)"
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove existing sysimage instead of building"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print sysimage information"
    )
    args = parser.parse_args()

    if args.info:
        sysimage_info()
    elif args.remove:
        remove_sysimage()
    else:
        build_sysimage(output_path=args.output, julia_executable=args.julia)


if __name__ == "__main__":
    build_sysimage_cli()
