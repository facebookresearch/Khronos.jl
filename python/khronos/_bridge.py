# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Julia bridge: lazy initialization of the Khronos.jl runtime.

The Julia process starts on first access to `get_khronos()`, not at import time.
Supports both development (git checkout) and installed (pip) modes.
Optionally loads a pre-built sysimage for fast startup.
"""
import os

_jl = None
_Khronos = None
_GP = None  # GeometryPrimitives
_dev_mode = False  # True when running from a git checkout


def _find_julia_project():
    """Locate the Khronos.jl Julia project directory.

    Strategy:
    1. If KHRONOS_JULIA_PROJECT env var is set, use that.
    2. Check for development mode: repo root has Project.toml with name="Khronos".
    3. Check for installed mode: khronos/julia_src/Project.toml exists.

    Returns (path, is_dev_mode) tuple.
    """
    # 1. Explicit override
    env_path = os.environ.get("KHRONOS_JULIA_PROJECT")
    if env_path:
        env_path = os.path.abspath(env_path)
        if os.path.isfile(os.path.join(env_path, "Project.toml")):
            return env_path, True  # Treat explicit override as dev mode
        raise RuntimeError(
            f"KHRONOS_JULIA_PROJECT={env_path} does not contain Project.toml"
        )

    # 2. Development mode: walk up from this file to find repo root
    #    This file is at python/khronos/_bridge.py
    #    Repo root is at python/../  (i.e., 2 levels up from this file)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    dev_root = os.path.normpath(os.path.join(this_dir, "..", ".."))
    dev_project = os.path.join(dev_root, "Project.toml")
    if os.path.isfile(dev_project):
        try:
            with open(dev_project) as f:
                content = f.read()
                if 'name = "Khronos"' in content:
                    return dev_root, True
        except OSError:
            pass

    # 3. Installed mode: julia_src/ is a subpackage
    try:
        from importlib.resources import files
        julia_src_path = str(files("khronos.julia_src"))
        if os.path.isfile(os.path.join(julia_src_path, "Project.toml")):
            return julia_src_path, False
    except (ImportError, TypeError, FileNotFoundError, ModuleNotFoundError):
        pass

    # 3b. Fallback: __file__-based lookup
    fallback = os.path.join(this_dir, "julia_src")
    if os.path.isfile(os.path.join(fallback, "Project.toml")):
        return fallback, False

    raise RuntimeError(
        "Cannot find Khronos.jl Julia project. Set KHRONOS_JULIA_PROJECT "
        "env var or reinstall the khronos package."
    )


def _find_sysimage():
    """Locate a pre-built sysimage, or return None.

    Search order:
    1. KHRONOS_SYSIMAGE env var (explicit path)
    2. ~/.khronos/sysimage/khronos_sysimage.so (Linux)
       ~/.khronos/sysimage/khronos_sysimage.dylib (macOS)
    """
    # 1. Explicit override
    env_path = os.environ.get("KHRONOS_SYSIMAGE")
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        print(f"WARNING: KHRONOS_SYSIMAGE={env_path} not found, ignoring.")
        return None

    # 2. Default location
    import platform
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    default_path = os.path.join(
        os.path.expanduser("~"), ".khronos", "sysimage", f"khronos_sysimage{ext}"
    )
    if os.path.isfile(default_path):
        return default_path

    return None


def _auto_build_sysimage():
    """Build a sysimage in a background process after first successful init.

    This runs as a subprocess so it doesn't block the current session.
    The sysimage will be available on the next import.
    """
    import subprocess
    import sys

    # Don't auto-build if disabled via env var
    if os.environ.get("KHRONOS_NO_AUTO_SYSIMAGE", "").lower() in ("1", "true", "yes"):
        return

    # Mark that a build is in progress (prevent concurrent builds)
    from ._sysimage import get_sysimage_dir, get_sysimage_path
    lock_file = os.path.join(get_sysimage_dir(), ".build_in_progress")
    os.makedirs(get_sysimage_dir(), exist_ok=True)

    if os.path.isfile(lock_file):
        # Check if stale (older than 1 hour)
        import time
        age = time.time() - os.path.getmtime(lock_file)
        if age < 3600:
            return  # Build already in progress
        # Stale lock, remove it
        os.remove(lock_file)

    # Create lock file
    with open(lock_file, "w") as f:
        f.write(str(os.getpid()))

    sysimage_path = get_sysimage_path()
    print(f"\nKhronos: Building sysimage in the background for faster future imports...")
    print(f"  This is a one-time operation. Output: {sysimage_path}")
    print(f"  To check progress: khronos-build-sysimage --info")
    print(f"  To disable: set KHRONOS_NO_AUTO_SYSIMAGE=1\n")

    # Build in a background subprocess so it doesn't block the user
    build_cmd = [
        sys.executable, "-m", "khronos._sysimage",
    ]
    try:
        subprocess.Popen(
            build_cmd,
            stdout=open(os.path.join(get_sysimage_dir(), "build.log"), "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach from parent process
        )
    except Exception as e:
        print(f"Khronos: Auto sysimage build failed to start: {e}")
        print(f"  You can build manually: khronos-build-sysimage")
        if os.path.isfile(lock_file):
            os.remove(lock_file)


def _init_julia():
    global _jl, _Khronos, _GP, _dev_mode
    if _jl is not None:
        return

    sysimage_path = _find_sysimage()
    julia_project, _dev_mode = _find_julia_project()

    # Configure juliacall to use sysimage BEFORE importing juliacall.
    # juliacall reads PYTHON_JULIACALL_SYSIMAGE at import time.
    if sysimage_path:
        os.environ["PYTHON_JULIACALL_SYSIMAGE"] = sysimage_path
        print(f"Khronos: Using sysimage {sysimage_path}")

    import juliacall
    _jl = juliacall.newmodule("KhronosWrapper")

    # Activate the Khronos project
    _jl.seval(f'using Pkg; Pkg.activate("{julia_project}")')

    # If no sysimage, instantiate deps on first run
    if not sysimage_path:
        _jl.seval("Pkg.instantiate()")

    _jl.seval("import Khronos")
    _jl.seval("using GeometryPrimitives")
    _Khronos = _jl.Khronos
    _GP = _jl.seval("GeometryPrimitives")

    # Auto-build sysimage in background if:
    #  - no sysimage exists
    #  - NOT in dev mode (sysimage would go stale with code changes)
    if not sysimage_path and not _dev_mode:
        try:
            _auto_build_sysimage()
        except Exception:
            pass  # Non-critical — don't break the user's session


def get_jl():
    """Get the Julia module for direct evaluation."""
    _init_julia()
    return _jl


def get_khronos():
    """Get the Khronos Julia module."""
    _init_julia()
    return _Khronos


def get_gp():
    """Get the GeometryPrimitives Julia module."""
    _init_julia()
    return _GP
