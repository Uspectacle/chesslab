"""Script to compile Python engines to standalone executables.

Uses PyInstaller to create .exe files for engines that can be run
as UCI chess engines without requiring Python installation.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()


class EngineCompiler:
    """Compiles Python chess engines to standalone executables using PyInstaller."""

    def __init__(self, project_root: Path):
        """Initialize compiler.

        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.engines_dir = project_root / "engines"
        self.dist_dir = self.engines_dir / "dist"

        # Ensure dist directory exists
        self.dist_dir.mkdir(exist_ok=True)

        logger.info(
            "Engine compiler initialized",
            project_root=str(project_root),
            dist_dir=str(self.dist_dir),
        )

    def compile_engine(
        self,
        engine_file: str,
        additional_files: Optional[List[str]] = None,
        hidden_imports: Optional[List[str]] = None,
    ) -> bool:
        """Compile a single engine to executable.

        Args:
            engine_file: Name of engine file (e.g., 'random_engine.py')
            additional_files: Additional files to include
            hidden_imports: Hidden imports to include

        Returns:
            True if compilation succeeded, False otherwise
        """
        engine_path = self.engines_dir / engine_file

        if not engine_path.exists():
            logger.error("Engine file not found", path=str(engine_path))
            return False

        logger.info("Compiling engine", engine=engine_file)

        # Build PyInstaller command
        cmd = [
            "pyinstaller",
            "--onefile",  # Single executable
            "--clean",  # Clean cache
            f"--distpath={self.dist_dir}",  # Output directory
            f"--workpath={self.engines_dir / 'build'}",  # Build directory
            f"--specpath={self.engines_dir}",  # Spec file directory
            "--log-level=WARN",  # Reduce output noise
        ]

        # Add hidden imports
        if hidden_imports:
            for imp in hidden_imports:
                cmd.append(f"--hidden-import={imp}")

        # Add additional data files
        if additional_files:
            for file in additional_files:
                cmd.append(f"--add-data={file}")

        # Add the engine file
        cmd.append(str(engine_path))

        try:
            subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=True,
            )

            # Get output executable name
            exe_name = engine_path.stem
            if sys.platform == "win32":
                exe_name += ".exe"

            exe_path = self.dist_dir / exe_name

            if exe_path.exists():
                logger.info(
                    "Engine compiled successfully",
                    engine=engine_file,
                    output=str(exe_path),
                )
                return True
            else:
                logger.error(
                    "Compilation succeeded but executable not found",
                    expected=str(exe_path),
                )
                return False

        except subprocess.CalledProcessError as e:
            logger.error("Compilation failed", engine=engine_file, error=e.stderr)
            return False

    def compile_all_engines(self) -> Dict[str, bool]:
        """Compile all supported engines.

        Returns:
            Dictionary mapping engine name to compilation success status
        """
        engines_to_compile = {
            "random_engine.py": {"hidden_imports": ["chess"]},
            "worstfish.py": {"hidden_imports": ["chess", "chess.engine"]},
        }

        results: Dict[str, bool] = {}

        for engine_file, config in engines_to_compile.items():
            success = self.compile_engine(
                engine_file, hidden_imports=config.get("hidden_imports")
            )
            results[engine_file] = success

        # Print summary
        print("\n=== Compilation Summary ===")
        for engine, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{status}: {engine}")

        success_count = sum(results.values())
        print(f"\nCompiled {success_count}/{len(results)} engines successfully")

        return results

    def clean_build_artifacts(self):
        """Remove build artifacts and temporary files."""
        logger.info("Cleaning build artifacts")

        # Remove build directory
        build_dir = self.engines_dir / "build"
        if build_dir.exists():
            import shutil

            shutil.rmtree(build_dir)
            logger.info("Removed build directory", path=str(build_dir))

        # Remove spec files
        for spec_file in self.engines_dir.glob("*.spec"):
            spec_file.unlink()
            logger.info("Removed spec file", path=str(spec_file))


def main():
    """Main compilation script entry point."""
    import logging

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    # Get project root (assuming script is in scripts/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("ChessLab Engine Compiler")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Engines directory: {project_root / 'engines'}")
    print(f"Output directory: {project_root / 'engines' / 'dist'}")
    print()

    # Check if PyInstaller is installed
    try:
        result = subprocess.run(
            ["pyinstaller", "--version"], capture_output=True, text=True, check=True
        )
        print(f"PyInstaller version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: PyInstaller not found!")
        print("Install with: pip install pyinstaller")
        sys.exit(1)

    print()

    # Create compiler and compile engines
    compiler = EngineCompiler(project_root)
    results = compiler.compile_all_engines()

    # Clean up build artifacts
    print("\nCleaning build artifacts...")
    compiler.clean_build_artifacts()

    # Exit with error code if any compilation failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
