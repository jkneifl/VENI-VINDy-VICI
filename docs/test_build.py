#!/usr/bin/env python3
"""
Test script to verify Sphinx documentation builds without errors.
Run this before committing documentation changes.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run documentation build tests."""
    docs_dir = Path(__file__).parent

    print("🔍 Testing Sphinx documentation build...\n")

    # Clean previous builds
    print("1️⃣ Cleaning previous builds...")
    result = subprocess.run(
        ["make", "clean"],
        cwd=docs_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ Clean failed!")
        print(result.stderr)
        return 1
    print("✅ Clean successful\n")

    # Build HTML documentation
    print("2️⃣ Building HTML documentation...")
    result = subprocess.run(
        ["make", "html"],
        cwd=docs_dir,
        capture_output=True,
        text=True
    )

    # Check for errors
    if result.returncode != 0:
        print("❌ Build failed!")
        print(result.stderr)
        print(result.stdout)
        return 1

    # Check for warnings
    warnings = [line for line in result.stderr.split('\n') if 'WARNING' in line or 'ERROR' in line]

    if warnings:
        print("⚠️  Build completed with warnings:")
        for warning in warnings:
            print(f"  {warning}")
        print("\n💡 Fix these warnings before deploying if possible.")
    else:
        print("✅ Build successful with no warnings!")

    print("\n📖 Documentation built successfully!")
    print(f"📁 Location: {docs_dir / 'build' / 'html' / 'index.html'}")
    print("\n🌐 To view locally, run:")
    print(f"   open {docs_dir / 'build' / 'html' / 'index.html'}")

    return 0  # Return 0 even with warnings (they're often not critical)

if __name__ == "__main__":
    sys.exit(main())
