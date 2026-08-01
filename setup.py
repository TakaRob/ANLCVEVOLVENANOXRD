"""Setuptools build customization for non-runtime source modules."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class RuntimeBuildPy(build_py):
    """Leave the maintenance-only regrid script out of distributions."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package == "xrd_app":
            modules = [module for module in modules if module[1] != "regrid_bins"]
        return modules

    def run(self):
        super().run()
        stale = Path(self.build_lib) / "xrd_app" / "regrid_bins.py"
        if stale.exists():
            stale.unlink()


setup(cmdclass={"build_py": RuntimeBuildPy})
