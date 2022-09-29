"""Setup script."""

import os
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension

# Use Cython if available.
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

include_dirs = []
try:
    import numpy

    include_dirs.append(numpy.get_include())
except ImportError:
    print("Numpy and its headers are required to run setup(). Exiting.")
    sys.exit(1)


# Parse the version from the fiona module.
with open("rio_trend/__init__.py") as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            break

long_description = """"""


def read(fname):
    """Read a file's contents."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if cythonize and "clean" not in sys.argv:
    ext_modules = cythonize(
        [
            Extension(
                "rio_trend.colorspace",
                ["rio_trend/colorspace.pyx"],
                extra_compile_args=["-O2"],
            )
        ]
    )
else:
    ext_modules = [Extension("rio_trend.colorspace", ["rio_trend/colorspace.c"])]

inst_reqs = [
    "click>=4.0",
    "rasterio~=1.0",
    "rio-mucho",
    "enum34 ; python_version < '3.4'",
]

setup(
    name="rio-trend",
    version=version,
    description=u"",
    long_description=long_description,
    classifiers=[
        "Topic :: Scientific/Engineering :: GIS",
    ],
    keywords="",
    author=u"",
    author_email="",
    url="",
    license="BSD",
    packages=find_packages(exclude=["ez_setup", "examples", "tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=inst_reqs,
    ext_modules=ext_modules,
    include_dirs=include_dirs,
    # extras_require={"test": ["pytest", "colormath==2.0.2", "pytest-cov", "codecov"]},
    entry_points="""
    [rasterio.rio_plugins]
    trend=rio_trend.scripts.cli:trend
    """,
)
