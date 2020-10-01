#!/usr/bin/env python


"""Setup fast transformers"""

from itertools import dropwhile

from setuptools import find_packages
from setuptools import setup

__version__ = '0.0.1'


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def setup_package():
    # with open("README.rst") as f:
    #     long_description = f.read()
    meta = {
        "version": '0.0.1',
        "description": "temporal spatial market prediction",
        "maintainer": "hhh",
        "url": "",
        "email": "ravehun@gmail.com",
        "license": "..."
    }
    setup(

        name="ts-prediction",
        version=meta["version"],
        description=meta["description"],
        long_description="...",
        long_description_content_type="text/x-rst",
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
        ],
        packages=find_packages(exclude=['test']),
        install_requires=["pyarrow", "xarray", "cppimport"]
    )


if __name__ == "__main__":
    setup_package()
