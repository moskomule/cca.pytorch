import sys
import warnings

from setuptools import setup

if sys.version_info[1] < 7:
    warnings.warn("This library may require Python 3.7")

with open("README.md") as f:
    readme = f.read()

setup(name="cca",
      version="0.1",
      author="moskomule",
      author_email="hataya@nlab.jp",
      packages=["cca"],
      url="https://github.com/moskomule/cca.pytorch",
      description="Look into the representation of DNNs",
      long_description=readme,
      license="BSD",
      )
