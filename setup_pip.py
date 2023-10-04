# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os


with open(os.path.join(os.path.dirname(__file__), 'ceml/VERSION')) as f:
      version = f.read().strip()

def readme():
    with open('README.rst') as f:
        return f.read()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()


setup(name='ceml',
      version=version,
      description='Counterfactuals for explaining machine learning models - A Python toolbox',
      long_description=readme(),
      keywords='machine learning counterfactual',
      url='https://github.com/andreArtelt/ceml',
      author='André Artelt',
      author_email='aartelt@techfak.uni-bielefeld.de',
      license='MIT',
      python_requires='>=3.8',
      install_requires=install_requires,
      packages=find_packages(),
      include_package_data=True,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
          ],
      zip_safe=False)
