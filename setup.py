#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', 'xlrd', 'scipy', 'numpy', 'pandas']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Cristian Donos",
    author_email='cristian.donos@g.unibuc.ro',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="Handful of tools for epilepsy research.",
    entry_points={
        'console_scripts': [
            'epi-preproc=pyepi.cli:preproc',
            'epi-trac=pyepi.cli:trac',
            'epi-pipe=pyepi.cli:pipeline'
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyepi',
    name='pyepi',
    packages=find_packages(include=['pyepi', 'pyepi.interfaces', 'pyepi.tools', 'pyepi.pipelines']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/cristidonos/pyepi',
    version='0.1.0',
    zip_safe=False,
)
