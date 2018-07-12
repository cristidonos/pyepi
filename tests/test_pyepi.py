#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyepi` package."""

import pytest
from click.testing import CliRunner

from pyepi import cli


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.preproc, ['dummy'])
    assert result.exit_code == 0
    assert '' in result.output
    help_result = runner.invoke(cli.preproc, ['--help'])
    assert help_result.exit_code == 0
    assert '  Script for running various pre-processing pipelines.' in help_result.output
