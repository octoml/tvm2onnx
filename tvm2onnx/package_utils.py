"""Common utilities and constants for packaging code."""
import pathlib
import typing

import structlog
from cookiecutter import generate

LOG = structlog.get_logger(__name__)

PACKAGE_TEMPLATE_NAME = "{{ cookiecutter.package_name }}"
"""The name of the package templates pre-cookiecutting."""

MODULE_TEMPLATE_NAME = "{{ cookiecutter.module_name }}"
"""The name of Python inner module folders pre-cookiecutting."""

PYTHON_BASE_RELATIVE_PATH_IN_PACKAGE = (
    pathlib.Path(PACKAGE_TEMPLATE_NAME) / MODULE_TEMPLATE_NAME
)
"""The relative path into which the Python base class template should be copied."""


def cookiecut_package(
    template_dir: pathlib.Path,
    output_dir: pathlib.Path,
    config: typing.Dict[str, typing.Any],
):
    """Cookiecuts the given template dir into the given output dir.

    :param template_dir: the directory to use as a template for the package.
    :param output_dir: the directory where the cookiecut package will go.
    :param config: the cookiecutter config to use.
    """
    LOG.debug(
        "Cookiecutting package from template.",
        template_dir=template_dir,
        output_dir=output_dir,
        config=config,
    )
    generate.generate_files(template_dir, {"cookiecutter": config}, output_dir)
