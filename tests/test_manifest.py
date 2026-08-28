"""The declaration in components.json is only worth reading if this repository
proves it, and proves it against the packaging metadata rather than by
describing.

estate-gates cannot do this. It has no Python toolchain, and building
twenty-two repositories in its CI is a matrix it does not have. This repository
already runs pytest on every push.

What is proved here is exactly the `checked` bucket and nothing else. The
`declared` bucket is not asserted against anything, on purpose: a test that
pretended to verify a sentence about purpose would be the failure this whole
design exists to avoid.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"ENGRAM_[A-Z0-9_]+")


def manifest() -> dict:
    return json.loads((ROOT / "components.json").read_text())


def components() -> list[dict]:
    cs = manifest()["components"]
    assert cs, "components.json declares nothing, so every test here measured nothing"
    return cs


def pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_every_console_script_this_package_installs_is_declared_and_the_reverse():
    """THE ONE THAT CLOSES THE HOLE.

    `[project.scripts]` is already a cross-repository contract: stack-k8s's
    console.Dockerfile installs the Python tools by reading it and says so in a
    comment. A script added here without a manifest entry is a component the
    console image would install and nothing would have declared.

    estate.json names one of these three, and that is correct rather than an
    omission: it records what a DEPLOYMENT installs and this records what the
    PACKAGE installs. The difference between the two lists is the information
    neither file holds alone.
    """
    scripts = pyproject()["project"].get("scripts") or {}
    assert scripts, "pyproject declares no console script, so this measured nothing"

    declared = {
        c["checked"]["console_script"] for c in components() if "console_script" in c["checked"]
    }
    assert declared, "no component declares a console script, so this measured nothing"

    for name in scripts:
        assert name in declared, (
            f"pyproject installs the script {name!r} and components.json does not "
            f"declare it. A component nobody declares is one no deployment can be "
            f"asked to install."
        )
    for name in sorted(declared):
        assert name in scripts, (
            f"components.json declares the script {name!r} and pyproject installs no such thing"
        )

    for c in components():
        if "entry_point" not in c["checked"]:
            continue
        script = c["checked"]["console_script"]
        assert c["checked"]["entry_point"] == scripts[script], (
            f"components.json says {script!r} enters at {c['checked']['entry_point']!r}; "
            f"pyproject says {scripts[script]!r}"
        )


def test_the_declared_distribution_name_is_the_one_pip_would_install():
    """Three names for one component, and this is the one that costs an afternoon.

    The repository is `engram`, the import is `engram`, and the thing you
    `pip install` is `engdbram`. Checked rather than restated.
    """
    want = manifest().get("distribution")
    assert want, "components.json records no distribution name, so this measured nothing"
    got = pyproject()["project"]["name"]
    assert want == got, (
        f"components.json says the distribution is {want!r} and pyproject says {got!r}"
    )
    assert want != "engram", (
        "the distribution name equals the repository name, which is the case this "
        "entry exists to record. If the PyPI name was recovered, say so here rather "
        "than leaving a check that now asserts nothing."
    )


def test_every_environment_variable_this_repository_reads_is_declared_and_the_reverse():
    """Every ENGRAM_ name in non-test source against every one declared."""
    declared: set[str] = set()
    for c in components():
        declared |= set(c["checked"].get("env", {}))
    assert declared, "no component declares an environment variable, so this measured nothing"

    in_source: set[str] = set()
    for p in ROOT.rglob("*.py"):
        s = str(p.relative_to(ROOT))
        if s.startswith((".venv", "tests/")) or "/tests/" in s:
            continue
        in_source |= {n for n in NAME.findall(p.read_text(errors="ignore")) if not n.endswith("_")}
    assert in_source, "no ENGRAM_ name found in non-test source, so this measured nothing"

    missing = sorted(in_source - declared)
    extra = sorted(declared - in_source)
    assert not missing, f"the code reads these and components.json declares none of them: {missing}"
    assert not extra, f"components.json declares these and no non-test source reads them: {extra}"


def test_every_declared_entry_point_names_a_module_that_exists_on_disk():
    """The half that works with no dependencies installed at all.

    Importing `engram.no_such_module` does not report `engram.no_such_module`:
    Python imports `engram` first, that reaches numpy, and the dependency error
    MASKS the missing module. Measured, by pointing the manifest at a module that
    does not exist and watching the import check skip instead of fail.

    So the existence of the module is checked on the filesystem, where no
    dependency can hide it, and the import check below is the stronger half that
    runs where the package is installed.
    """
    checked = 0
    for c in components():
        ep = c["checked"].get("entry_point")
        if not ep:
            continue
        checked += 1
        module, _, _func = ep.partition(":")
        parts = module.split(".")
        as_file = ROOT.joinpath(*parts).with_suffix(".py")
        as_package = ROOT.joinpath(*parts, "__init__.py")
        assert as_file.exists() or as_package.exists(), (
            f"components.json says {c['name']} enters at {ep!r} and neither "
            f"{as_file.relative_to(ROOT)} nor {as_package.relative_to(ROOT)} exists"
        )
    assert checked, "no component declares an entry point, so this measured nothing"


def test_every_declared_entry_point_is_importable_and_callable():
    """AND THE HALF NO PACKAGING FILE COULD EVER DO: the entry point resolves.

    `pyproject` will happily name a module that does not exist and a function
    that is not there; nothing fails until somebody runs the installed script.
    So each declared `module:function` is imported and the attribute looked up.

    SKIPPED when this package is not importable, which is the state on a machine
    with no virtualenv for it, and it is skipped rather than passed so that a run
    which proved nothing says so.
    """
    checked = 0
    skipped: list[str] = []
    for c in components():
        ep = c["checked"].get("entry_point")
        if not ep:
            continue
        module, _, func = ep.partition(":")
        assert func, f"{ep!r} names no function"
        try:
            mod = importlib.import_module(module)
        except ImportError as e:
            # A MISSING DEPENDENCY IS NOT A BROKEN ENTRY POINT, and the difference
            # is exactly `e.name`. `No module named 'numpy'` means this machine
            # has no virtualenv for the package; `No module named 'engram.cli'`
            # means the manifest points at something that does not exist, and
            # that is a finding rather than an environment.
            missing = e.name or ""
            if missing == module or missing.startswith("engram"):
                raise AssertionError(
                    f"components.json says {c['name']} enters at {ep!r} and importing "
                    f"{module} failed with a missing {missing!r}, which is this "
                    f"package rather than a dependency"
                ) from e
            skipped.append(f"{ep} (needs {missing})")
            continue
        checked += 1
        assert callable(getattr(mod, func, None)), (
            f"components.json says {c['name']} enters at {ep!r} and {module} has no callable {func}"
        )
    if not checked:
        pytest.skip(
            "no entry point could be imported without a dependency this machine "
            f"does not have, so this measured nothing: {skipped}.\n"
            "The module's EXISTENCE is still checked, by the test above, which the "
            "filesystem answers without importing anything."
        )


def test_every_declared_console_script_exists_once_installed():
    """The other half: the script is on PATH after an install.

    SKIPPED when it is not, for the same reason and with the same honesty: this
    is the assertion that only an installed package can answer, and in CI it is
    installed.
    """
    scripts = [
        c["checked"]["console_script"] for c in components() if "console_script" in c["checked"]
    ]
    assert scripts, "no component declares a console script, so this measured nothing"

    found = [s for s in scripts if shutil.which(s)]
    if not found:
        pytest.skip("no declared console script is on PATH, so this package is not installed here")

    for script in found:
        got = subprocess.run([script, "--help"], capture_output=True, text=True)
        assert got.returncode == 0, (
            f"`{script} --help` exited {got.returncode}, so a script this manifest "
            f"declares does not run:\n{got.stderr}"
        )
