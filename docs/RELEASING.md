# Releasing

Maintainers follow these steps to publish a new version.

## One-time setup

1. On PyPI, add a pending trusted publisher with these values:
   - Project name: `quantpreflight`
   - Owner: `layer8labs8`
   - Repository: `quantpreflight`
   - Workflow: `release.yml`
   - Environment: `pypi`
2. On GitHub, create an environment named `pypi` and add yourself as a required reviewer.
3. In the repository's security settings, turn on private vulnerability reporting.

## First release

Until the first release, the docs install from Git and pin the Action to `main`. After it:

1. Change `INSTALL` and `ACTION_REF` in `tools/build_site.py`.
2. Update the install commands, the `uses:` line, and `rev:` in `README.md`, and add the PyPI badge.

## Each release

1. Run `python tools/update_advisories.py` to refresh the bundled engine advisories.
2. Update `__version__` in `src/quantpreflight/__init__.py` and `version` in `pyproject.toml`.
3. Replace `Unreleased` in `CHANGELOG.md` with the version and date.
4. Merge to `main` and confirm that CI passes.
5. Create and push the tag:

   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```

6. Approve the `pypi` deployment on the workflow run. The workflow then publishes to PyPI and creates the GitHub release.

A release can't be replaced on PyPI, so a mistake needs a new version number.
