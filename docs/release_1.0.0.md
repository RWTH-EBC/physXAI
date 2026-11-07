# Release 1.0.0: checklist after merging into `main`

Steps for the maintainer, in this order. The CI changed with 1.0.0 to the shared template of the institute
(`RWTH-EBC/ci_templates`): documentation and badges are published on the branch `gh-pages` instead of being deployed
by a workflow, which needs a change of the repository settings.

## 1. Merge

1. Merge all open PRs into `refactoring_v1.0.0`.
2. Merge the PR `refactoring_v1.0.0` → `main`.
   AgentLib-MPC installs `physXAI @ git+https://github.com/RWTH-EBC/physXAI.git` (`main`) in every CI run, so its
   tests run against 1.0.0 from now on (`tests/test_physXAI_plugin` passes with 1.0.0).

## 2. Wait for the CI on `main`

In *Actions*, the workflow **Python CI** of the merge commit on `main` has to finish, in particular the job
`ci / gh_pages / gh_pages`. It writes the documentation and the reports to the branch `gh-pages`, folder
`docs/main/`.

Check (on GitHub: branch `gh-pages`, or `git fetch origin gh-pages && git ls-tree --name-only origin/gh-pages docs/main/`):
`docs/main/` contains `docs/`, `coverage/`, `pylint/`, `ruff/` and `build/`.

Do **not** switch GitHub Pages before: the start page `https://rwth-ebc.github.io/physXAI/` redirects to
`.../main/docs/index.html`, which only exists after this step. Until the switch, the old MkDocs site stays online.

## 3. Switch GitHub Pages

*Settings → Pages → Build and deployment*:

1. **Source:** `Deploy from a branch` (instead of `GitHub Actions`).
2. **Branch:** `gh-pages`, folder **`/docs`**, then **Save**.
3. *Settings → Actions → General → Workflow permissions:* `Read and write permissions` (the template pushes to
   `gh-pages` with `GITHUB_TOKEN`; `ci.yml` also requests `contents: write`). This was already sufficient for the
   phase branches, so it is likely set.

GitHub deploys the branch within a few minutes (*Actions → pages build and deployment*).

## 4. Check the pages

| URL | Expected |
|---|---|
| https://rwth-ebc.github.io/physXAI/ | redirect to the documentation of `main` |
| https://rwth-ebc.github.io/physXAI/main/docs/index.html | documentation 1.0.0 (Sphinx) |
| https://rwth-ebc.github.io/physXAI/main/docs/migration.html | migration guide |
| https://rwth-ebc.github.io/physXAI/main/coverage/badge.svg | coverage badge (README) |
| https://rwth-ebc.github.io/physXAI/main/pylint/pylint.svg | pylint badge (README) |
| https://rwth-ebc.github.io/physXAI/main/docs/doc.svg | documentation badge (README) |

Then check the badges and links of the README on the start page of the repository.

Every other branch gets its own preview after a push: `https://rwth-ebc.github.io/physXAI/<branch>/docs/index.html`.

## 5. Tag and release

1. Create the tag `v1.0.0` on the merge commit on `main` (e.g. *Releases → Draft a new release → Choose a tag:
   v1.0.0 → Target: main*, release notes from `CHANGELOG.md`). Keep the **known limitation** at the top of the release
   notes: the export of `RNNModel` / `PCNNModel` needs AgentLib-MPC branch `143-feature-multistep-model-support`.
2. The workflow **Version Check** runs on the tag: it compares the tag with `version = "1.0.0"` in `pyproject.toml`
   and requires the entry `## [1.0.0]` in `CHANGELOG.md`. If the release date differs from the date in the
   CHANGELOG, update it in a commit before tagging.
3. PyPI is not used yet (`PYPI_RELEASE: false` in `ci.yml`).
4. Later releases: update `constraints.txt` to the versions the release was tested with (the job `tested-versions`
   of `windows.yml` installs them on Linux and Windows).

## 6. Clean up

1. **Old branches:** all branches listed in [`branch_review.md`](branch_review.md) can be deleted, as well as
   `refactoring_v1.0.0` and the phase branches `v1/...` after the merge.
2. **`gh-pages`:** the template deletes folders of branches without commits for 30 days, but only top-level folders.
   Folders of branch names with `/` (`docs/v1/...`) stay; delete them on the branch `gh-pages` if wanted.
3. **`.github/workflows`:** remove the branch `refactoring_v1.0.0` from the `pull_request` triggers of
   `verify_examples.yml` (marked with a comment).
4. **Optional, required status checks for `main`** (*Settings → Branches*; currently none are required): useful are
   `ci / tests / test (3.11)` … `(3.13)`, `tests (3.11)` … `(3.13)` (Windows, with the required AgentLib-MPC contract
   tests), `tested-versions (ubuntu-latest, 3.11)` … `(windows-latest, 3.13)`, `lint`, `docs`,
   `verify-installation (ubuntu-latest)` and `verify-installation (windows-latest)`. Not `examples (...)` (only runs
   when the package, the examples or the data change) and not the report jobs of the template (coverage, pylint, ruff), which never fail.

## 7. AgentLib-MPC

- Branch `143-feature-multistep-model-support` (recurrent models) misses a fix of AgentLib-MPC `main`
  (`int(model.rank_)` in `machine_learning_plugins/physXAI/model_config_creation.py`); merge `main` into it. The
  export of RNN and PCNN models of physXAI 1.0.0 is verified against branch 143.
