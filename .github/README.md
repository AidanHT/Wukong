# `.github/`

GitHub-specific metadata for the Wukong compiler repository. Today this folder holds only
[`workflows/`](workflows/) — there are no issue templates, Dependabot configs, or CODEOWNERS
files checked in.

CI is defined entirely in [`workflows/ci.yml`](workflows/ci.yml). It runs on every pull request
and on pushes to `main`. See [`workflows/README.md`](workflows/README.md) for what each job does
and how the default `test` job differs from `gpu-check`.
