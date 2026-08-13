# `.vscode/`

This directory is for local VS Code / Cursor workspace settings. The repo root `.gitignore`
lists `.vscode/`, so nothing here is committed by default — only this README is force-added as
orientation for contributors.

A fresh checkout has no shared editor config: no `settings.json`, `extensions.json`, or
`launch.json` ship with the tree. If you use rust-analyzer, you may want a local
`settings.json` (for example `rust-analyzer.check.command: "clippy"` or a custom
`rust-analyzer.cargo.extraArgs` when working with `--features gpu`), but those choices stay on
your machine.
