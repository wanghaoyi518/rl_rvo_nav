Documentation Maintenance Guide
===============================

Purpose
-------

This guide explains how to maintain and extend the documentation system in the `rl_rvo_nav` repository.
It is intended for both humans and AI assistants working on this project.

Principles
----------

- Separate **contracts** (what a module promises externally) from **implementation details**.
- Keep a **single source of truth** for each piece of information.
- Organize docs in layers: project-level `docs/` and per-module `docs/`.
- Avoid duplicate or versioned copies of the same logical document.

When adding a new feature or capability
---------------------------------------

1. Identify which module is responsible for implementing the feature.
2. Update or create the appropriate contract document in that module's `docs/` directory:
   - Use the `*.contract.md` naming convention.
   - Describe goals, interfaces, inputs/outputs, error handling, and performance constraints.
3. If the feature affects cross-module flows, update:
   - Project-level `docs/architecture.md` to reflect new dependencies or flows.
   - The caller module's integration note under its `docs/integrations/` directory (once those exist).
4. Update `docs/doc-index.md` if new major docs are added.

When changing an existing contract
----------------------------------

1. Edit the existing `*.contract.md` file instead of creating a new file with a different name.
2. Review all known consumers listed in the contract:
   - Update their integration notes to reflect the new behavior.
   - Check whether any breaking changes need explicit migration steps.
3. If architecture or main flows are affected, update `docs/architecture.md`.

When adding or modifying scripts and run flows
----------------------------------------------

1. Prefer stable script or CLI entry points (e.g., shell scripts, Python entry modules).
2. Reference only these stable entry points from `docs/repro/runbook.md`.
3. When new run flows are added:
   - First, define or update the scripts.
   - Then, document how to use them in `docs/repro/runbook.md`.

When changing CBS (deadlock_resolution) code or config
------------------------------------------------------

1. Update `deadlock_resolution/docs/cbs_implementation_notes.md`: add or adjust the relevant subsection (viz, solver fixes, config, troubleshooting).
2. Keep the "Fixes applied" and "Root causes and fixes" tables in sync with the code so future debugging has a single place to look.

File and naming conventions
---------------------------

- Project-level:
  - `docs/README.md`: Overview and conventions.
  - `docs/doc-index.md`: Global index of all docs.
  - `docs/architecture.md`: Architecture and module boundaries.
  - `docs/repro/runbook.md`: Environment and run instructions.
- Module-level:
  - `module_name/docs/README.md`: Module overview and links to other docs.
  - `module_name/docs/*.contract.md`: Contracts for external capabilities.
  - `module_name/docs/*.design.md`: Internal design/implementation details (optional).
  - `module_name/docs/integrations/*.md`: Integration notes for this module as a caller.

Do not
------

- Do not create parallel documents with names like `api-v2.md`, `api-final.md`, etc.
- Do not duplicate large sections of text between documents; link instead.
- Do not rely on undocumented behavior—if it matters to callers, it must appear in a contract or integration note.

