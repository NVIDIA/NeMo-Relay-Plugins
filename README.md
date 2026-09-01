# NVIDIA PLC OSS Standard Repository Template

This repository provides the standard templates and starting guidance for NVIDIA projects that publish open-source or source-available software. It is a guided superset: keep and customize the sections and files that match your project, and trim anything that does not apply.

The template set is intentionally more complete than any one project necessarily needs. Before treating the template set as finished, define the project's lifecycle, software maturity, license model, support posture, and contribution policy, then make sure every retained file describes that profile consistently.

This README provides a quick overview of the template and a project README skeleton below the customization separator. For agent instructions on applying the templates, see [TEMPLATE_INSTRUCTIONS.md](TEMPLATE_INSTRUCTIONS.md).

**After customization, remove the template guidance above the separator and retain the completed project README below it.**

## Included templates

This is a map of the reusable files included in this repository, not a list of files every derived repository must retain. Use the agent instructions to determine applicability.

### Root

- [README.md](README.md): Project context, status, getting started, usage, support, security, releases, roadmap, and license skeleton
- [TEMPLATE_INSTRUCTIONS.md](TEMPLATE_INSTRUCTIONS.md): Agent-facing workflow for applying the template; remove it from the derived public repository
- [LICENSE](LICENSE): License text that every repository must confirm or replace with its exact license
- [CONTRIBUTING.md](CONTRIBUTING.md): Open, limited, or closed contribution-policy alternatives
- [SECURITY.md](SECURITY.md): Private vulnerability-reporting guidance
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md): Community standards for projects that accept public participation
- [AGENTS.md](AGENTS.md): Concise, repository-specific context, important paths, and verified commands for coding agents used by contributors or users
- [RELEASE.md](RELEASE.md): Maintainer instructions for preparing, publishing, verifying, and recovering releases
- [SUPPORT.md](SUPPORT.md): Detailed support guidance retained only when the README is not sufficient
- [GOVERNANCE.md](GOVERNANCE.md): Decision and role model retained when the project publishes one
- [MAINTAINERS.md](MAINTAINERS.md): Current maintainer information, with public contact or role-change guidance when the project publishes it
- [CITATION.md](CITATION.md): Preferred citation metadata retained when the software is citable
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): Attribution and notice records retained when license or dependency obligations require them

### `.github/`

- [ISSUE_TEMPLATE/01_bug_report.yml](.github/ISSUE_TEMPLATE/01_bug_report.yml): Form for accepted public bug reports
- [ISSUE_TEMPLATE/02_feature_request.yml](.github/ISSUE_TEMPLATE/02_feature_request.yml): Form for accepted public feature requests
- [ISSUE_TEMPLATE/03_documentation_request.yml](.github/ISSUE_TEMPLATE/03_documentation_request.yml): Form for accepted public documentation requests
- [ISSUE_TEMPLATE/config.yml](.github/ISSUE_TEMPLATE/config.yml): Issue chooser configuration, blank-issue policy, and question and security contact links ([GitHub documentation](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository))
- [PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md): Pull request prompts for projects that accept pull requests ([GitHub documentation](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository))
- [CODEOWNERS](.github/CODEOWNERS): Optional automatic review requests for specific users or teams based on the files changed

## Common project-specific additions

The template does not prescribe a complete software layout. Add the documentation, automation, packaging, tests, examples, and other files the project needs. Common additions include:

- `docs/`
- `examples/`
- `tests/`
- `scripts/`
- Configured `.github/workflows/`
- Container or development environment: `Dockerfile`, `docker/`, or `.devcontainer/`
- Build and package files:
  - Python: `pyproject.toml`, `setup.cfg`, `setup.py`, `requirements.txt`, or `environment.yml`
  - C++: `CMakeLists.txt`, `cmake/`, or `vcpkg.json`
- Repository hygiene: `.gitignore`, `.gitattributes`, `.editorconfig`, `.pre-commit-config.yaml`, or `.clang-format`
- Project-specific license or notice files when the included starters do not apply

## Usage for new NVIDIA repositories

1. Clone [PLC-OSS-Template](https://github.com/NVIDIA-GitHub-Management/PLC-OSS-Template).
2. Point your coding agent at the agent-facing [TEMPLATE_INSTRUCTIONS.md](TEMPLATE_INSTRUCTIONS.md).
3. Replace `__PROJECT__` throughout the retained templates, `__ORG__` in `.github/CODEOWNERS` and retained GitHub URLs, and every remaining `__PLACEHOLDER_LIKE_THIS__` with accurate project-specific content.
4. Keep, customize, or remove conditional sections and files according to your project's needs. Conditional content blocks use `<!-- TEMPLATE:BEGIN ... -->` and `<!-- TEMPLATE:END ... -->`. **Use when** identifies conditional content, **Choose** identifies alternatives, and **Write** identifies author guidance that must be replaced with project content. Remove unselected content blocks, then remove all visible author guidance and conditional markers from the retained content.
5. Add any project-specific content or files the software needs; this template is a framework meant to be built on.
6. Configure repository settings and channels to match the written policies.

**Remove before publishing**

Remove `TEMPLATE_INSTRUCTIONS.md`, the template guidance above the README separator, and any files or sections that do not apply. From retained files, remove author guidance, conditional markers, setup comments, and unresolved placeholders.

**Files requiring little or no customization**

- `SECURITY.md`, after confirming the standard NVIDIA reporting path remains applicable
- `CODE_OF_CONDUCT.md`, when the project accepts public participation, after replacing its project placeholder
- Applicable issue and pull request templates, after resolving placeholders, links, and repository settings

**Files you must customize or remove**

- `README.md`
- `CONTRIBUTING.md`
- `LICENSE` and legal contribution files
- `.github/CODEOWNERS`: Customize and retain when pull requests should automatically request review from specific users or teams based on the files changed; otherwise remove.
- `AGENTS.md`: Customize with concise repository context, important paths, and verified commands when contributors or users may use coding agents; otherwise remove.
- `RELEASE.md`: Customize with the verified maintainer release process, or remove.
- `SUPPORT.md` and `CITATION.md`: Customize or remove.
- `MAINTAINERS.md` and `GOVERNANCE.md`: Include only when relevant to the project.
- Notice files: Include only when required.

**Release and roadmap surfaces**

- Release history: When the project publishes releases, use GitHub Releases as the canonical public release history.
- Release process: Use `RELEASE.md` when maintainers need a documented release procedure, and keep it aligned with release automation.
- Public roadmap: Use a GitHub Project for an actively tracked roadmap or a pinned issue for a concise roadmap, and link it from the project README.

## Usage for existing NVIDIA repositories

Follow the same profile and consistency process, but preserve accurate project-specific material and merge only the applicable sections and files into the existing repository. Build on the template with anything else the project needs.

<!-- REMOVE THE LINE BELOW AND EVERYTHING ABOVE AFTER CUSTOMIZATION -->
-------------------------------------------------------------------------------

# __PROJECT__

<!-- TEMPLATE:BEGIN id="readme.badges" condition="project-has-verifiable-public-status-information" -->
> **Use when:** The project has maintained CI, release, coverage, or other public status information that a badge would help readers verify.
>
> **Write:** Add relevant badges such as CI status, license, latest release, or coverage.

<!-- badges go here -->
<!-- TEMPLATE:END id="readme.badges" -->

> **Write:** In one sentence, state what the software is, what it does, and who it is for.

<!-- TEMPLATE:BEGIN id="readme.snapshot-notice" condition="lifecycle=snapshot" -->
> **Use when:** The repository is a snapshot.

> [!IMPORTANT]
> This repository is a fixed snapshot provided as-is. It is not actively maintained. Issues and pull requests might not receive a response.

<!-- TEMPLATE:END id="readme.snapshot-notice" -->

<!-- TEMPLATE:BEGIN id="readme.maintenance-notice" condition="lifecycle=maintenance-only" -->
> **Use when:** The project is maintenance-only.

> [!NOTE]
> This project is in maintenance-only mode. Only __SUPPORTED_MAINTENANCE_SCOPE__ is planned; new feature development is not planned.

<!-- TEMPLATE:END id="readme.maintenance-notice" -->

<!-- TEMPLATE:BEGIN id="readme.archive-notice" condition="lifecycle=archived" -->
> **Use when:** The repository is archived.

> [!WARNING]
> This repository is archived and no longer maintained. __LINK_TO_SUCCESSOR_OR_FINAL_STATUS__.

<!-- TEMPLATE:END id="readme.archive-notice" -->

<!-- TEMPLATE:BEGIN id="readme.source-available-notice" condition="license-model=source-available" -->
> **Use when:** The software is source available rather than open source.

> [!IMPORTANT]
> This software is source available, not open source. Review the [source-available usage terms](#source-available-usage-terms) before using, modifying, or distributing it.

<!-- TEMPLATE:END id="readme.source-available-notice" -->

## Overview

> **Write:** Explain what the software does, why it is useful, and its main capabilities. Where relevant, cover the intended audience, common use cases, and important scope details. Include a concise statement describing the project's status and summarizing its support and contribution policies.

### Features

> **Write:** List key capabilities in bullets or a concise table.

## Getting started

> **Write:** Provide the shortest path to install or obtain the software and run something meaningful. If no runnable quick start applies, direct readers to the most useful next step.

```bash
# Install or obtain the software
__QUICK_START_INSTALL_COMMAND__

# Run a minimal example
__QUICK_START_COMMAND__
```

Expected result:

```text
__EXPECTED_OUTPUT__
```

## Requirements

> **Write:** List only prerequisites and remove categories that do not apply.

- OS and architecture: __SUPPORTED_OS_AND_ARCHITECTURE__
- Runtime or compiler: __RUNTIME_OR_COMPILER_VERSIONS__
- NVIDIA dependencies: __NVIDIA_DEPENDENCIES_OR_NOT_APPLICABLE__
- GPU, driver, and CUDA requirements: __GPU_DRIVER_CUDA_REQUIREMENTS_OR_NOT_APPLICABLE__
- Known-good environment: __KNOWN_GOOD_ENVIRONMENT__

## Installation

> **Write:** Explain the recommended installation method and any supported alternatives. If the software is not distributed or installed separately, explain that.

```bash
__INSTALL_COMMANDS__
```

## Usage

> **Write:** Show a realistic, copy-pastable example of the software's primary use.

```text
__MINIMAL_USAGE_EXAMPLE__
```

<!-- TEMPLATE:BEGIN id="readme.telemetry" condition="software-collects-telemetry-or-usage-data" -->
## Telemetry and data collection

> **Use when:** The software collects telemetry or usage data.

- Data collected: __DATA_COLLECTED__
- Data not collected: __DATA_NOT_COLLECTED__
- Purpose: __COLLECTION_PURPOSE__
- Disable or opt out: __OPT_OUT_INSTRUCTIONS__
- More information: __TELEMETRY_DOCUMENTATION__
<!-- TEMPLATE:END id="readme.telemetry" -->

## Documentation

> **Write:** Keep only links to documentation surfaces the project actually maintains.

- Documentation and API reference: __DOCUMENTATION_HOME__
- Examples and tutorials: __EXAMPLES_LINK__

<!-- TEMPLATE:BEGIN id="readme.architecture" condition="architecture-context-is-useful" -->
## Architecture

> **Use when:** The software benefits from an architecture overview.
>
> **Write:** Describe the main components and how they relate. Include or link to a diagram when it makes the architecture materially easier to understand.

<!-- TEMPLATE:END id="readme.architecture" -->

<!-- TEMPLATE:BEGIN id="readme.performance" condition="project-publishes-performance-claims" -->
## Performance

> **Use when:** The project publishes performance claims or benchmarks.
>
> **Write:** Summarize benchmarks and link to detailed results. Include the hardware, software, and methodology used.

<!-- TEMPLATE:END id="readme.performance" -->

## Support and contributions

- Bug reports: __BUG_REPORT_PATH_OR_NOT_ACCEPTED__
- Questions and support: __SUPPORT_PATH_OR_NOT_PROVIDED__
- Feature requests: __FEATURE_REQUEST_PATH_OR_NOT_ACCEPTED__
- Response expectations: __SUPPORT_RESPONSE_EXPECTATION__
- Contribution scope: __CONTRIBUTION_SCOPE_OR_NOT_ACCEPTED__

See [CONTRIBUTING.md](CONTRIBUTING.md) for the project's contribution policy and participation guidance.

<!-- TEMPLATE:BEGIN id="readme.code-of-conduct" condition="project-accepts-public-participation" -->
> **Use when:** The project accepts public participation through contributions, bug reports, questions, or community channels.

All project participants must follow the [Code of Conduct](CODE_OF_CONDUCT.md).
<!-- TEMPLATE:END id="readme.code-of-conduct" -->

<!-- TEMPLATE:BEGIN id="readme.support-file" condition="project-maintains-support-file" -->
> **Use when:** The project maintains `SUPPORT.md` with additional support guidance.

See [SUPPORT.md](SUPPORT.md) for additional support guidance.
<!-- TEMPLATE:END id="readme.support-file" -->

<!-- TEMPLATE:BEGIN id="readme.reproducibility" condition="repository-accompanies-research-or-contains-reproducible-results" -->
## Reproducing published results

> **Use when:** The repository accompanies published research or contains results intended to be reproduced.

- Reference commit or release: __REFERENCE_COMMIT_OR_RELEASE__
- Environment: __REPRODUCIBILITY_ENVIRONMENT__
- Hardware: __KNOWN_GOOD_HARDWARE__
- Command: `__REPRODUCTION_COMMAND__`
- Expected result: __EXPECTED_REPRODUCTION_RESULT__
<!-- TEMPLATE:END id="readme.reproducibility" -->

<!-- TEMPLATE:BEGIN id="readme.limitations" condition="known-limitations-or-research-profile" -->
## Known limitations

> **Use when:** The software has known limitations worth highlighting or the repository accompanies published research.
>
> **Write:** Describe known limitations, failure modes, unsupported environments, and any expected variability or drift.

<!-- TEMPLATE:END id="readme.limitations" -->

<!-- TEMPLATE:BEGIN id="readme.releases" condition="project-publishes-github-releases" -->
## Releases

> **Use when:** The project publishes releases through GitHub Releases.

See [GitHub Releases](__GITHUB_RELEASES_URL__) for release notes.

<!-- TEMPLATE:BEGIN id="readme.release-process" condition="release-process-file-retained" -->
See [RELEASE.md](RELEASE.md) for the maintainer release process.
<!-- TEMPLATE:END id="readme.release-process" -->
<!-- TEMPLATE:END id="readme.releases" -->

<!-- TEMPLATE:BEGIN id="readme.roadmap" condition="project-publishes-public-roadmap" -->
## Roadmap

> **Use when:** The project publishes a public roadmap.
>
> **Write:** Link to the project's canonical GitHub Project or pinned roadmap issue. Do not duplicate changing status or dates here.

<!-- TEMPLATE:END id="readme.roadmap" -->

<!-- TEMPLATE:BEGIN id="readme.governance" condition="project-publishes-governance-or-maintainer-information" -->
## Governance and maintainers

> **Use when:** The project publishes governance or maintainer information. Keep only the links to files the project publishes.

- Governance: [GOVERNANCE.md](GOVERNANCE.md)
- Maintainers: [MAINTAINERS.md](MAINTAINERS.md)
<!-- TEMPLATE:END id="readme.governance" -->

## Security

Do not report security vulnerabilities through public GitHub issues. See [SECURITY.md](SECURITY.md) for the reporting path.

<!-- TEMPLATE:BEGIN id="readme.community" condition="project-provides-community-or-maintainer-contact-route" -->
## Community

> **Use when:** The project provides a public community channel or another way to communicate with maintainers.
>
> **Write:** List public community channels. If none exist, state where questions should go or that no public question channel is provided.

<!-- TEMPLATE:BEGIN id="readme.community-meetings" condition="project-holds-public-community-meetings" -->
### Community meetings

> **Use when:** The project holds public meetings that contributors or users can join.
>
> **Write:** State the cadence, time, and time zone, and link to joining details and past recordings.

<!-- TEMPLATE:END id="readme.community-meetings" -->
<!-- TEMPLATE:END id="readme.community" -->

<!-- TEMPLATE:BEGIN id="readme.references" condition="project-has-relevant-references" -->
## References

> **Use when:** The project has references that materially help readers understand the software and its context.
>
> **Write:** List the papers, specifications, upstream projects, or other sources that help readers understand the software and its context.

<!-- TEMPLATE:END id="readme.references" -->

<!-- TEMPLATE:BEGIN id="readme.citation" condition="software-is-citable" -->
## Citation

> **Use when:** The software has a preferred citation.
>
> **Write:** Provide the preferred citation and, when applicable, BibTeX. Keep it consistent with [CITATION.md](CITATION.md).

<!-- TEMPLATE:END id="readme.citation" -->

## License

The software in this repository is licensed under __LICENSE_NAME__. See [LICENSE](LICENSE) for details.

<!-- TEMPLATE:BEGIN id="readme.source-available-terms" condition="license-model=source-available" -->
### Source-available usage terms

> **Use when:** The software is source available rather than open source.

This is a source-available license, not an open-source license. The summary below does not replace the license terms.

- Permitted uses: __PERMITTED_USES__
- Restricted or prohibited uses: __RESTRICTED_USES__
<!-- TEMPLATE:END id="readme.source-available-terms" -->

<!-- TEMPLATE:BEGIN id="readme.notices" condition="software-distribution-includes-required-notices" -->
> **Use when:** The software distribution requires copyright or third-party notices.

See __NOTICE_FILE_LINK__ for copyright and third-party attribution notices.
<!-- TEMPLATE:END id="readme.notices" -->
