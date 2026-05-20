# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | ✅ Active  |
| < 1.0   | ❌ Pre-release; no security backports. Please upgrade to 1.0+. |

## Reporting a Vulnerability

If you believe you've found a security vulnerability in UnitPort, **please do
not open a public GitHub issue**. Public disclosure before a fix is available
puts other users at risk.

Instead, report it privately:

1. **Preferred** — use GitHub's "Report a vulnerability" form on the
   [Security tab](https://github.com/DrLavier/UnitPort/security/advisories/new)
   of the repository. This creates a private advisory only maintainers can
   see.
2. **Email** — `security@uniport.ai` (replace with whatever address the
   maintainers monitor; PGP key available on request).

Please include:

- A description of the issue and its impact.
- Steps to reproduce, or a proof-of-concept if you have one.
- The UnitPort version (`system.ini[System].version`, or the git commit).
- Your OS and Python version.
- Whether the issue is reachable in a default install or requires an
  unusual configuration (Isaac Lab, cloud sync, a specific vendor SDK, …).

## What to expect

- **Acknowledgement** within 5 business days.
- **Triage** — we'll confirm reproduction and let you know whether we
  consider it a vulnerability, a hardening opportunity, or out of scope.
- **Fix timeline** — depends on severity:
  - Critical / remote-code-execution: target 7 days.
  - High (privilege escalation, auth bypass): target 30 days.
  - Lower-severity issues are batched into the next minor release.
- **Disclosure** — coordinated. We'll publish a GitHub Security Advisory and
  CVE (if applicable) at the same time as the fix, crediting the reporter
  unless they prefer anonymity.

## Scope

In scope:

- The UnitPort application (`main.py` and the `src/` tree).
- The `unitport_sdk` package.
- The install / start / reset scripts.
- The cloud sync layer (Supabase auth + storage integration).
- The Mission Control adapters (Unitree, Boston Dynamics, MangDang) **as
  shipped by this repository** — issues in the vendors' upstream SDKs
  should be reported to those vendors.

Out of scope:

- Vulnerabilities in third-party dependencies (PyTorch, PyQt6, MuJoCo,
  Isaac Lab, ROS 2, vendor SDKs). Please report those to their respective
  projects. We will of course bump dependency versions once upstream
  publishes fixes.
- Issues that require the attacker to already have control of the host
  machine or to coerce the user into running an attacker-supplied bundle
  (bundles are not a security boundary; load only bundles you trust).
- Self-XSS in markdown rendering of strings the user themselves provided.

## Hardening notes for operators

- **Supabase `anon_key` is publishable.** The string in
  `src/config/system.ini[auth].supabase_anon_key` is a Supabase
  `sb_publishable_…` key designed for client distribution; the security
  boundary is row-level security policies on the Supabase server, not the
  key. Do not treat its presence in the repository as a credential leak.
- **SSH passwords are never persisted** in the engines or projects JSON.
  They live in the OS keyring via `SecureCredentialStore.ssh_password`.
- **Bundles execute Python.** A bundle is a self-contained artifact that
  may include custom node code. Only load bundles you produced yourself or
  that come from a trusted source. We recommend code-review of any
  third-party bundle before loading.

Thank you for helping keep UnitPort and its users safe.
