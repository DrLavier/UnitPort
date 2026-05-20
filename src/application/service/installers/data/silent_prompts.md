# Silent-prompts disclosure

This document tells the user, **before** they accept, every interactive
prompt that the UnitPort in-app installer will silently pass-through to
`isaaclab.bat -i` / `isaaclab.sh -i` on their behalf during the Isaac
Sim 5.x install.

The list below is the practical equivalent of typing **`y` <Enter>**
to every prompt; the installer also exports a matching set of
environment variables so newer Isaac Sim builds that switched from
interactive prompts to env-var gating still auto-accept.

---

## Env-var auto-accept

The installer exports these env vars when running the Isaac Sim and
Isaac Lab install scripts. Values are taken from
`installers/_constants.INSTALLER_ACCEPT_ENV`; any change there should
also update this list.

| Variable | Value | Replaces interactive prompt |
|---|---|---|
| `ACCEPT_EULA` | `Y` | Isaac Sim binary EULA banner |
| `OMNI_KIT_ACCEPT_EULA` | `YES` | Omniverse Kit runtime EULA |
| `NVIDIA_OMNIVERSE_LICENSE` | `accepted` | NVOLA banner displayed once on first launch |
| `PRIVACY_CONSENT` | `Y` | Isaac Sim 5.x telemetry opt-out prompt (we always opt **in**; users who want full opt-out should locate an existing Isaac Sim install instead and run the official uninstall-telemetry helper) |

## stdin auto-yes fallback

Older `isaaclab` scripts read confirmations from stdin rather than env
vars. As a defence-in-depth measure the installer pipes
`y\ny\ny\ny\n` into the subprocess so up to four sequential prompts
are auto-answered without user interaction.

If you are uncomfortable with any of the above, **cancel the wizard
now and run the Isaac Sim installer manually**, then use the
"Locate existing Isaac Lab installation" wizard option to register
your install with UnitPort.

---

## What this acknowledgement covers

By accepting this disclosure you confirm that:

1. You have read the NVOLA tab (NVIDIA Omniverse License Agreement).
2. You have read the Isaac Lab BSD-3-Clause licence tab.
3. You consent to UnitPort answering the interactive prompts above on
   your behalf so that the multi-hour install can run unattended.

Acceptance is recorded under
`<USER_CONFIG_DIR>/eula_acceptance.json` and survives a clean reinstall of
UnitPort. If the env-var list above changes in a future UnitPort
release, you will be re-prompted with the updated disclosure.
