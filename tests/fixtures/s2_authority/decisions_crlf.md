# Decision log (synthetic fixture for gate.s2_authority; not a real decision record)

Format: `D-nnn · date · decision · why · what it changes`.

- **D-001 · 2026-09-22 · Field pick.** Synthetic opening entry.
  Continuation prose without any grant line.

## Vast host qualification authorized

- **D-019 · 2026-11-01 · Bounded Vast.ai rental, engineering qualification and cleanup.** Closed allocation.
  - authorizes: s2_pilot_packet sha256:d6c0423d3c60818ba9bc6c8fe410cd2efa92dbf7f6bc51a666e19944ce8202fc
  - records: terms:harmonizer

- **D-020 · 2026-11-01 · Actual Vast host check completed; allocation closed.** Outcome only.

## Proposed path

- **P-031 · proposed 2026-11-01 · Make G9 teacher RL the primary research path after S2.** A proposal.
  - authorizes: s2_pilot_packet sha256:5d687c931fe121319227f886f20d92f459e8d373a3251ee5e83abc9ed710e0d3

## CPU build

- **D-024 · 2026-11-08 · Build the remaining S2 code and the S4 engineering harness on CPU before any further rental.**
  - **Unchanged.** No rental follows from this decision. It mentions s2_pilot_packet sha256:b59b238b529029cc6841b05ce444cab0b99b69510438d37e619f5c599f083ee0 in prose only.

## Synthetic pilot approval

- **D-025 · 2026-11-09 · Synthetic pilot rental approval.** Huy approves the bound packet.
  - authorizes: s2_pilot_packet sha256:b59b238b529029cc6841b05ce444cab0b99b69510438d37e619f5c599f083ee0
  - authorizes: s2_readiness_report sha256:02f0725785c709d910339e5fb09c4aaa8a940cd37aef27636315611e3f7a1137
  - engineering_only_diagnostic: true
  - records: terms:harmonizer
  - records: acquire:teacher
  - **D-025 implementation addendum · 2026-11-09.** A nested addendum is part of D-025, not a new entry.
    - records: host_setup

### D-025 review addendum · 2026-11-09

Addendum prose under a level-3 heading.

- **D-026 · 2026-11-09 · A different approval.** Binds another packet.
  authorizes: s2_pilot_packet sha256:5d687c931fe121319227f886f20d92f459e8d373a3251ee5e83abc9ed710e0d3

- **D-027 · 2026-11-09 · Quoted and fenced hashes.** None of these lines is the exact grant form.
  - `authorizes: s2_pilot_packet sha256:cee7932aad8c569b76cd7c8bcd7d79c55ea984e042cc95fc8d74cbaa27fe96d2`
  > authorizes: s2_pilot_packet sha256:cee7932aad8c569b76cd7c8bcd7d79c55ea984e042cc95fc8d74cbaa27fe96d2
  - Huy wrote "authorizes: s2_pilot_packet sha256:cee7932aad8c569b76cd7c8bcd7d79c55ea984e042cc95fc8d74cbaa27fe96d2" as an example.
  - **authorizes:** s2_pilot_packet sha256:cee7932aad8c569b76cd7c8bcd7d79c55ea984e042cc95fc8d74cbaa27fe96d2
  ```
  authorizes: s2_pilot_packet sha256:874b585a236f3ca5a97c99d5b38e30f1e06644217ff821405233e1c29b266803
  records: terms:renderer
  ```
