# AGENTS.md

## Project
Android app (Kotlin, Jetpack Compose) that listens to ambient sound and alerts deeply hard-of-hearing users when one of 6 danger sounds is detected: fire alarm, siren, glass breaking, doorbell, door knock, car horn.
Detection runs on-device with a YAMNet + EfficientAT + PANNs ensemble (TensorFlow Lite). Min SDK 26.

## Hard rules
- Never store, log, or transmit audio. Process it in memory only. No network calls for detection.
- Every status must use color + icon + text. Never rely on color alone.
- All screens must work with TalkBack (contentDescription, readable state labels).
- Detection runs in a Foreground Service and only while the user has started it.
- Keep the app to two screens: permission onboarding and main. Do not add a settings screen.

## Docs (read before making changes)
- `docs/prd.md`: product requirements, scope, screen states, acceptance criteria. Source of truth for behavior.
- `docs/backlog.md`: user stories with IDs (BL-xxx) and acceptance criteria. Reference the story ID in commits and PRs.

If a change conflicts with these docs, stop and ask before implementing.

## Conventions
- Use Compose for all UI. Keep state in a ViewModel.
- Detection logic goes behind an interface so it can be tested without a microphone.
- Thresholds are internal constants per sound class, not user settings.
- Write tests for state transitions: off, idle (green), detected (red), auto-return after 10s, tap-to-return.

## Out of scope for MVP
Settings screen, per-sound toggles, sensitivity control, notification history, boot auto-start, wearables, cloud, iOS, voice guidance, gas alarm.
