# callcheck

callcheck is an open-source evaluation harness for voice agents. A simulated caller, driven by a
speech-to-speech model, places real calls to your agent over a streaming audio connection. Each
call is recorded and transcribed with timestamps, a judge scores it against a written rubric with
reasons, and a detector classifies turn-taking incidents as agent bugs or simulator artifacts.

**Status:** milestone 1 in progress, not yet usable.

Quickstart (coming):

```
callcheck run scenarios/intake_short.yaml --target nova-sonic --caller nova-sonic
```
