# Upstream provenance

This skill was seeded from GEPA's `gepa-optimize-anything` Agent Skill:

- repository: <https://github.com/gepa-ai/gepa>
- upstream path: `.claude/skills/gepa-optimize-anything`
- pinned commit: `67da814e33328e6714c3636428d03c86adb66cd7`
- package release used here: <https://github.com/gepa-ai/gepa/releases/tag/v0.1.4>
- upstream license: <https://github.com/gepa-ai/gepa/blob/67da814e33328e6714c3636428d03c86adb66cd7/LICENSE>

The pinned upstream skill documents a newer development API. This repository-local adaptation was
rewritten around the introspected `gepa==0.1.4` API and adds Formalizer-specific dataset isolation,
transcript handling, and local artifact requirements. It deliberately removes newer multi-engine,
optimizer-level test-set, and `OptimizeAnythingConfig` instructions that do not apply to 0.1.4.
