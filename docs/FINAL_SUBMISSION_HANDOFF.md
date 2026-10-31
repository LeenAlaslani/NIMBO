# NIMBO — final submission handoff (31 Oct 2026)

## Hackathon platform requirements (team announcement)

- [ ] 8–10 slide presentation: problem, solution, technical architecture, features, testing and value
- [ ] Demo video **3 minutes maximum** showing the working prototype
- [ ] GitHub source link: https://github.com/LeenAlaslani/NIMBO
- [ ] Upload slides/video/link to the hackathon platform
- [ ] **Click Publish** on the platform after uploading; saving drafts is not submission
- [ ] All above before **8:00 PM on 31 Oct 2026** (check event's local time convention)

## Repository handoff

- Git branch `judging-submission-20261031`: evidence and fixes for review; merge into `main` only after remote CI success
- The UI expects a **local** FastAPI backend. A public GitHub repository is **not a hosted public NIMBO demo**. Do not claim an online deployment URL.
- All model weights, raw benchmark logs, live customer data and local tokens must remain uncommitted.
- Check external model/data licensing and redistribution terms before public release; existing source publication does not establish rights to distribute model weights.
- Validate README links, `.github/workflows/ci.yml`, `docs/FINAL_JUDGING_EVIDENCE.md`, `docs/CROSS_DOMAIN_PUBLIC_EVIDENCE.md` and `docs/JUDGING_CLAIMS.md`.
- Public CI checks code and *summary consistency*; it does **not** reproduce the GPU/CPU benchmarks or operate a customer service.
- Independent end-user testing and production deployment have not been established.
