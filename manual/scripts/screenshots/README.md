# Manual screenshot recapture

Playwright scripts that drive the **live k8s app** and regenerate the manual's
screenshots into `/tmp/shots_mistudio` (and `/tmp/shots_millm`), matching the
filenames referenced under `manual/static/img/`.

## Run

Requires the Playwright skill's runner (or a local `playwright` install with
chromium). With the skill:

```bash
SKILL=~/.claude/plugins/cache/playwright-skill/playwright-skill/*/skills/playwright-skill
cd $SKILL
node run.js <repo>/manual/scripts/screenshots/capture-mistudio-part1.js   # browse/config/templates/training/extractions
node run.js <repo>/manual/scripts/screenshots/capture-mistudio-part2.js   # labeling/clusters/steering/monitor/settings
node run.js <repo>/manual/scripts/screenshots/capture-mistudio-features.js # extraction feature browser + star colors
node run.js <repo>/manual/scripts/screenshots/capture-feature-modal.js    # feature detail modal
node run.js <repo>/manual/scripts/screenshots/capture-millm.js            # miLLM dashboard/models/saes/profiles/settings
```

Then swap into place and rebuild:

```bash
cd <repo>/manual
for img in $(grep -rhoE 'miStudio_[A-Za-z0-9_.-]+\.jpg' docs/ | sort -u); do
  [ -f /tmp/shots_mistudio/$img ] && cp /tmp/shots_mistudio/$img static/img/$img
done
npm run build   # verify no broken image refs
```

## Notes / gotchas (learned 2026-07-22)

- Target: `k8s-mistudio.hitsai.local` / `k8s-millm.hitsai.local` (2x deviceScaleFactor).
- The **feature browser** lives inside a **Completed extraction card** — click the
  card's chevron to expand (`button:has(svg.lucide-chevron-down)`); the feature
  **detail modal** opens by clicking a feature table row's label cell
  (`table tbody tr > td:nth(1)`), NOT the id or a button.
- Run sections as **separate scripts**: a modal that fails to close can stall a
  single long script (fresh page load per section avoids the trap).
- Data-dependent panels capture whatever real data exists; the enhanced-labeling
  Queued/Completed transient states need a live labeling run to show mid-flight.
