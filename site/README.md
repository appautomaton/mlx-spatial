# mlx-spatial landing page

Static GitHub Pages site for [mlx-spatial](https://github.com/appautomaton/mlx-spatial),
published at <https://appautomaton.renocrypt.com/mlx-spatial/>.

## Editing rule

**Nothing on this page carries a number or a status that has to be
maintained.** Face counts, timings, quality figures, and per-pipeline
readiness all live in the repository, where they are produced and where they
change. The page states what does not change — the task each pipeline
performs, the file it leaves behind, the shape of the mesh stage, the scope —
and links out for anything measured.

That is also why the page lists five pipelines and not six: a pipeline still
in development would need its badge revised the day it graduates. The
capability table in the root README carries that state instead, and the page
points at it.

## Stack

A single self-contained `index.html` — no build step, no framework. Styles and
the small amount of JS (theme toggle, mobile menu, scroll reveal) are inline.

- **Type:** Zodiak (display), Synonym (body), DM Mono (data). The display face
  is a high-contrast serif because the subject is *making objects* — it should
  read as carved, not as UI chrome.
- **Colour:** the XYZ axis convention every 3D tool shares — X red, Y green,
  Z blue — on the neutral grey of a viewport. Sections advance through the
  three axes, so scrolling walks the gizmo. The choice is semantic: anyone who
  has opened a 3D application reads those three colours before the words.
- **Signature:** the headline sits inside a bounding box whose corners
  converge onto it on load. Selecting an object in any 3D tool draws exactly
  this, and the fit is the idea — the library turns a flat image into
  something that has bounds.
- **Ground:** an isometric lattice rather than a square grid, so the page is
  in three dimensions before anything is read.
- **Theme:** light by default. Light/dark via `data-theme` on `<html>`,
  persisted in localStorage, falling back to the OS scheme. Deep-link with
  `?theme=light` / `?theme=dark`.
- **Icons:** hand-drawn SVG, no icon library. Two rules here are easy to break:
  - The dash rules select `.d`, not `[pathLength]`. Blink does not invalidate
    a camelCase attribute selector on an SVG child when a class lands on an
    ancestor, so selecting the attribute leaves every icon undrawn. A new
    stroke needs `class="d"` as well as `pathLength="100"`.
  - A filled shape carries its resting opacity in `--fill-o`, because a CSS
    `opacity` outranks the SVG presentation attribute and would otherwise
    flatten a translucent solid into an opaque blob.
- **Motion:** [Motion](https://motion.dev) 13.0.0 from jsDelivr, pinned. It
  only drives scroll reveals; the bounding box is pure CSS. The import is
  guarded — if the CDN fails, everything is revealed rather than left hidden.
- **Responsive:** breakpoints at 720px, 940px, and 1040px.
- `prefers-reduced-motion` is respected — every animation renders in its final
  state.

Contrast was measured on every text node in **both** themes, not just the one
the page loads in.

## The share card

`assets/og.png` is 2400×1260 (1.91:1 at 2×), rendered from `assets/og.html`
with the real webfonts rather than approximated. The source is committed so
the card can be re-cut when the headline changes:

1. open `site/assets/og.html` at a 1200×630 viewport, `deviceScaleFactor` 2;
2. wait for `document.fonts.ready` — screenshotting early bakes in the
   fallback face, which is the whole reason this is a real browser render;
3. capture the viewport to `site/assets/og.png`.

`og.html` carries `noindex` and is not in the sitemap; it deploys with the
rest of the directory only because the workflow uploads `site/` as-is.

## Deploy

Published by `.github/workflows/pages.yaml` on every push to `main` that
touches `site/`. There is no build: the workflow uploads this directory as-is.
The Pages source must be set to **GitHub Actions**. `.nojekyll` keeps Jekyll
out of the way.

## Local preview

```bash
python3 -m http.server -d site 8000
# open http://localhost:8000/
```
