# Postable

**Every face, at its best.**

Group photos are never perfect — someone always blinks, grimaces, or looks away. Postable lets you pick the best version of each person's face from a burst of group shots and combines them into one photo.

Everything runs in your browser. Your photos are never uploaded anywhere.

## How it works

1. **Upload** two or more photos of the same group.
2. **Detect** — Postable finds every face in every photo using [face-api.js](https://github.com/justadudewhohacks/face-api.js) (SSD MobileNet v1 detection, 68-point landmarks, and face descriptors).
3. **Match** — faces are grouped by person across photos by comparing face descriptors (Euclidean distance below `0.5` counts as the same person).
4. **Choose** — pick a base photo for the overall framing and background, then tap the best face for each person.
5. **Compose** — each chosen face is aligned to the base photo using its eye positions (rotation, scale, translation), then blended in with a feathered elliptical mask.
6. **Download** the result.

## Running locally

There's no build step or dependencies to install. Serve the folder with any static file server:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>. The landing page is `index.html` and the app is `app.html`.

On first use, the face detection models (a few MB) download from jsDelivr, and your browser caches them after that.

## Project structure

| File          | Purpose                                         |
| ------------- | ----------------------------------------------- |
| `index.html`  | Landing page                                    |
| `landing.css` | Landing page styles                             |
| `app.html`    | The app (upload → face picker → result screens) |
| `app.js`      | Face detection, matching, and compositing logic |
| `styles.css`  | App styles                                      |

## Limitations

- Works best when the photos are taken from roughly the same position, such as a burst or several quick shots.
- Large changes in head angle or lighting between shots can leave visible seams.
- People must be detected in the base photo for their face to be swapped.
