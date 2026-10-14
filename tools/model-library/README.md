# Model library viewer

Run `./tools/model-library/start.sh` from any directory, then open the printed
URL. `SAM3D_VIEWER_PORT` selects another port. The server binds to localhost and
serves the repository root. The viewer imports Three.js from `esm.sh`, so first
load requires network access.

The default catalog points to the repository's bundled crate. Use **Open GLB**
for another self-contained GLB, or edit `model-library.json` to list local
outputs. Model files resolve relative to this viewer directory; for example,
`../../outputs/example/mesh.glb`. Include `id`, `name`, `file`, `version`, and
`date`. Optional `camera` is a three-component direction. Optional `variants`
contain `id`, `label`, `file`, and `version`; `default_variant` chooses one.
`animation_clip` selects a clip by name, and `animation_label` names the play
control. Deep links accept `?model=ID&version=VARIANT_ID`.

The viewer supports orbit/zoom, reset, download, animation playback, reduced
motion preferences, and skeleton display. Generated GLBs, dated catalogs,
screenshots, and browser reports remain outside this reusable tool. Viewing a
rig does not establish anatomical or deformation quality.
