# Optional Marker Landing (Separated)

Marker landing is intentionally disabled in the default CS20 runtime.

This folder keeps the marker-related configuration separate so the main
workflow focuses on optical-flow and fusion accuracy checks.

To re-enable marker detection later, set in your launcher/runtime:

- `APRILTAG_ENABLED = True`
- `APRILTAG_DICT_NAME` (e.g. `DICT_4X4_100`)
- `APRILTAG_SIZE_M` (marker size in meters)

Current default CS20 mode keeps this off.
