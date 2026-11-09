# Manifest schemas

Only header-only example schemas are distributed. Real images, masks and labels must remain outside the public repository. Paths resolve relative to the manifest CSV; physical width_mm/height_mm are calibrated field of view, not network resolution. IDs must be nonempty and image_id unique. sha256 identifies the original image bytes.

Segmentation records retain original_image_id, specimen_id and slab_id; all derivatives of a slab must share train/val/test split. Gradation records retain specimen_id, gradation_id, label_source and reference_id. Label source must distinguish design labels from independently measured sieve references. The six target columns are P0.075, P2.36, P4.75, P9.5, P13.2 and P16, percentages on a 0–100 scale. Stacking rejects zero/near-zero references for ordinary MAPE, rather than silently substituting epsilon.

Field records retain road_id/cross_section_id/point_id/patch_id. Image masks may be 2D integer TIFF/PNG with 0 background, or JSON-described disjoint binary-mask collections. See README for the feature input contract.
