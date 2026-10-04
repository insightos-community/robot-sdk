# Fixed R1Pro collision spheres

`r1pro_collision_spheres.json` stores centres and radii in metres in each named link frame. It contains the fixed robot geometry; held-object geometry remains generated from the actual held object.

Exported from the compact-hand configuration validated on 2026-09-24, `.integration/curobo-compact-hand-20260924/left-robot-config.json`. Includes the current shared R1Pro assets, compact fingers/palms, and existing wrist/camera/body coverage. The SDK loads this file directly and does not refit meshes during planning. Joint positions still transform each link through FK. The existing collision buffers and self-collision exclusions remain in the native configuration.

269 robot spheres, 30 links. cuRobo separately reserves 128 attached-object slots for the selected hand and 32 for the other hand, for a 429-slot model. Asset edits require an explicit offline regeneration and coverage/path validation before updating this file.

Source configuration SHA256: d8ab009fadcf4439537dee8005f63c8583b162b9f9e6b71b1a9e2e3c17a661bf
