# cuRobo Fixed Multi-Link Object Carrying

[English](FIXED_ATTACHMENTS.md) | [简体中文](FIXED_ATTACHMENTS.zh-CN.md)

The native Runtime uses `CuroboPlanner.prepare()` to generate a numeric snapshot on
the simulation main thread. Carried-object identity still uses the native assisted
grasp result; Skill inputs and the Runtime API are unchanged.

Single-link objects follow the original attachment flow. With multiple links,
`curobo_attachments.py` reads the joints in the object's USD subtree and requires all
links to be connected into one rigid whole through enabled `PhysicsFixedJoint`s.
Articulated joints, unconnected links, and joints connecting outside the object or to
the world continue to return `unsupported_attached_object`; an articulated joint is
not treated as a fixed connection even when it is currently stationary. `obj.joints`
is not used to determine fixed connections, because the native articulation view may
omit fixed joints.

`scene_geometry()` already includes the enabled collision meshes of all links with
their world-coordinate vertices. `prepare()` transforms them into the same planning
base frame and preserves each mesh name. The numeric thread continues to use cuRobo's
`attach_objects_to_robot(..., merge_meshes=True)`, merging all meshes into
gripper-local attached spheres based on the measured EEF pose while disabling the
corresponding world obstacles. On planning cleanup, detach restores all corresponding
obstacles; a new snapshot after release no longer attaches the object and reads its
current position instead. Fixed metadata links without collision geometry do not need
fabricated meshes.

The checked radio `wxnicr` uses two links: the body and a button metadata link,
connected by a fixed joint. The asset's existing 14 collision meshes all belong to the
body; the button link has no collision meshes.

`tests/conformance/test_curobo_attachments.py` covers single-link compatibility, fixed
connection chains, rejection of articulated/disabled/disconnected/external connections,
rotation and translation coordinate transforms, full-mesh attachment, and post-release
snapshots. USD-related tests require importable `pxr.Usd` / `pxr.UsdPhysics`; they are
skipped when these are missing.

Local native deployments load the SDK source through `PYTHONPATH`, so the Runtime must
be restarted for changes to take effect. Passing only the geometry and attachment
validation does not equal successful full carried-object trajectory planning or real
execution.
