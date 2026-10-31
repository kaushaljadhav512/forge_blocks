"""Generate the block and slot USDs from forge_blocks/geometry.py.

The slot is built from box primitives (floor + 4 walls) instead of a concave mesh, so collisions
are exact and cheap even at 1 mm clearance.

Run either way:
    ./isaaclab.sh -p scripts/make_assets.py          # on the training PC
    pip install usd-core && python scripts/make_assets.py   # anywhere, no Isaac Sim needed
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    # Inside an Isaac Sim install, pxr is only importable once the app is up.
    from isaaclab.app import AppLauncher

    _app = AppLauncher(headless=True).app
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

from forge_blocks import geometry as g

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "forge_blocks", "assets")


def _new_stage(path: str, root_name: str):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())
    # Single rigid body that Isaac Lab can load as an Articulation (Factory loads both assets that way).
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    return stage, root


def _add_box(stage, path: str, size_xyz, center_xyz):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center_xyz))
    cube.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def make_block(path: str):
    stage, root = _new_stage(path, "Block")
    w, h = g.BLOCK_WIDTH, g.BLOCK_HEIGHT
    _add_box(stage, "/Block/body", (w, w, h), (0.0, 0.0, h / 2))  # origin = bottom-face centre
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(g.BLOCK_MASS)
    stage.GetRootLayer().Save()


def make_slot(path: str, clearance_mm: float):
    stage, root = _new_stage(path, "Slot")
    inner = g.slot_inner_width(clearance_mm)
    t, d, f = g.SLOT_WALL_THICKNESS, g.SLOT_DEPTH, g.SLOT_FLOOR_THICKNESS
    outer = inner + 2 * t
    # Origin = centre of the slot floor (top surface of the floor plate).
    _add_box(stage, "/Slot/floor", (outer, outer, f), (0.0, 0.0, -f / 2))
    off = (inner + t) / 2
    _add_box(stage, "/Slot/wall_px", (t, outer, d), (+off, 0.0, d / 2))
    _add_box(stage, "/Slot/wall_nx", (t, outer, d), (-off, 0.0, d / 2))
    _add_box(stage, "/Slot/wall_py", (inner, t, d), (0.0, +off, d / 2))
    _add_box(stage, "/Slot/wall_ny", (inner, t, d), (0.0, -off, d / 2))
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(g.SLOT_MASS)
    stage.GetRootLayer().Save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=ASSET_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    targets = [(os.path.join(args.out, g.BLOCK_USD_NAME), make_block, ())]
    targets += [(os.path.join(args.out, g.slot_usd_name(c)), make_slot, (c,)) for c in g.CLEARANCES_MM]
    for path, fn, fn_args in targets:
        if os.path.exists(path):
            os.remove(path)  # Usd.Stage.CreateNew refuses to overwrite
        fn(path, *fn_args)
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
