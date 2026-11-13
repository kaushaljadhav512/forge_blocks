"""Parametric CadQuery source, millimetres. Tested with CadQuery 2.7.0.

Run: python cad_source.py
Custom calibrated bore: python cad_source.py --bore 9.16 --peg-diameter 7.98
Output is placed in ./exports (or --out). CAD sizes are NOT measured print sizes.
"""
from pathlib import Path
import argparse
import json
import math
import cadquery as cq

PLATE_SIZE = 100.0
PLATE_THICKNESS = 3.0
BOSS_SIZE = 24.4
BOSS_RADIUS = 7.0
BOSS_HEIGHT = 25.0
CHAMFER = 0.5
PEG_LENGTH = 50.0
MOUNT_SPACING = 80.0
MOUNT_DIAMETER = 6.5
COUPON_DIAMETERS = (9.0, 9.1, 9.2, 9.3)


def peg(diameter=7.98):
    return cq.Workplane("XY").circle(diameter / 2).extrude(PEG_LENGTH).edges("%CIRCLE").chamfer(CHAMFER)


def bore_tool(diameter, floor_z, top_z, x=0, y=0):
    """Flat-bottom blind bore with a 0.5 x 45 degree mouth chamfer."""
    cylinder = cq.Workplane("XY").workplane(offset=floor_z).center(x, y).circle(diameter / 2).extrude(top_z - floor_z)
    mouth = cq.Solid.makeCone(diameter / 2, diameter / 2 + CHAMFER, CHAMFER,
                              cq.Vector(x, y, top_z - CHAMFER), cq.Vector(0, 0, 1))
    return cylinder.union(cq.Workplane("XY").newObject([mouth]))


def hole_base(diameter=9.0):
    plate = cq.Workplane("XY").box(PLATE_SIZE, PLATE_SIZE, PLATE_THICKNESS, centered=(True, True, False))
    # Holes permit attachment to a sacrificial board; they do not require table holes.
    for x in (-MOUNT_SPACING / 2, MOUNT_SPACING / 2):
        for y in (-MOUNT_SPACING / 2, MOUNT_SPACING / 2):
            plate = plate.cut(cq.Workplane("XY").center(x, y).circle(MOUNT_DIAMETER / 2).extrude(PLATE_THICKNESS))
    boss = (cq.Workplane("XY").box(BOSS_SIZE, BOSS_SIZE, BOSS_HEIGHT, centered=(True, True, False))
            .edges("|Z").fillet(BOSS_RADIUS).translate((0, 0, PLATE_THICKNESS)))
    return plate.union(boss).cut(bore_tool(diameter, PLATE_THICKNESS, PLATE_THICKNESS + BOSS_HEIGHT))


def coupon():
    """10 mm high coupon, 7 mm blind bores and a 3 mm floor, like the base."""
    part = cq.Workplane("XY").box(80, 28, 10, centered=(True, True, False))
    for x, d in zip((-27, -9, 9, 27), COUPON_DIAMETERS):
        part = part.cut(bore_tool(d, 3, 10, x, 3))
        label = (cq.Workplane("XY").workplane(offset=9.5).center(x, -8)
                 .text(f"{d:.1f}", 4.0, 0.5, font="DejaVu Sans", kind="bold", combine=True))
        part = part.cut(label)
    return part


def key(number):
    return f"{number:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def verify(shape, expected_dimensions):
    solid = shape.val()
    assert solid.isValid(), "Invalid CAD solid"
    assert len(shape.solids().vals()) == 1, "Expected one connected solid"
    bb = solid.BoundingBox()
    sizes = [bb.xlen, bb.ylen, bb.zlen]
    assert all(abs(a - b) < 1e-5 for a, b in zip(sizes, expected_dimensions)), sizes
    return {"valid_solid": True, "solid_count": 1, "bounding_box_mm": sizes,
            "volume_mm3": solid.Volume()}


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bore", type=float, help="Single calibrated CAD bore diameter; default exports all four candidates")
    parser.add_argument("--peg-diameter", type=float, default=7.98)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "exports")
    args = parser.parse_args()
    if not 7 < args.peg_diameter < 9:
        parser.error("Peg diameter must be between 7 and 9 mm")
    if args.bore is not None and not 8 < args.bore < 11:
        parser.error("Bore diameter must be between 8 and 11 mm")
    args.out.mkdir(parents=True, exist_ok=True)
    parts = {f"peg_D{key(args.peg_diameter)}_L50": (peg(args.peg_diameter), (args.peg_diameter, args.peg_diameter, PEG_LENGTH)),
             "clearance_coupon_PRINT_FIRST": (coupon(), (80, 28, 10))}
    for diameter in ((args.bore,) if args.bore is not None else COUPON_DIAMETERS):
        parts[f"base_CAD_bore_{key(diameter)}_UNCALIBRATED"] = (hole_base(diameter), (100, 100, 28))
    report = {"units": "mm", "cadquery_version": cq.__version__, "physical_prints_verified": False,
              "notes": "CAD checks only. Select bore using actual coupon measurements; custom bore exports also require verification.", "parts": {}}
    for name, (part, dimensions) in parts.items():
        report["parts"][name] = verify(part, dimensions)
        cq.exporters.export(part, str(args.out / f"{name}.step"))
        cq.exporters.export(part, str(args.out / f"{name}.stl"), tolerance=0.005, angularTolerance=0.05)
    # Verify the blind bore floor and chamfer with point classifications.
    for diameter in ((args.bore,) if args.bore is not None else COUPON_DIAMETERS):
        s = hole_base(diameter).val()
        assert s.isInside(cq.Vector(0, 0, 2.9)), "Floor missing"
        assert not s.isInside(cq.Vector(0, 0, 3.1)), "Blind bore starts too high"
        assert not s.isInside(cq.Vector(diameter / 2 - .01, 0, 15)), "Bore too small"
        assert s.isInside(cq.Vector(diameter / 2 + .01, 0, 15)), "Bore too large"
        assert not s.isInside(cq.Vector(diameter / 2 + .2, 0, 27.75)), "Mouth chamfer missing"
        assert s.isInside(cq.Vector(diameter / 2 + .3, 0, 27.75)), "Mouth chamfer too large"
        boss_area = BOSS_SIZE ** 2 - (4 - math.pi) * BOSS_RADIUS ** 2
        r = diameter / 2
        chamfer_removed = math.pi * CHAMFER * (r * CHAMFER + CHAMFER ** 2 / 3)
        expected_volume = PLATE_SIZE ** 2 * PLATE_THICKNESS - 4 * math.pi * (MOUNT_DIAMETER / 2) ** 2 * PLATE_THICKNESS + boss_area * BOSS_HEIGHT - math.pi * r ** 2 * BOSS_HEIGHT - chamfer_removed
        assert abs(s.Volume() - expected_volume) < 1e-4, "Base volume mismatch"
    (args.out / "CAD_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
