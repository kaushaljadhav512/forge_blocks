"""Block and slot dimensions, shared by the asset generator and the task configs.

Pure Python on purpose: importable without Isaac Sim (laptop, robot PC, tests).

Frame conventions follow Factory's peg_insert task, which our task reuses:
  * Block (held asset) origin: centre of its bottom face.
  * Slot (fixed asset) origin: centre of the slot floor (where the block's bottom ends up).
  * Slot "tip" (FORGE's observation/action frame) = origin + SLOT_DEPTH in z, i.e. the opening.

TODO(measure): replace the placeholder dimensions with calipers on the real block and print.
"""

# --- Block -----------------------------------------------------------------------------------
BLOCK_WIDTH = 0.030  # [m] square cross-section (x and y)
BLOCK_HEIGHT = 0.065  # [m]
BLOCK_MASS = 0.030  # [kg]

# --- Slot ------------------------------------------------------------------------------------
SLOT_DEPTH = 0.020  # [m] floor to rim
SLOT_FLOOR_THICKNESS = 0.005  # [m] material under the floor
SLOT_WALL_THICKNESS = 0.010  # [m]
SLOT_MASS = 0.200  # [kg] irrelevant once the root is fixed, but PhysX wants one

# Total clearance (slot inner width - block width), in mm. Keys are used in task IDs.
CLEARANCES_MM = (5, 2, 1)

# Franka fingerpad length, copied from Factory's RobotCfg (fingertip frame sits this far above
# the pad bottom). Used only for the sanity check below.
FRANKA_FINGERPAD_LENGTH = 0.017608
FRANKA_MAX_FINGER_OPENING = 0.04  # [m] per finger


def slot_inner_width(clearance_mm: float) -> float:
    return BLOCK_WIDTH + clearance_mm * 1e-3


def slot_usd_name(clearance_mm: float) -> str:
    return f"slot_{clearance_mm:g}mm.usda"


BLOCK_USD_NAME = "block.usda"


def check() -> None:
    """Fail early on geometry that the Factory reset/grasp logic cannot handle."""
    # Factory opens each finger to diameter/2 * 1.25 before grasping.
    assert BLOCK_WIDTH / 2 * 1.25 <= FRANKA_MAX_FINGER_OPENING, "block too wide for the Franka gripper"
    # Factory grasps with the fingertip frame at (BLOCK_HEIGHT - fingerpad) above the block bottom, so
    # the pads cover the top ~fingerpad of the block. Fully inserted, they must stay above the rim.
    pad_bottom_when_inserted = BLOCK_HEIGHT - 2 * FRANKA_FINGERPAD_LENGTH
    assert pad_bottom_when_inserted > SLOT_DEPTH + 0.005, "fingers would hit the slot rim; use a taller block"


check()
