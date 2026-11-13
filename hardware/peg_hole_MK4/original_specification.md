# Peg and hole: modelling and printing

We have a policy trained in simulation to insert an 8 mm peg into a hole with a Franka arm. To
test it on the real robot we need a physical peg and hole that match the simulated ones. **Please
model and print the parts below.** The dimensions come from the meshes the simulator uses, so
please keep them as close as you can. Notes on what matters and what doesn't are below each part.

Units are mm. "Diametral clearance" is the hole diameter minus the peg diameter.

## What to deliver

1. **Peg** (×3; they're cheap, and one may get damaged).
2. **Hole base** (×1, plus a spare if there's time).
3. **Clearance test coupon** (print it first; see "Getting the hole size right").
4. **Grasp jig** (×1; optional, but it makes the robot runs repeatable).
5. The CAD source (STEP plus your native file), the STLs, and the **measurement table** at the end,
   filled in.

## 1. Peg

```
        ┌─┐  ← 0.5 × 45° chamfer
        │ │
        │ │   Ø 7.98
        │ │
        │ │  50.0 long
   ─ ─ ─│─│─ ─ ─  grasp line: 32.0 from the bottom
        │ │
        │ │
        └─┘  ← 0.5 × 45° chamfer (bottom)
```

| Feature | Value | Tolerance |
|---|---|---|
| Diameter | 7.98 | ±0.05 |
| Length | 50.0 | ±0.3 |
| Chamfers, both ends | 0.5 × 45° | not critical |
| Grasp line | 32.0 from the bottom end | Mark it with a fine marker; **don't** cut a groove |

- **Print it standing up** (axis vertical), so the cross-section comes out round. A peg printed on its
  side comes out oval with a flat face.
- 0.1–0.12 mm layers, 100% infill (or at least 6 walls), PLA or PETG. Use a brim, because a thin
  standing part can get knocked over. Sand the side lightly if layer lines catch. Don't sand the diameter down.
- **Alternative:** a standard **Ø8 × 50 mm steel dowel pin** (ISO 8734) with chamfered ends works as
  well or better. It's rounder and straighter, and its mass (about 20 g) matches the simulated peg
  (19 g). Ø8.00 in the 9.0 hole is 1.0 mm of clearance, the same as the target. If you can get one easily, get one.

## 2. Hole base

```
  side section                               top view
                 Ø 9.0 bore                ┌───────────────────────┐
              ┌──┐    ┌──┐ ← 0.5×45°       │   ╭───────────────╮   │
              │  │    │  │   chamfer       │   │               │   │
   boss       │  │    │  │                 │   │      ( )      │   │  boss: 24.4 × 24.4,
   25.0 tall  │  │    │  │  bore 25.0 deep │   │     Ø9.0      │   │  corner radius 7
              │  │    │  │                 │   ╰───────────────╯   │
         ┌────┘  └────┘  └────┐            │                       │
  plate  │                    │ 3.0        └───────────────────────┘
         └────────────────────┘              plate: at least 40 × 40
```

| Feature | Value | Tolerance |
|---|---|---|
| Bore diameter | **9.0** | **±0.05** (the most important number here) |
| Bore depth | 25.0, from the boss top down to the plate top (blind hole) | ±0.2 |
| Bore top chamfer | 0.5 × 45° | not critical, but keep it small: no bigger than 0.5 |
| Boss | 24.4 × 24.4 square, 7 corner radius, 25.0 tall above the plate | ±0.5 (a round Ø25–28 boss is fine) |
| Plate thickness | 3.0 | not critical |
| Plate size | at least 40 × 40; **extend it for clamping** (see below) | — |
| Boss top face | flat and square to the bore | this face is what the peg slides on |

- **Clamping.** The simulated hole is welded to the table, so the real one must not move at all.
  Extend the plate to about 100 × 100 and add 4 × Ø6.5 holes (for M6 bolts) near the corners, or leave
  flat tabs for C-clamps. Anything more than 15 mm from the bore centre and at or below plate level
  is free to change, because the peg never gets there.
- **Print it upright** (bore vertical), so the bore comes out round and the boss top is the last,
  cleanest surface. Use 0.12–0.16 mm layers, at least 4 walls, at least 40% infill, PLA or PETG.
  No supports are needed.
- The bore must go **straight down** and be **perpendicular** to the boss top. A tilted bore makes
  insertion much harder.

## 3. Getting the hole size right (print this first)

Printers usually make holes 0.1–0.3 mm too small, so a 9.0 hole in CAD won't print at 9.0. Before
printing the base, print a **coupon**: a small block about 10 mm tall with 4 vertical holes,
labelled **9.0, 9.1, 9.2, 9.3** in CAD. Use the same printer, material and settings as the final part.

1. Measure each printed hole with a pin gauge if we have them, or with the inside jaws of calipers
   at the top and at 5 mm down.
2. Take the CAD size whose hole **measures 9.0 ± 0.05**, and use it for the bore of the base.
3. Check by hand that the real peg drops into that hole freely with about 0.5 mm of play on each
   side, and doesn't rattle a lot.

The peg works the same way: if it measures large, scale its CAD diameter down by the error. What the
policy was trained on is the **clearance, about 1.0 mm** (anything from 0.9 to 1.1 is fine), so a
matched pair matters more than either exact number.

## 4. Grasp jig (optional)

The sim always grasps the peg **32 mm above its bottom end** (±2 mm). The jig holds the peg upright
so the robot can pick it up at that height every time.

- A block about 40 × 40 × 20 with a **Ø8.6 blind hole, 18 deep**, so the peg stands loosely with 32
  sticking out.
- Plus bolt holes or clamp tabs, like the base.
- Leave at least 25 mm of clear space around the hole at the top face, so the gripper fingers fit.

## Measurement table (please fill in and send back)

| Item | Measured |
|---|---|
| Printer, material, nozzle, layer height | |
| Peg diameter at 10 / 25 / 40 mm from the bottom | |
| Peg length | |
| Peg mass (g) | |
| CAD size used for the bore (from the coupon) | |
| Bore diameter, measured at the top / 5 mm down / 20 mm down | |
| Bore depth | |
| Boss height above the plate | |
| Does the peg drop in freely by hand? Any tight spot? | |
| Photos: both parts, the bore from above, the peg in the hole | |

Questions about why a number matters: see `../deploy/README.md`, or ask.
