# Biobuzz – FTC Pollen & Nectar Hunter

Autonomous code for an FTC robot with a **four-wheel mecanum drive** that uses a model trained in
[Google Teachable Machine](https://teachablemachine.withgoogle.com/) to find **pollen (yellow)** and
**our alliance's nectar (red or blue)** on the field, drive to it and collect it, while steering
clear of the other alliance's nectar, which we are not allowed to possess.

Everything lives in `TeamCode/src/main/java/org/firstinspires/ftc/teamcode/biobuzz/`. Copy that
folder into the `TeamCode` module of your
[FtcRobotController](https://github.com/FIRST-Tech-Challenge/FtcRobotController) project
(built and checked against SDK 12.0).

| File | What it does |
|---|---|
| `PollenNectarAutoRed.java` / `PollenNectarAutoBlue.java` | **The autonomous OpModes.** Pick the one for your alliance |
| `PollenNectarAuto.java` | Shared autonomous logic and tuning: search, align, approach, avoid, collect, repeat |
| `PollenNectarVisionTest.java` | TeleOp: watch the model's output, tune it, and **capture training images** |
| `PollenNectarProcessor.java` | VisionPortal processor: color filter → crop → classify → target positions |
| `TeachableMachineClassifier.java` | Loads and runs the Teachable Machine `.tflite` model |
| `MecanumDrive.java` | Mecanum drive and IMU heading |

---

## How it finds things

The field never looks the same twice. Robots, scored pieces, people at the wall and lighting
all change from match to match and during a match. A model shown whole scenes would have to
learn every possible background, so this code **never asks Teachable Machine to look at the
whole scene**:

```
 camera frame ──▶ 1. PROPOSE ──▶ 2. CROP ──▶ 3. CLASSIFY ──▶ x, y of each Pollen / Nectar
                  OpenCV color    tight square   Teachable Machine:
                  filter finds    crop around    Pollen / Nectar / Other
                  colored blobs   each blob
```

1. **Propose:** FTC tiles are gray, so cheap HSV color masks pick out **yellow**, **red** and
   **blue** blobs as candidates. Everything else is thrown away before the model runs.
2. **Crop:** each blob is cut out as a tight square with a little padding. The object fills
   most of the crop, so whatever is behind it barely affects the result.
3. **Classify:** the model only has to answer "is this blob Pollen, Nectar, or Other?" *Other*
   only needs to cover things that pass the color filter (robot parts, alliance tape, field
   elements), which is a much smaller set than "the whole field".
4. **Apply alliance rules:** the blob's color decides what a detection means:

   | Model says | Blob color | Result |
   |---|---|---|
   | Pollen | yellow | **Target** |
   | Nectar | our alliance's color | **Target** |
   | Nectar | the other alliance's color | **Hazard**: never collect, steer around it |
   | anything else | any | ignored |

   The model doesn't have to tell red nectar from blue nectar. The color mask already knows, so
   the **Nectar** class is trained with both colors. That gives it twice the examples.

A crop counts only if its top label scores at least `minConfidence` **and** beats the
runner-up by `minMargin`. If the model is unsure, the crop is ignored rather than chased.
The robot also needs `CONFIRM_FRAMES` frames in a row before it commits.

Each sighting gives:

* **x** (−1 … +1) shows how far left or right the object's center is. The robot turns until
  x ≈ 0.
* **y** (0 … 1) shows how low the object's bottom edge is in the image. With the camera
  tilted down, lower means closer. The robot drives until `y ≥ NEAR_Y`, then collects.

If there are several of the same target, it goes for the closest one.

### Staying away from the other alliance's nectar

Being cautious costs less than a possession penalty, so opposing nectar only needs a Nectar
score of `hazardConfidence` (0.4) to count as a hazard. Targets need 0.75 plus a clear margin.

* **On the way to a target:** if opposing nectar sits between the robot and the target (closer
  than the target and within `PATH_HALF_WIDTH` of the straight-line path), the robot stops
  driving forward. It **strafes sideways** away from the hazard while keeping the camera on the
  target. If it can't get around within `AVOID_TIMEOUT_S`, it gives up on that target.
* **Right in front of the intake** (below `INTAKE_ZONE_Y`, in the robot's path): the robot won't
  start collecting. If it's already collecting, it runs the **intake in reverse** and backs
  away (`EJECT`). The robot keeps remembering the hazard for `HAZARD_MEMORY_S` after it slides
  out of the camera's view underneath the robot.
* **Search legs** end early if opposing nectar appears in front of the intake.

### Autonomous state machine

```
SEARCH ──seen──▶ ALIGN ──centered──▶ APPROACH ──close──▶ COLLECT ─▶ BACK_OFF ─▶ TURN_AWAY ─┐
  ▲  turn 30°, pause, look;   │ lost           │  ▲ │                 │                       │
  │  after 360° drive a leg   │                │  │ ▼ opposing nectar │ opposing nectar       │
  │                           │                │ strafe around it     ▼ at the intake         │
  │                           │                │ (gives up after 2 s) EJECT (intake reverse)  │
  │                           ▼                ▼ lost / timeout       └──────▶ TURN_AWAY ─────┤
  └───────────────────────────┴────────────────┴──────────────────────────────────────────────┘
                     time up or enough collected ─▶ PARK ─▶ DONE
```

---

## 1. Train the model in Teachable Machine

The model is trained on **crops from the robot's own camera**, not on photos of the whole
field. The vision test OpMode saves those crops for you.

### First model (bootstrapping)

1. Go to <https://teachablemachine.withgoogle.com/> → **Get Started** → **Image Project** →
   **Standard image model**.
2. Create these classes. **The names must match** `POLLEN_LABEL` / `NECTAR_LABEL` in
   `PollenNectarAuto.java` (not case-sensitive):
   * `Pollen`
   * `Nectar`: **both red and blue** nectar in the same class
   * `Other`: yellow, red or blue things that *aren't* pollen or nectar, such as alliance
     tape, bumpers, red/blue field elements, robot parts and signs at the wall.
3. For a first model you can take close-up photos with a laptop webcam, holding each piece
   so it fills most of the square preview. Vary the angle, distance, lighting and what's
   behind it.
4. **Train Model**, then export it (step 5 below) and put it on the robot.

### Retraining from real gameplay (do this often)

This is how the model keeps up with a changing field:

1. Run **Biobuzz: Vision Test**. Drive around a field set up like a real match, with other
   robots, scored pieces and people at the walls. Press **A** to save a snapshot, or **B** to
   save continuously.
   For real matches and scrimmages, set `CAPTURE_EVERY_N_FRAMES` (for example 15) in
   `PollenNectarAuto.java` so the auto collects crops while it runs.
2. Pull the images off the Control Hub:
   `adb pull /sdcard/FIRST/biobuzz-captures/ ./captures`
3. Sort the crop files into Pollen / Nectar / Other folders. They are named by the color
   mask that found them (`*_yellow0.png`, `*_red1.png`, `*_blue2.png`). These are exactly the
   224×224 crops the model sees. Pay attention to the **mistakes**. Every false
   detection you put into *Other* teaches the model something new.
4. Upload each folder to its class in Teachable Machine. Use **Save project to Drive** so
   you can keep adding images all season. Then retrain and re-export.
5. Delete the old captures: `adb shell rm -r /sdcard/FIRST/biobuzz-captures/`

### Export

1. **Export Model** → **Tensorflow Lite** tab → **Floating point** → **Download my model**.
   The zip contains `model_unquant.tflite` and `labels.txt`. (*Quantized* also works and runs
   faster. If you use it, set `settings.modelFile` to that file's name, usually `model.tflite`.)

## 2. Put the model on the robot

Use either option. The code checks the Control Hub folder first, then the app's assets.

* **Control Hub folder (no rebuild needed to swap models):** copy both files to
  `/sdcard/FIRST/tflitemodels/` (for example with
  `adb push model_unquant.tflite labels.txt /sdcard/FIRST/tflitemodels/`).
* **Built into the app:** copy both files into `TeamCode/src/main/assets/`.

## 3. Add TensorFlow Lite to the build

FTC SDK 10 and later no longer include TensorFlow, so add it to `TeamCode/build.gradle`:

```gradle
dependencies {
    implementation project(':FtcRobotController')
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'   // <-- add this line
}
```

> On an older SDK (9.x or earlier) TensorFlow Lite is already included. Skip this step there,
> or you may get "duplicate class" errors.

## 4. Robot configuration

Run **Biobuzz: Hunt - RED** or **Biobuzz: Hunt - BLUE** to match your alliance. Check the alliance
on the Driver Station telemetry during INIT.


| Device | Name in config |
|---|---|
| Front-left / front-right / back-left / back-right motors | `frontLeft`, `frontRight`, `backLeft`, `backRight` |
| Control Hub IMU | `imu` |
| Webcam | `Webcam 1` |
| Intake motor (optional) | `intake` |

Mount the webcam at the front, **tilted down** so the floor in front of the robot fills the
image. The "close = low in the image" rule depends on this.

In `MecanumDrive.java`, set `LOGO_DIRECTION` / `USB_DIRECTION` to match how your hub is
mounted. If the robot doesn't drive forward on `drive(1, 0, 0)`, flip the motor directions.

## 5. Test, tune, run

1. Run **Biobuzz: Vision Test** (TeleOp). Open the camera stream on the Driver Station during
   INIT, or watch the Robot Controller screen. Press **X** to switch alliance.
   * **Gray box:** the color mask found it, but the model said it isn't a game piece.
   * **Yellow / red / blue box:** a target.
   * **Crossed-out box marked AVOID:** the other alliance's nectar.
   * **A game piece has no box at all:** the color mask missed it. Adjust its HSV range (see
     below) or lower `minCandidateArea`. **Check this at every venue.** Lighting changes colors.
   * **Too many gray boxes (low FPS):** tighten the HSV ranges or lower `maxCandidates`. Each
     candidate costs one model run.
   * **Wrong labels:** capture with **A**, add those crops to the right class, retrain.
   * Park the robot where collecting should start and note the sighting's `y`. Use that as
     `NEAR_Y`. Put opposing nectar just in front of the intake and check that its `y` is at least
     `INTAKE_ZONE_Y`. Also check that `PATH_HALF_WIDTH` roughly covers the intake's width.
2. Run **Biobuzz: Hunt - RED** or **Biobuzz: Hunt - BLUE** (Autonomous). Choose `TARGET_MODE` (`POLLEN`,
   `NECTAR`, `EITHER`) and adjust the tuning constants at the top of `PollenNectarAuto.java`.
3. Add your real parking path in `park()`.

### Adjusting HSV ranges

The defaults in `PollenNectarProcessor.Settings` (OpenCV scale: H 0–180, S 0–255, V 0–255) are:

| Color | Low (H, S, V) | High (H, S, V) |
|---|---|---|
| Yellow (pollen) | 15, 100, 100 | 35, 255, 255 |
| Red (nectar) | 0, 120, 70 | 8, 255, 255 |
| Red, second range (red wraps around the hue circle) | 170, 120, 70 | 180, 255, 255 |
| Blue (nectar) | 95, 120, 60 | 130, 255, 255 |

To change them, add lines like these in `PollenNectarAuto.runOpMode()` and
`PollenNectarVisionTest`, right after `new PollenNectarProcessor.Settings()`:

```java
settings.yellowRanges.clear();
settings.yellowRanges.add(new PollenNectarProcessor.HsvRange(18, 90, 90, 32, 255, 255));
```

If dim yellow pieces are missed, lower the yellow S/V minimums. If orange-ish red
pieces show up as yellow, raise the yellow H minimum.

### Tuning cheat sheet

| Symptom | Change |
|---|---|
| Robot chases things that aren't there | Capture them and add them to *Other*; raise `minMargin` / `CONFIRM_FRAMES` |
| Robot touches opposing nectar | Raise `PATH_HALF_WIDTH`, lower `INTAKE_ZONE_Y`, lower `hazardConfidence` |
| Robot avoids things that aren't opposing nectar | Add them to *Other*; raise `hazardConfidence` slightly |
| Robot gives up on targets too easily | Raise `AVOID_TIMEOUT_S`, lower `PATH_HALF_WIDTH` |
| Misses real targets | Check for a gray box first (color filter), then lower `minConfidence` |
| Spins right past targets | Lower `MAX_TURN`, raise `SCAN_DWELL_S`, lower `SCAN_STEP_DEG` |
| Wobbles while lining up | Lower `ALIGN_KP` / `APPROACH_STEER_KP`, raise `ALIGN_TOLERANCE` |
| Stops too early / too late before collecting | Adjust `NEAR_Y`, `COLLECT_TIME_S` |
| Finds the same target again after collecting | Raise `TURN_AWAY_DEG` |
