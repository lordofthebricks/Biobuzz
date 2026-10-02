# Biobuzz – FTC Pollen & Nectar Hunter

Autonomous code for an FTC robot with a **four-wheel mecanum drive** that uses a model trained in
[Google Teachable Machine](https://teachablemachine.withgoogle.com/) to find pollen and/or nectar
on the field, drive to it, and collect it.

Everything lives in `TeamCode/src/main/java/org/firstinspires/ftc/teamcode/biobuzz/`. Copy that
folder into the `TeamCode` module of your
[FtcRobotController](https://github.com/FIRST-Tech-Challenge/FtcRobotController) project
(built and checked against SDK 12.0).

| File | What it does |
|---|---|
| `PollenNectarAuto.java` | **The autonomous.** Search, align, approach, collect, repeat |
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

1. **Propose:** FTC tiles are gray, so a cheap HSV color filter picks out colored blobs as
   candidates. By default it takes anything clearly colored. Narrow `candidateRanges` once you
   know the game pieces' colors.
2. **Crop:** each blob is cut out as a tight square with a little padding. The object fills
   most of the crop, so whatever is behind it barely affects the result.
3. **Classify:** the model only has to answer "is this blob Pollen, Nectar, or Other?" *Other*
   only needs to cover things that pass the color filter (robot parts, alliance tape, field
   elements), which is a much smaller set than "the whole field".

A crop counts only if its top label scores at least `minConfidence` **and** beats the
runner-up by `minMargin`. If the model is unsure, the crop is ignored rather than chased.
The robot also needs `CONFIRM_FRAMES` frames in a row before it commits.

Each sighting gives:

* **x** (−1 … +1) shows how far left or right the object's center is. The robot turns until
  x ≈ 0.
* **y** (0 … 1) shows how low the object's bottom edge is in the image. With the camera
  tilted down, lower means closer. The robot drives until `y ≥ NEAR_Y`, then collects.

If there are several of the same target, it goes for the closest one.

### Autonomous state machine

```
SEARCH ──seen──▶ ALIGN ──centered──▶ APPROACH ──close──▶ COLLECT ─▶ BACK_OFF ─▶ TURN_AWAY ─┐
  ▲  turn 30°, pause, look;   │ lost           │ lost / timeout                              │
  │  after 360° drive a leg   ▼                ▼                                             │
  └───────────────────────────┴────────────────┴─────────────────────────────────────────────┘
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
   * `Nectar`
   * `Other`: anything colored that *isn't* a target, such as robot parts, alliance-colored
     tape and field elements, bumpers, and the edges of other game pieces.
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
3. Sort the `*_crop*.png` files into Pollen / Nectar / Other folders. These are exactly the
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
   INIT, or watch the Robot Controller screen. **Gray boxes** are candidates from the color
   filter. **Yellow and cyan boxes** are crops the model accepted as Pollen or Nectar.
   * **A target has no box at all:** the color filter missed it. Widen `candidateRanges` or
     lower `minCandidateArea`.
   * **Too many gray boxes (low FPS):** narrow `candidateRanges` to the game pieces' colors,
     or lower `maxCandidates`. Each candidate costs one model run.
   * **Wrong labels:** capture with **A**, add those crops to the right class, retrain.
   * Park the robot where collecting should start and note the sighting's `y`. Use that as
     `NEAR_Y`.
2. Run **Biobuzz: Pollen/Nectar Hunt** (Autonomous). Choose `TARGET_MODE` (`POLLEN`,
   `NECTAR`, `EITHER`) and adjust the tuning constants at the top of `PollenNectarAuto.java`.
3. Add your real parking path in `park()`.

### Finding HSV ranges

OpenCV HSV uses H 0–180, S 0–255, V 0–255. Some starting points:

| Color | Low (H, S, V) | High (H, S, V) |
|---|---|---|
| Yellow | 15, 100, 100 | 35, 255, 255 |
| Orange | 5, 120, 100 | 18, 255, 255 |
| Blue | 95, 120, 60 | 130, 255, 255 |
| Purple | 130, 60, 60 | 160, 255, 255 |
| Red (wraps around) | 0, 120, 70 → 8, 255, 255 **and** 170, 120, 70 → 180, 255, 255 |

```java
// In PollenNectarAuto and PollenNectarVisionTest, right after "new PollenNectarProcessor.Settings()":
settings.candidateRanges.clear();
settings.candidateRanges.add(new PollenNectarProcessor.HsvRange(15, 100, 100, 35, 255, 255)); // pollen
settings.candidateRanges.add(new PollenNectarProcessor.HsvRange(95, 120, 60, 130, 255, 255)); // nectar
```

### Tuning cheat sheet

| Symptom | Change |
|---|---|
| Robot chases things that aren't there | Capture them and add them to *Other*; raise `minMargin` / `CONFIRM_FRAMES` |
| Misses real targets | Check for a gray box first (color filter), then lower `minConfidence` |
| Spins right past targets | Lower `MAX_TURN`, raise `SCAN_DWELL_S`, lower `SCAN_STEP_DEG` |
| Wobbles while lining up | Lower `ALIGN_KP` / `APPROACH_STEER_KP`, raise `ALIGN_TOLERANCE` |
| Stops too early / too late before collecting | Adjust `NEAR_Y`, `COLLECT_TIME_S` |
| Finds the same target again after collecting | Raise `TURN_AWAY_DEG` |
