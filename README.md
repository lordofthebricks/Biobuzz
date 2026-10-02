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
| `PollenNectarVisionTest.java` | TeleOp: drive around and watch the model's output to check and tune it |
| `PollenNectarProcessor.java` | VisionPortal processor that turns classifications into target positions |
| `TeachableMachineClassifier.java` | Loads and runs the Teachable Machine `.tflite` model |
| `MecanumDrive.java` | Mecanum drive and IMU heading |

---

## How it finds things

Teachable Machine trains an image **classifier**: it says *what* is in a picture, not *where*.
To get a location, each camera frame is cut into a grid (3 columns × 2 rows by default), and
every cell is classified separately:

```
+---------+---------+---------+
|  Bkgnd  | Pollen  |  Bkgnd  |   top row    = far away
+---------+---------+---------+
|  Bkgnd  | Pollen  | Nectar  |   bottom row = close (camera tilted down)
+---------+---------+---------+
  turn left  straight  turn right
```

Cells that score above `minConfidence` are combined into a weighted centroid for each label:

* **x** (−1 … +1) shows how far left or right the target is. The robot turns until x ≈ 0.
* **y** (0 … 1) shows how low the target is in the image. Lower means closer. The robot drives
  forward until `y ≥ NEAR_Y`, then collects.

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

1. Go to <https://teachablemachine.withgoogle.com/> → **Get Started** → **Image Project** →
   **Standard image model**.
2. Create these classes. **The names must match** `POLLEN_LABEL` / `NECTAR_LABEL` in
   `PollenNectarAuto.java` (not case-sensitive):
   * `Pollen`
   * `Nectar`
   * `Background`: empty field tiles, walls, robots, field elements. **This class is
     essential.** Without it, every cell gets labeled pollen or nectar.
3. Collect images **with the robot's own webcam at its real mounting angle** if you can
   (for example, save frames from the vision test OpMode, or plug the webcam into a laptop). Aim for 150+ images
   per class, with:
   * game pieces at different distances, angles and positions in the frame
   * **partial views**, because the grid often cuts an object in half
   * close-up shots that roughly match one grid cell (about ⅓ of the frame width)
   * different lighting conditions
4. Click **Train Model**. Test with the preview and add more images where it's wrong.
5. **Export Model** → **Tensorflow Lite** tab → **Floating point** → **Download my model**.
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
   INIT, or watch the Robot Controller screen. Each grid cell shows its best label and score.
   * Raise `minConfidence` in `PollenNectarProcessor.Settings` if you see false detections.
     Lower it if real targets are missed.
   * Park the robot where collecting should start and note the sighting's `y`. Use that as
     `NEAR_Y`.
   * Watch the FPS. Each grid cell is one model run. If it's too slow, use a quantized model
     or fewer cells. For finer steering, try `columns = 5` if the FPS allows.
2. Run **Biobuzz: Pollen/Nectar Hunt** (Autonomous). Choose `TARGET_MODE` (`POLLEN`,
   `NECTAR`, `EITHER`) and adjust the tuning constants at the top of `PollenNectarAuto.java`.
3. Add your real parking path in `park()`.

### Tuning cheat sheet

| Symptom | Change |
|---|---|
| Robot chases things that aren't there | Raise `minConfidence`, raise `CONFIRM_FRAMES`, add more Background images |
| Spins right past targets | Lower `MAX_TURN`, raise `SCAN_DWELL_S`, lower `SCAN_STEP_DEG` |
| Wobbles while lining up | Lower `ALIGN_KP` / `APPROACH_STEER_KP`, raise `ALIGN_TOLERANCE` |
| Stops too early / too late before collecting | Adjust `NEAR_Y`, `COLLECT_TIME_S` |
| Finds the same target again after collecting | Raise `TURN_AWAY_DEG` |
