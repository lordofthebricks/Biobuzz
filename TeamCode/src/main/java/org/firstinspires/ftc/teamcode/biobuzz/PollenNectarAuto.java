package org.firstinspires.ftc.teamcode.biobuzz;

import android.util.Size;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.util.ElapsedTime;
import com.qualcomm.robotcore.util.Range;

import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.firstinspires.ftc.vision.VisionPortal;

import org.firstinspires.ftc.teamcode.biobuzz.PollenNectarProcessor.Alliance;
import org.firstinspires.ftc.teamcode.biobuzz.PollenNectarProcessor.Sighting;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Autonomous that searches the field for pollen (yellow) and our alliance's nectar, drives to
 * each one it sees and runs the intake to collect it, while steering clear of the opposing
 * alliance's nectar (we may not possess it).
 *
 * Run {@link PollenNectarAutoRed} or {@link PollenNectarAutoBlue} from the Driver Station.
 *
 * State machine:
 *   SEARCH     rotate in steps (pausing so the camera gets a clear frame); if a full circle
 *              finds nothing, drive forward a short leg and search again
 *   ALIGN      turn in place until the target is centered in the image
 *   APPROACH   drive forward, steering to keep the target centered, until it is low in the image;
 *              if opposing nectar is in the way, strafe sideways around it
 *   COLLECT    creep forward with the intake running
 *   EJECT      opposing nectar got near the intake: run the intake backward and reverse
 *   BACK_OFF   reverse a little, then turn away so the same spot is not re-detected
 *   PARK/DONE  out of time or collected enough
 */
public abstract class PollenNectarAuto extends LinearOpMode {

    public enum TargetMode { POLLEN, NECTAR, EITHER }

    // ======================= Tuning =======================
    public static String WEBCAM_NAME = "Webcam 1";
    public static TargetMode TARGET_MODE = TargetMode.EITHER;
    public static String POLLEN_LABEL = "Pollen";   // must match Teachable Machine class names
    public static String NECTAR_LABEL = "Nectar";

    public static int TARGETS_TO_COLLECT = 3;
    public static double TIME_LIMIT_S = 28.0;       // leave a little of the 30 s for parking

    // Save candidate crops every N camera frames for retraining (0 = off). Turn this on in
    // practice/scrimmage matches to collect training images from real gameplay.
    public static int CAPTURE_EVERY_N_FRAMES = 0;

    // Vision filtering
    public static int CONFIRM_FRAMES = 2;           // consecutive frames before a sighting is trusted
    public static double STALE_RESULT_MS = 500;     // ignore results older than this
    public static double LOST_TIMEOUT_S = 0.75;     // how long a target may disappear before giving up

    // Search
    public static double SCAN_STEP_DEG = 30;
    public static double SCAN_DWELL_S = 0.35;       // pause after each step so frames aren't blurred
    public static double SEARCH_LEG_POWER = 0.4;
    public static double SEARCH_LEG_TIME_S = 0.9;
    public static int MAX_SEARCH_LEGS = 4;

    // Heading control
    public static double HEADING_KP = 0.02;         // turn power per degree of error
    public static double MAX_TURN = 0.4;
    public static double HEADING_TOLERANCE_DEG = 4;

    // Align / approach (x is -1..1 across the image, y is 0 top .. 1 bottom)
    public static double ALIGN_KP = 0.35;
    public static double ALIGN_MIN_TURN = 0.08;
    public static double ALIGN_TOLERANCE = 0.2;
    public static double APPROACH_POWER = 0.35;
    public static double APPROACH_STEER_KP = 0.3;
    public static double NEAR_Y = 0.85;             // object's bottom edge this low in the image = close enough
    public static double APPROACH_TIMEOUT_S = 4.0;

    // Opposing nectar avoidance
    public static double PATH_HALF_WIDTH = 0.3;     // half the robot's intake width, in image x units, near the robot
    public static double INTAKE_ZONE_Y = 0.75;      // opposing nectar below this line and in the path = danger
    public static double HAZARD_MEMORY_S = 0.6;     // keep treating a hazard as present after it leaves view
    public static double AVOID_STRAFE_POWER = 0.35;
    public static double AVOID_TIMEOUT_S = 2.0;     // give up on a target if we can't get around in time

    // Collect
    public static double COLLECT_POWER = 0.25;
    public static double COLLECT_TIME_S = 0.8;
    public static double INTAKE_POWER = 1.0;
    public static double EJECT_TIME_S = 0.6;
    public static double BACK_OFF_POWER = 0.3;
    public static double BACK_OFF_TIME_S = 0.4;
    public static double TURN_AWAY_DEG = 60;
    // ======================================================

    private enum State { SEARCH, ALIGN, APPROACH, COLLECT, EJECT, BACK_OFF, TURN_AWAY, PARK, DONE }
    private enum SearchPhase { TURN, DWELL, LEG }

    /** Which alliance this OpMode plays for. */
    protected abstract Alliance alliance();

    private MecanumDrive drive;
    private DcMotor intake;  // optional
    private PollenNectarProcessor vision;
    private VisionPortal portal;

    private State state = State.SEARCH;
    private SearchPhase searchPhase = SearchPhase.DWELL;
    private final ElapsedTime matchTimer = new ElapsedTime();
    private final ElapsedTime stateTimer = new ElapsedTime();
    private final ElapsedTime lastSeenTimer = new ElapsedTime();
    private final ElapsedTime clock = new ElapsedTime();
    private double lastIntakeHazardS = -1e9;   // clock time opposing nectar was last in front of the intake
    private final ElapsedTime avoidTimer = new ElapsedTime();

    private double scanTargetHeading;
    private int scanStepsDone;
    private int searchLegs;
    private double holdHeading;
    private double turnAwayHeading;
    private int collected;
    private boolean avoiding;

    private long lastFrameTimestamp;
    private int consecutiveHits;
    private Sighting lastSighting;
    private List<Sighting> hazards = Collections.emptyList();

    @Override
    public void runOpMode() {
        drive = new MecanumDrive(hardwareMap);
        intake = hardwareMap.tryGet(DcMotor.class, "intake");

        PollenNectarProcessor.Settings settings = new PollenNectarProcessor.Settings();
        settings.alliance = alliance();
        settings.pollenLabel = POLLEN_LABEL;
        settings.nectarLabel = NECTAR_LABEL;
        settings.autoCaptureEveryNFrames = CAPTURE_EVERY_N_FRAMES;
        try {
            vision = new PollenNectarProcessor(hardwareMap.appContext, settings);
        } catch (Exception e) {
            telemetry.addLine("Could not load Teachable Machine model: " + e.getMessage());
            telemetry.addLine("Copy " + settings.modelFile + " and " + settings.labelsFile + " to "
                    + TeachableMachineClassifier.SDCARD_MODEL_DIR);
            telemetry.update();
            waitForStart();
            return;
        }

        portal = new VisionPortal.Builder()
                .setCamera(hardwareMap.get(WebcamName.class, WEBCAM_NAME))
                .setCameraResolution(new Size(640, 480))
                .addProcessor(vision)
                .enableLiveView(true)
                .build();

        List<String> wanted = wantedLabels();

        // Show what the camera sees while waiting, so the team can sanity-check the model.
        while (opModeInInit()) {
            PollenNectarProcessor.Result r = vision.getLatestResult();
            telemetry.addData("Alliance", alliance());
            telemetry.addData("Model labels", vision.getLabels());
            telemetry.addData("Hunting", wanted);
            telemetry.addData("Intake motor", intake != null ? "found" : "not configured");
            telemetry.addData("Targets", r == null ? "(no frames yet)" : r.sightings.values());
            telemetry.addData("Avoiding", r == null ? "" : r.hazards);
            telemetry.update();
            sleep(50);
        }
        if (isStopRequested()) {
            shutdown();
            return;
        }

        matchTimer.reset();
        drive.resetHeading();
        startSearch();

        while (opModeIsActive() && state != State.DONE) {
            Sighting sighting = updateVision(wanted);
            boolean confirmed = sighting != null && consecutiveHits >= CONFIRM_FRAMES;

            if (state != State.PARK && state != State.EJECT
                    && (matchTimer.seconds() > TIME_LIMIT_S || collected >= TARGETS_TO_COLLECT)) {
                setState(State.PARK);
            }

            switch (state) {
                case SEARCH:    search(confirmed); break;
                case ALIGN:     align(); break;
                case APPROACH:  approach(sighting); break;
                case COLLECT:   collect(); break;
                case EJECT:     eject(); break;
                case BACK_OFF:  backOff(); break;
                case TURN_AWAY: turnAway(); break;
                case PARK:      park(); break;
                default:        break;
            }

            telemetry.addData("Alliance", alliance());
            telemetry.addData("State", state + (state == State.SEARCH ? " / " + searchPhase : "")
                    + (avoiding ? " (avoiding)" : ""));
            telemetry.addData("Collected", "%d / %d", collected, TARGETS_TO_COLLECT);
            telemetry.addData("Target", lastSighting == null ? "none" : lastSighting.toString());
            telemetry.addData("Hazards", hazards.size());
            telemetry.addData("Heading", "%.1f", drive.getHeading());
            telemetry.addData("Time", "%.1f", matchTimer.seconds());
            telemetry.update();
        }

        shutdown();
    }

    // ------------------------------------------------------------------ vision

    /**
     * Reads the newest vision result, tracks how many consecutive frames contained a target,
     * updates the hazard list, and returns the current sighting (or null).
     */
    private Sighting updateVision(List<String> wanted) {
        PollenNectarProcessor.Result r = vision.getLatestResult();
        if (r == null || r.ageMillis() > STALE_RESULT_MS) {
            consecutiveHits = 0;
            hazards = Collections.emptyList();
            return null;
        }
        hazards = r.hazards;
        if (intakeZoneHazard() != null) lastIntakeHazardS = clock.seconds();

        Sighting s = r.best(wanted);
        if (r.timestampNanos != lastFrameTimestamp) {  // only count each camera frame once
            lastFrameTimestamp = r.timestampNanos;
            consecutiveHits = (s != null) ? consecutiveHits + 1 : 0;
        }
        if (s != null) {
            lastSighting = s;
            lastSeenTimer.reset();
        }
        return s;
    }

    private boolean targetLost() {
        return lastSeenTimer.seconds() > LOST_TIMEOUT_S;
    }

    /** True if opposing nectar was near the front of the intake within the last HAZARD_MEMORY_S. */
    private boolean intakeDanger() {
        return clock.seconds() - lastIntakeHazardS < HAZARD_MEMORY_S;
    }

    /** Opposing nectar right in front of the intake, or null. */
    private Sighting intakeZoneHazard() {
        for (Sighting h : hazards) {
            if (h.y >= INTAKE_ZONE_Y && Math.abs(h.x) < h.halfWidth + PATH_HALF_WIDTH) return h;
        }
        return null;
    }

    /**
     * Opposing nectar between the robot and the target, or null. The robot's path in the image
     * runs from the bottom center (x=0, y=1) to the target (target.x, target.y).
     */
    private Sighting hazardInPath(Sighting target) {
        Sighting worst = null;
        for (Sighting h : hazards) {
            if (h.y < target.y - 0.05) continue;  // farther away than the target: not in the way
            double pathX = target.y >= 1 ? target.x : target.x * (1 - h.y) / (1 - target.y);
            if (Math.abs(h.x - pathX) < h.halfWidth + PATH_HALF_WIDTH) {
                if (worst == null || h.y > worst.y) worst = h;  // closest one matters most
            }
        }
        return worst;
    }

    private List<String> wantedLabels() {
        switch (TARGET_MODE) {
            case POLLEN: return Arrays.asList(POLLEN_LABEL);
            case NECTAR: return Arrays.asList(NECTAR_LABEL);
            default:     return Arrays.asList(POLLEN_LABEL, NECTAR_LABEL);
        }
    }

    // ------------------------------------------------------------------ states

    private void startSearch() {
        setState(State.SEARCH);
        searchPhase = SearchPhase.DWELL;  // look first before moving
        scanStepsDone = 0;
        scanTargetHeading = drive.getHeading();
        lastSighting = null;
        avoiding = false;
    }

    private void search(boolean confirmed) {
        if (confirmed) {
            drive.stop();
            setState(State.ALIGN);
            return;
        }

        switch (searchPhase) {
            case TURN:
                drive.drive(0, 0, drive.turnToward(scanTargetHeading, HEADING_KP, MAX_TURN));
                if (headingError(scanTargetHeading) < HEADING_TOLERANCE_DEG) {
                    drive.stop();
                    searchPhase = SearchPhase.DWELL;
                    stateTimer.reset();
                }
                break;

            case DWELL:
                drive.stop();
                if (stateTimer.seconds() < SCAN_DWELL_S) break;
                int stepsPerCircle = (int) Math.ceil(360.0 / SCAN_STEP_DEG);
                if (scanStepsDone < stepsPerCircle) {
                    scanStepsDone++;
                    scanTargetHeading = MecanumDrive.angleWrap(scanTargetHeading + SCAN_STEP_DEG);
                    searchPhase = SearchPhase.TURN;
                } else if (searchLegs < MAX_SEARCH_LEGS) {
                    // Full circle with nothing found: move to a new spot and look again
                    searchLegs++;
                    holdHeading = drive.getHeading();
                    searchPhase = SearchPhase.LEG;
                    stateTimer.reset();
                } else {
                    setState(State.PARK);
                }
                break;

            case LEG:
                drive.drive(SEARCH_LEG_POWER, 0, drive.turnToward(holdHeading, HEADING_KP, MAX_TURN));
                // End the leg early rather than drive over opposing nectar
                if (stateTimer.seconds() > SEARCH_LEG_TIME_S || intakeDanger()) {
                    drive.stop();
                    scanStepsDone = 0;
                    scanTargetHeading = drive.getHeading();
                    searchPhase = SearchPhase.DWELL;
                    stateTimer.reset();
                }
                break;
        }
    }

    private void align() {
        if (targetLost()) {
            startSearch();
            return;
        }
        double x = lastSighting.x;  // + means target is right of center -> turn clockwise (+)
        if (Math.abs(x) < ALIGN_TOLERANCE) {
            drive.stop();
            avoiding = false;
            setState(State.APPROACH);
            return;
        }
        double turn = ALIGN_KP * x;
        turn = Math.copySign(Math.max(Math.abs(turn), ALIGN_MIN_TURN), turn);
        drive.drive(0, 0, Range.clip(turn, -MAX_TURN, MAX_TURN));
    }

    private void approach(Sighting sighting) {
        if (targetLost()) {
            // If it was already low in the image it probably slid under the camera: collect it,
            // unless opposing nectar was just seen in front of the intake.
            if (lastSighting != null && lastSighting.y >= NEAR_Y - 0.15 && !intakeDanger()) {
                setState(State.COLLECT);
            } else {
                startSearch();
            }
            return;
        }
        if (stateTimer.seconds() > APPROACH_TIMEOUT_S) {
            startSearch();
            return;
        }

        double turn = Range.clip(APPROACH_STEER_KP * lastSighting.x, -MAX_TURN, MAX_TURN);

        // Opposing nectar in the way: stop driving forward and strafe sideways around it,
        // keeping the camera pointed at our target.
        Sighting blocker = hazardInPath(lastSighting);
        if (blocker != null || intakeDanger()) {
            if (!avoiding) {
                avoiding = true;
                avoidTimer.reset();
            }
            if (avoidTimer.seconds() > AVOID_TIMEOUT_S) {
                giveUpTarget();
                return;
            }
            double side = 1;  // strafe right by default
            if (blocker != null) {
                double pathX = lastSighting.y >= 1 ? lastSighting.x
                        : lastSighting.x * (1 - blocker.y) / (1 - lastSighting.y);
                side = blocker.x > pathX ? -1 : 1;  // hazard right of path -> go left, and vice versa
            }
            drive.drive(0, side * AVOID_STRAFE_POWER, turn);
            return;
        }
        avoiding = false;

        if (sighting != null && sighting.y >= NEAR_Y && consecutiveHits >= CONFIRM_FRAMES) {
            setState(State.COLLECT);
            return;
        }
        // Slow down as the target gets lower (closer) in the image
        double forward = APPROACH_POWER * Range.clip(1.2 - lastSighting.y, 0.4, 1.0);
        drive.drive(forward, 0, turn);
    }

    private void collect() {
        if (intakeDanger()) {
            // Opposing nectar is right in front of the intake: get rid of anything we grabbed
            setState(State.EJECT);
            return;
        }
        setIntake(INTAKE_POWER);
        drive.drive(COLLECT_POWER, 0, 0);
        if (stateTimer.seconds() > COLLECT_TIME_S) {
            drive.stop();
            setIntake(0);
            collected++;
            setState(State.BACK_OFF);
        }
    }

    private void eject() {
        setIntake(-INTAKE_POWER);
        drive.drive(-BACK_OFF_POWER, 0, 0);
        if (stateTimer.seconds() > EJECT_TIME_S) {
            drive.stop();
            setIntake(0);
            giveUpTarget();
        }
    }

    /** Abandon the current target and look elsewhere. */
    private void giveUpTarget() {
        drive.stop();
        avoiding = false;
        turnAwayHeading = MecanumDrive.angleWrap(drive.getHeading() + TURN_AWAY_DEG);
        setState(State.TURN_AWAY);
    }

    private void backOff() {
        drive.drive(-BACK_OFF_POWER, 0, 0);
        if (stateTimer.seconds() > BACK_OFF_TIME_S) {
            drive.stop();
            turnAwayHeading = MecanumDrive.angleWrap(drive.getHeading() + TURN_AWAY_DEG);
            setState(State.TURN_AWAY);
        }
    }

    private void turnAway() {
        drive.drive(0, 0, drive.turnToward(turnAwayHeading, HEADING_KP, MAX_TURN));
        if (headingError(turnAwayHeading) < HEADING_TOLERANCE_DEG || stateTimer.seconds() > 2.0) {
            drive.stop();
            searchLegs = 0;
            startSearch();
        }
    }

    private void park() {
        // Add your parking path here (e.g. drive to the observation zone). For now: stop safely.
        drive.stop();
        setIntake(0);
        setState(State.DONE);
    }

    // ------------------------------------------------------------------ helpers

    private void setState(State next) {
        state = next;
        stateTimer.reset();
    }

    private double headingError(double target) {
        return Math.abs(MecanumDrive.angleWrap(target - drive.getHeading()));
    }

    private void setIntake(double power) {
        if (intake != null) intake.setPower(power);
    }

    private void shutdown() {
        drive.stop();
        setIntake(0);
        if (portal != null) portal.close();
        if (vision != null) vision.close();
    }
}
