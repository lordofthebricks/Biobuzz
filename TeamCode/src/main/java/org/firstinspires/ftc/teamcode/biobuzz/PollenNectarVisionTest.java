package org.firstinspires.ftc.teamcode.biobuzz;

import android.util.Size;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.firstinspires.ftc.vision.VisionPortal;

import java.util.List;
import java.util.Locale;

/**
 * Drive around with gamepad 1, watch what the model reports, and collect training images.
 *
 *   Left stick      drive / strafe          Right stick X   turn
 *   A               save this frame + every candidate crop to /sdcard/FIRST/biobuzz-captures/
 *   B               toggle continuous capture (a set of crops every 10 frames)
 *   X               switch alliance (RED / BLUE)
 *
 * Gray boxes are candidates the color filter found that the model rejected. Yellow / red / blue
 * boxes are targets. Crossed-out boxes marked AVOID are the opposing alliance's nectar.
 * Use this to tune the HSV ranges and minConfidence, and to find NEAR_Y and INTAKE_ZONE_Y
 * (park where collecting should start and read the sighting's y).
 *
 * Camera stream: Driver Station menu -> Camera Stream (during INIT), or the Robot Controller screen.
 */
@TeleOp(name = "Biobuzz: Vision Test", group = "Biobuzz")
public class PollenNectarVisionTest extends LinearOpMode {

    public static int CONTINUOUS_CAPTURE_EVERY_N_FRAMES = 10;

    @Override
    public void runOpMode() throws InterruptedException {
        MecanumDrive drive = new MecanumDrive(hardwareMap);

        PollenNectarProcessor.Settings settings = new PollenNectarProcessor.Settings();
        settings.pollenLabel = PollenNectarAuto.POLLEN_LABEL;
        settings.nectarLabel = PollenNectarAuto.NECTAR_LABEL;
        PollenNectarProcessor vision;
        try {
            vision = new PollenNectarProcessor(hardwareMap.appContext, settings);
        } catch (Exception e) {
            telemetry.addLine("Could not load model: " + e.getMessage());
            telemetry.update();
            waitForStart();
            return;
        }

        VisionPortal portal = new VisionPortal.Builder()
                .setCamera(hardwareMap.get(WebcamName.class, PollenNectarAuto.WEBCAM_NAME))
                .setCameraResolution(new Size(640, 480))
                .addProcessor(vision)
                .enableLiveView(true)
                .build();

        telemetry.addData("Labels", vision.getLabels());
        telemetry.addLine("Press START. A = capture, B = continuous capture, X = alliance.");
        telemetry.update();
        waitForStart();

        List<String> labels = vision.getLabels();
        boolean lastA = false, lastB = false, lastX = false;
        while (opModeIsActive()) {
            drive.drive(-gamepad1.left_stick_y, gamepad1.left_stick_x, gamepad1.right_stick_x);

            if (gamepad1.a && !lastA) vision.requestCapture();
            if (gamepad1.b && !lastB) {
                vision.setAutoCapture(vision.getAutoCapture() > 0 ? 0 : CONTINUOUS_CAPTURE_EVERY_N_FRAMES);
            }
            if (gamepad1.x && !lastX) {
                vision.setAlliance(vision.getAlliance() == PollenNectarProcessor.Alliance.RED
                        ? PollenNectarProcessor.Alliance.BLUE : PollenNectarProcessor.Alliance.RED);
            }
            lastX = gamepad1.x;
            lastA = gamepad1.a;
            lastB = gamepad1.b;

            telemetry.addData("Alliance", vision.getAlliance());
            telemetry.addData("Images saved", "%d  (continuous %s)", vision.getCapturesSaved(),
                    vision.getAutoCapture() > 0 ? "ON" : "off");

            PollenNectarProcessor.Result r = vision.getLatestResult();
            if (r == null) {
                telemetry.addLine("Waiting for camera...");
            } else {
                telemetry.addData("FPS", "%.1f", portal.getFps());
                telemetry.addData("Candidates", r.detections.size());
                for (PollenNectarProcessor.Sighting s : r.sightings.values()) {
                    telemetry.addData("TARGET", s.toString());
                }
                for (PollenNectarProcessor.Sighting h : r.hazards) {
                    telemetry.addData("AVOID", h.toString());
                }
                for (PollenNectarProcessor.Detection d : r.detections) {
                    StringBuilder sb = new StringBuilder(d.color + " " + d.role + ": ");
                    for (int i = 0; i < labels.size(); i++) {
                        sb.append(String.format(Locale.US, "%s %.0f%%  ", labels.get(i), d.scores[i] * 100));
                    }
                    telemetry.addData("@" + d.blob.x + "," + d.blob.y, sb.toString());
                }
            }
            telemetry.update();
        }

        drive.stop();
        portal.close();
        vision.close();
    }
}
