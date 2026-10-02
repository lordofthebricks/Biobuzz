package org.firstinspires.ftc.teamcode.biobuzz;

import android.util.Size;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.firstinspires.ftc.vision.VisionPortal;

import java.util.List;

/**
 * Drive around with gamepad 1 and watch what the Teachable Machine model reports for every
 * grid cell. Use this to check the model, tune minConfidence, and find a good NEAR_Y value
 * (park the robot where it should start collecting and read the sighting's y).
 *
 * Camera stream: Driver Station menu -> Camera Stream (during INIT), or the Robot Controller screen.
 */
@TeleOp(name = "Biobuzz: Vision Test", group = "Biobuzz")
public class PollenNectarVisionTest extends LinearOpMode {

    @Override
    public void runOpMode() throws InterruptedException {
        MecanumDrive drive = new MecanumDrive(hardwareMap);

        PollenNectarProcessor.Settings settings = new PollenNectarProcessor.Settings();
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
        telemetry.addLine("Press START. Left stick = drive/strafe, right stick X = turn.");
        telemetry.update();
        waitForStart();

        List<String> labels = vision.getLabels();
        while (opModeIsActive()) {
            drive.drive(-gamepad1.left_stick_y, gamepad1.left_stick_x, gamepad1.right_stick_x);

            PollenNectarProcessor.Result r = vision.getLatestResult();
            if (r == null) {
                telemetry.addLine("Waiting for camera...");
            } else {
                telemetry.addData("FPS", "%.1f", portal.getFps());
                telemetry.addData("Result age", "%.0f ms", r.ageMillis());
                for (PollenNectarProcessor.Sighting s : r.sightings.values()) {
                    telemetry.addData("SIGHTING", s.toString());
                }
                for (PollenNectarProcessor.Cell cell : r.cells) {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < labels.size(); i++) {
                        sb.append(String.format("%s %.0f%%  ", labels.get(i), cell.scores[i] * 100));
                    }
                    telemetry.addData("Cell c" + cell.column + " r" + cell.row, sb.toString());
                }
            }
            telemetry.update();
        }

        drive.stop();
        portal.close();
        vision.close();
    }
}
