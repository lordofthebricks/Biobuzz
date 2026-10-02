package org.firstinspires.ftc.teamcode.biobuzz;

import com.qualcomm.hardware.rev.RevHubOrientationOnRobot;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.DcMotorSimple;
import com.qualcomm.robotcore.hardware.HardwareMap;
import com.qualcomm.robotcore.hardware.IMU;
import com.qualcomm.robotcore.util.Range;

import org.firstinspires.ftc.robotcore.external.navigation.AngleUnit;

/**
 * Robot-centric four-wheel mecanum drive with an IMU for heading.
 *
 * Conventions used everywhere in this package:
 *   forward  > 0  drives forward
 *   strafe   > 0  strafes right
 *   turn     > 0  rotates clockwise (turns right), same as a gamepad right stick
 *   heading       IMU yaw in degrees, counter-clockwise positive (FTC standard)
 *
 * Motor names in the robot configuration: frontLeft, frontRight, backLeft, backRight, imu.
 */
public class MecanumDrive {

    // ---- Change these to match how your Control Hub is mounted ----
    public static RevHubOrientationOnRobot.LogoFacingDirection LOGO_DIRECTION =
            RevHubOrientationOnRobot.LogoFacingDirection.UP;
    public static RevHubOrientationOnRobot.UsbFacingDirection USB_DIRECTION =
            RevHubOrientationOnRobot.UsbFacingDirection.FORWARD;

    private final DcMotor frontLeft, frontRight, backLeft, backRight;
    private final IMU imu;

    public MecanumDrive(HardwareMap hardwareMap) {
        frontLeft  = hardwareMap.get(DcMotor.class, "frontLeft");
        frontRight = hardwareMap.get(DcMotor.class, "frontRight");
        backLeft   = hardwareMap.get(DcMotor.class, "backLeft");
        backRight  = hardwareMap.get(DcMotor.class, "backRight");

        // Typical mecanum build: left side motors are mounted mirrored, so reverse them.
        // If the robot spins when you push "forward", swap these.
        frontLeft.setDirection(DcMotorSimple.Direction.REVERSE);
        backLeft.setDirection(DcMotorSimple.Direction.REVERSE);
        frontRight.setDirection(DcMotorSimple.Direction.FORWARD);
        backRight.setDirection(DcMotorSimple.Direction.FORWARD);

        for (DcMotor m : new DcMotor[]{frontLeft, frontRight, backLeft, backRight}) {
            m.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
            m.setMode(DcMotor.RunMode.STOP_AND_RESET_ENCODER);
            m.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        }

        imu = hardwareMap.get(IMU.class, "imu");
        imu.initialize(new IMU.Parameters(new RevHubOrientationOnRobot(LOGO_DIRECTION, USB_DIRECTION)));
        imu.resetYaw();
    }

    /** Robot-centric drive. All inputs are roughly -1..1; output is normalized so no wheel exceeds 1. */
    public void drive(double forward, double strafe, double turn) {
        double fl = forward + strafe + turn;
        double bl = forward - strafe + turn;
        double fr = forward - strafe - turn;
        double br = forward + strafe - turn;

        double max = Math.max(1.0, Math.max(Math.max(Math.abs(fl), Math.abs(bl)),
                                            Math.max(Math.abs(fr), Math.abs(br))));
        frontLeft.setPower(fl / max);
        backLeft.setPower(bl / max);
        frontRight.setPower(fr / max);
        backRight.setPower(br / max);
    }

    public void stop() {
        drive(0, 0, 0);
    }

    /** IMU yaw in degrees, counter-clockwise positive, range (-180, 180]. */
    public double getHeading() {
        return imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.DEGREES);
    }

    public void resetHeading() {
        imu.resetYaw();
    }

    /**
     * Turn command (clockwise-positive, suitable for {@link #drive}) that rotates the robot
     * toward {@code targetHeadingDeg} using a proportional controller.
     */
    public double turnToward(double targetHeadingDeg, double kP, double maxTurn) {
        double error = angleWrap(targetHeadingDeg - getHeading()); // + means target is CCW (left)
        return Range.clip(-kP * error, -maxTurn, maxTurn);         // CCW needs negative (CW+) turn
    }

    /** Wraps an angle in degrees into (-180, 180]. */
    public static double angleWrap(double degrees) {
        while (degrees > 180) degrees -= 360;
        while (degrees <= -180) degrees += 360;
        return degrees;
    }
}
