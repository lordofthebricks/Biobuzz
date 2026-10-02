package org.firstinspires.ftc.teamcode.biobuzz;

import com.qualcomm.robotcore.eventloop.opmode.Autonomous;

import org.firstinspires.ftc.teamcode.biobuzz.PollenNectarProcessor.Alliance;

/** Collects pollen and red nectar; avoids the other alliance's nectar. */
@Autonomous(name = "Biobuzz: Hunt - RED", group = "Biobuzz")
public class PollenNectarAutoRed extends PollenNectarAuto {
    @Override
    protected Alliance alliance() {
        return Alliance.RED;
    }
}
