package org.firstinspires.ftc.teamcode.biobuzz;

import com.qualcomm.robotcore.eventloop.opmode.Autonomous;

import org.firstinspires.ftc.teamcode.biobuzz.PollenNectarProcessor.Alliance;

/** Collects pollen and blue nectar; avoids the other alliance's nectar. */
@Autonomous(name = "Biobuzz: Hunt - BLUE", group = "Biobuzz")
public class PollenNectarAutoBlue extends PollenNectarAuto {
    @Override
    protected Alliance alliance() {
        return Alliance.BLUE;
    }
}
