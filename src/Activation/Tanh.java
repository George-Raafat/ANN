package Activation;

import java.io.Serial;
import java.io.Serializable;

public class Tanh implements ActivationFunction, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    @Override
    public double activate(double input) {
        return Math.tanh(input);
    }

    @Override
    public double derivative(double input) {
        double t = activate(input);
        return 1.0 - (t * t);
    }
}
