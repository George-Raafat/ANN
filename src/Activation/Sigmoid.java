package Activation;

import java.io.Serial;
import java.io.Serializable;

public class Sigmoid implements ActivationFunction, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    @Override
    public double activate(double input) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    @Override
    public double derivative(double input) {
        double a = activate(input);
        return a * (1.0 - a);
    }
}
