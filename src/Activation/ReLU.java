package Activation;

import java.io.Serial;
import java.io.Serializable;

public class ReLU implements ActivationFunction, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    @Override
    public double activate(double input) {
        return Math.max(0.0, input);
    }

    @Override
    public double derivative(double input) {
        return input > 0.0 ? 1.0 : 0.0;
    }
}