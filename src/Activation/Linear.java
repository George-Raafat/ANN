package Activation;

import java.io.Serial;
import java.io.Serializable;

public class Linear implements ActivationFunction, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    @Override
    public double activate(double input) {
        return input;
    }

    @Override
    public double derivative(double input) {
        return 1.0;
    }
}
