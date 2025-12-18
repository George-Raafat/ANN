package Activation;

public class Tanh implements ActivationFunction {

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
