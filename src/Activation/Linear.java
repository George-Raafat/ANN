package Activation;

public class Linear implements ActivationFunction {

    @Override
    public double activate(double input) {
        return input;
    }

    @Override
    public double derivative(double input) {
        return 1.0;
    }
}
