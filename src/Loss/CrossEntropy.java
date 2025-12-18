package Loss;

public class CrossEntropy implements LossFunction {

    private static final double EPSILON = 1e-15;

    @Override
    public double calculateLoss(double[] predictedOutputs, double[] expectedOutputs) {
        double loss = 0.0;
        for (int i = 0; i < predictedOutputs.length; i++) {
            double predicted = clamp(predictedOutputs[i]);
            double expected = expectedOutputs[i];
            loss -= expected * Math.log(predicted);
        }
        return loss;
    }

    @Override
    public double derivative(double predictedOutput, double expectedOutput) {
        predictedOutput = clamp(predictedOutput);
        return - (expectedOutput / predictedOutput);
    }
    private double clamp(double value) {
        return Math.max(EPSILON, Math.min(1.0 - EPSILON, value));
    }
}
