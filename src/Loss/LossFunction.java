package Loss;

public interface LossFunction {
    double calculateLoss(double[] predictedOutputs, double[] expectedOutputs);

    double derivative(double predictedOutput, double expectedOutput);
}
