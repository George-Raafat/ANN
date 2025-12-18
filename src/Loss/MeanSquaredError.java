package Loss;

public class MeanSquaredError implements LossFunction {
    @Override
    public double calculateLoss(double[] predictedOutputs, double[] expectedOutputs) {
        double loss = 0;
        for (int i = 0; i < predictedOutputs.length; i++) {
            double error = predictedOutputs[i] - expectedOutputs[i];
            loss += error * error;
        }
        return 0.5 * loss;
    }

    @Override
    public double derivative(double predictedOutput, double expectedOutput) {
        return predictedOutput - expectedOutput;
    }
}
