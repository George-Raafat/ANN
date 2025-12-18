package Initialization;

import java.util.Random;

public class Xavier implements WeightInitializer {

    private final Random random = new Random();

    @Override
    public double[] init(int nInput, int nOutput) {
        double[] weights = new double[nInput * nOutput];

        double limit = Math.sqrt(6.0 / (nInput + nOutput));

        for (int i = 0; i < weights.length; ++i) {
            weights[i] = (random.nextDouble() * 2 * limit) - limit;
        }

        return weights;
    }
}
