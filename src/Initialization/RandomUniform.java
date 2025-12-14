package Initialization;

import java.util.Random;

public class RandomUniform implements WeightInitializer{

    Random random = new Random();

    @Override
    public double[] init(int nInput, int nOutput) {
        double[] weights = new double[nInput * nOutput];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextDouble() - 0.5;
        }
        return weights;
    }
}
