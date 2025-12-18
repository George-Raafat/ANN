package Data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class DataHandler {
    private DataHandler() {

    }

    public static void featureScaling(double[][] inputs) {
        if (inputs == null || inputs.length == 0)
            return;

        int numFeatures = inputs[0].length;
        double[] min = new double[numFeatures];
        double[] max = new double[numFeatures];

        // Initialize min/max with first row
        for (int j = 0; j < numFeatures; j++) {
            min[j] = inputs[0][j];
            max[j] = inputs[0][j];
        }

        // Find min and max for each feature
        for (int i = 1; i < inputs.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (inputs[i][j] < min[j])
                    min[j] = inputs[i][j];
                if (inputs[i][j] > max[j])
                    max[j] = inputs[i][j];
            }
        }

        // Apply Min-Max scaling
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (max[j] != min[j]) {
                    inputs[i][j] = (inputs[i][j] - min[j]) / (max[j] - min[j]);
                } else {
                    inputs[i][j] = 1.0; // Handle case where all values are the same
                }
            }
        }
    }

    public static Dataset[] splitData(double[][] inputs, int[] labels, double trainRatio) {
        if (inputs == null || labels == null || inputs.length != labels.length) {
            throw new IllegalArgumentException("Invalid data or labels.");
        }

        int totalSize = inputs.length;
        int trainSize = (int) (totalSize * trainRatio);
        int testSize = totalSize - trainSize;

        // Create indices
        List<Integer> indices = new ArrayList<Integer>();
        for (int i = 0; i < totalSize; i++)
            indices.add(i);

        Collections.shuffle(indices);

        double[][] trainX = new double[trainSize][inputs[0].length];
        int[] trainY = new int[trainSize];
        double[][] testX = new double[testSize][inputs[0].length];
        int[] testY = new int[testSize];

        for (int i = 0; i < trainSize; i++) {
            trainX[i] = inputs[indices.get(i)];
            trainY[i] = labels[indices.get(i)];
        }

        for (int i = 0; i < testSize; i++) {
            testX[i] = inputs[indices.get(trainSize + i)];
            testY[i] = labels[indices.get(trainSize + i)];
        }

        return new Dataset[] { new Dataset(trainX, trainY), new Dataset(testX, testY) };
    }
}
