package Data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DataHandler {
    private DataHandler() {
    }

    public static void featureScaling(double[][] inputs) {
        validateInputs(inputs);

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
                    inputs[i][j] = 0.0; // Handle case where all values are the same
                }
            }
        }
    }

    public static void featureScaling(Dataset dataset) {
        featureScaling(dataset.inputs);
    }

    public static Dataset[] splitData(double[][] inputs, int[] labels, double trainRatio) {
        if (inputs == null || labels == null || inputs.length != labels.length) {
            throw new IllegalArgumentException("Invalid data or labels.");
        }

        int totalSize = inputs.length;
        int trainSize = (int) (totalSize * trainRatio);
        int testSize = totalSize - trainSize;

        // Create indices
        List<Integer> indices = new ArrayList<>();
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

        return new Dataset[]{new Dataset(trainX, trainY), new Dataset(testX, testY)};
    }

    public static Dataset[] splitData(Dataset dataset, double trainRatio) {
        return splitData(dataset.inputs, dataset.labels, trainRatio);
    }

    public static void replaceNaNWithMean(double[][] inputs) {
        if (inputs == null || inputs.length == 0)
            return;

        int numSamples = inputs.length;
        int numFeatures = inputs[0].length;

        double[] sum = new double[numFeatures];
        int[] count = new int[numFeatures];

        // 1. Compute sums and counts (ignore NaNs)
        for (double[] input : inputs) {
            for (int j = 0; j < numFeatures; j++) {
                double value = input[j];
                if (!Double.isNaN(value)) {
                    sum[j] += value;
                    count[j]++;
                }
            }
        }

        // 2. Compute means
        double[] mean = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            if (count[j] > 0) {
                mean[j] = sum[j] / count[j];
            } else {
                mean[j] = 0.0; // feature is entirely NaN
            }
        }

        // 3. Replace NaNs with mean
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (Double.isNaN(inputs[i][j])) {
                    inputs[i][j] = mean[j];
                }
            }
        }
    }

    public static void replaceNaNWithMean(Dataset dataset) {
        replaceNaNWithMean(dataset.inputs);
    }

    public static void validateInputs(double[][] inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("Inputs array is null.");
        }

        if (inputs.length == 0) {
            throw new IllegalArgumentException("Inputs array is empty.");
        }

        if (inputs[0].length == 0) {
            throw new IllegalArgumentException("Rows must have at least one feature.");
        }

        int numFeatures = inputs[0].length;

        for (int i = 0; i < inputs.length; i++) {
            double[] row = inputs[i];

            if (row == null) {
                throw new IllegalArgumentException("Row " + i + " is null.");
            }

            if (row.length != numFeatures) {
                throw new IllegalArgumentException(
                        "Inconsistent feature length at row " + i +
                                ": expected " + numFeatures + ", got " + row.length
                );
            }

            for (int j = 0; j < row.length; j++) {
                double value = row[j];
                if (Double.isNaN(value) || Double.isInfinite(value)) {
                    throw new IllegalArgumentException(
                            "Invalid value at row " + i + ", column " + j +
                                    ": " + value
                    );
                }
            }
        }
    }

    public static void validateInputRow(double[] input) {
        if (input == null) {
            throw new IllegalArgumentException("Input row is null.");
        }

        if (input.length == 0) {
            throw new IllegalArgumentException("Input row is empty.");
        }

        for (int j = 0; j < input.length; j++) {
            double value = input[j];
            if (Double.isNaN(value) || Double.isInfinite(value)) {
                throw new IllegalArgumentException(
                        "Invalid value at column " + j + ": " + value
                );
            }
        }
    }

    public static void validateLabels(int[] labels, int numClasses) {
        if (labels == null) {
            throw new IllegalArgumentException("Labels array is null.");
        }

        if (labels.length == 0) {
            throw new IllegalArgumentException("Labels array is empty.");
        }

        for (int i = 0; i < labels.length; i++) {
            int label = labels[i];
            if (label < 0 || label >= numClasses) {
                throw new IllegalArgumentException(
                        "Invalid label at index " + i + ": " + label
                );
            }
        }
    }
}
