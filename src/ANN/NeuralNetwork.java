package ANN;

import Activation.ActivationFunction;
import Activation.Sigmoid;
import Data.DataHandler;
import Initialization.WeightInitializer;
import Loss.LossFunction;
import Loss.MeanSquaredError;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NeuralNetwork implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;
    private final Layer[] layers;
    private final int numFeatures;
    private final int numClasses;
    private transient LossFunction lossFunction = new MeanSquaredError();
    private double learningRate = 0.1;
    private int batchSize = 100;
    private int epochs = 10;

    // Create the neural network
    public NeuralNetwork(int[] layerSizes, WeightInitializer weightInitializer, ActivationFunction activationFunction) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("Number of layers should be at least 2");
        }
        layers = new Layer[layerSizes.length - 1];
        for (int i = 0; i < layers.length; i++) {
            if (layerSizes[i] < 1) {
                throw new IllegalArgumentException("Number of neurons can't be less than 1");
            }
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], activationFunction, weightInitializer);
        }
        numFeatures = layerSizes[0];
        numClasses = layerSizes[layerSizes.length - 1];
        if (numClasses < 1) {
            throw new IllegalArgumentException("Number of neurons can't be less than 1");
        }
    }

    public NeuralNetwork(int[] layerSizes, WeightInitializer weightInitializer) {
        this(layerSizes, weightInitializer, new Sigmoid());
    }

    private void validateInputs(double[][] inputs) {
        DataHandler.validateInputs(inputs);
        if (inputs[0].length != numFeatures) {
            throw new IllegalArgumentException("Number of features doesn't match the ANN");
        }
    }

    private int maxValueIndex(double[] values) {
        double maxValue = Double.NEGATIVE_INFINITY;
        int index = 0;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > maxValue) {
                maxValue = values[i];
                index = i;
            }
        }
        return index;
    }

    private double[] feedForward(double[] inputs) {
        for (Layer layer : layers) {
            inputs = layer.calculateOutputs(inputs);
        }
        return inputs;
    }

    private void backwardPropagation(double[] trainingData, double[] expected) {
        double[] output = feedForward(trainingData);
        double[] subdeltas = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            double costDerivative = lossFunction.derivative(output[i], expected[i]);
            subdeltas[i] = costDerivative;
        }
        for (int i = layers.length - 1; i > 0; i--) {
            subdeltas = layers[i].accumulateGradients(subdeltas, layers[i - 1].getActivations());
        }
        layers[0].accumulateGradients(subdeltas, trainingData);
    }

    public void train(double[][] trainingData, int[] labelIndices) {
        validateInputs(trainingData);
        DataHandler.validateLabels(labelIndices, numClasses);
        if (trainingData.length != labelIndices.length) {
            throw new IllegalArgumentException("Number of Inputs doesn't match the outputs");
        }
        int n = trainingData.length;

        List<Integer> indices = new ArrayList<>(n);
        for (int i = 0; i < n; i++) indices.add(i);

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(indices);

            for (int i = 0; i < n; i += batchSize) {
                int end = Math.min(i + batchSize, n);

                for (int j = i; j < end; j++) {
                    int idx = indices.get(j);
                    double[] expected = oneHot(labelIndices[idx]);
                    backwardPropagation(trainingData[idx], expected);
                }

                for (Layer layer : layers) {
                    layer.applyGradients(learningRate, end - i);
                }
            }

            double epochLoss = 0.0;
            int trainCorrect = 0;

            for (int i = 0; i < n; i++) {
                double[] output = feedForward(trainingData[i]);
                double[] expected = oneHot(labelIndices[i]);
                epochLoss += lossFunction.calculateLoss(output, expected);
                if (maxValueIndex(output) == labelIndices[i]) {
                    trainCorrect++;
                }
            }

            epochLoss /= n;
            double trainAcc = (double) trainCorrect / n;

            System.out.printf("Epoch %d | Loss: %.6f | Train Acc: %.4f\n", epoch + 1, epochLoss, trainAcc);
        }
    }

    private int classifyInternal(double[] inputs) {
        double[] outputs = feedForward(inputs);
        return maxValueIndex(outputs);
    }

    public int classify(double[] inputs) {
        DataHandler.validateInputRow(inputs);
        if (inputs.length != numFeatures) {
            throw new IllegalArgumentException("Number of features doesn't match the ANN");
        }
        double[] outputs = feedForward(inputs);
        return maxValueIndex(outputs);
    }

    public int[] classifyAll(double[][] inputs) {
        validateInputs(inputs);
        int[] predictions = new int[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            predictions[i] = classifyInternal(inputs[i]);
        }
        return predictions;
    }

    public double evaluateAccuracy(double[][] inputs, int[] labelIndices) {
        validateInputs(inputs);
        DataHandler.validateLabels(labelIndices, numClasses);
        if (inputs.length != labelIndices.length) {
            throw new IllegalArgumentException("Number of Inputs doesn't match the outputs");
        }
        int correct = 0;
        for (int i = 0; i < inputs.length; i++) {
            int predicted = classifyInternal(inputs[i]);
            if (predicted == labelIndices[i]) {
                correct++;
            }
        }
        return (double) correct / inputs.length;
    }

    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        for (Layer layer : layers) {
            layer.setActivationFunction(activationFunction);
        }
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    private double[] oneHot(int label) {
        double[] y = new double[numClasses];
        y[label] = 1.0;
        return y;
    }
}