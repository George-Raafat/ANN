package ANN;

import Activation.ActivationFunction;
import Activation.Sigmoid;
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
    private final int numClasses;
    private transient LossFunction lossFunction = new MeanSquaredError();
    private double learningRate = 0.1;
    private int batchSize = 100;
    private int epochs = 10;

    // Create the neural network
    public NeuralNetwork(int[] layerSizes, WeightInitializer weightInitializer, ActivationFunction activationFunction) {
        layers = new Layer[layerSizes.length - 1];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], activationFunction, weightInitializer);
        }
        numClasses = layerSizes[layerSizes.length - 1];
    }

    public NeuralNetwork(int[] layerSizes, WeightInitializer weightInitializer) {
        this(layerSizes, weightInitializer, new Sigmoid());
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

    public int classify(double[] inputs) {
        double[] outputs = feedForward(inputs);
        return maxValueIndex(outputs);
    }

    public double[] feedForward(double[] inputs) {
        for (Layer layer : layers) {
            inputs = layer.calculateOutputs(inputs);
        }
        return inputs;
    }

    public void backwardPropagation(double[] trainingData, double[] expected) {
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
        int n = trainingData.length;

        List<Integer> allIndices = new ArrayList<>(n);
        for (int i = 0; i < n; i++) allIndices.add(i);
        Collections.shuffle(allIndices);

        int trainSize = (int) (0.8 * n);
        List<Integer> trainIdx = allIndices.subList(0, trainSize);
        List<Integer> testIdx  = allIndices.subList(trainSize, n);

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(trainIdx);

            for (int i = 0; i < trainSize; i += batchSize) {
                int end = Math.min(i + batchSize, trainSize);

                for (int j = i; j < end; j++) {
                    int idx = trainIdx.get(j);
                    double[] expected = oneHot(labelIndices[idx]);
                    backwardPropagation(trainingData[idx], expected);
                }

                for (Layer layer : layers) {
                    layer.applyGradients(learningRate, end - i);
                }
            }

            double epochLoss = 0.0;
            int trainCorrect = 0;
            int testCorrect = 0;

            for (int idx : trainIdx){
                double[] output = feedForward(trainingData[idx]);
                double[] expected = oneHot(labelIndices[idx]);
                epochLoss += lossFunction.calculateLoss(output, expected);
                if (maxValueIndex(output) == labelIndices[idx]) {
                    trainCorrect++;
                }
            }

            epochLoss /= trainSize;
            double trainAcc = (double) trainCorrect / trainSize;

            for (int idx : testIdx) {
                int predicted = classify(trainingData[idx]);
                if (predicted == labelIndices[idx]) {
                    testCorrect++;
                }
            }

            double testAcc = (double) testCorrect / testIdx.size();

            System.out.printf(
                    "Epoch %d | Loss: %.6f | Train Acc: %.4f | Test Acc: %.4f%n",
                    epoch + 1,
                    epochLoss,
                    trainAcc,
                    testAcc
            );
        }
    }

    public int[] classifyAll(double[][] inputs) {
        int[] predictions = new int[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            predictions[i] = classify(inputs[i]);
        }
        return predictions;
    }

    public double evaluateAccuracy(double[][] inputs, int[] labelIndices) {
        int correct = 0;
        for (int i = 0; i < inputs.length; i++) {
            int predicted = classify(inputs[i]);
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