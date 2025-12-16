package ANN;

import Activation.ActivationFunction;
import Activation.Sigmoid;
import Initialization.WeightInitializer;
import Loss.LossFunction;
import Loss.MeanSquaredError;

public class NeuralNetwork {
    private final Layer[] layers;
    private LossFunction lossFunction = new MeanSquaredError();
    private double learningRate = 0.1;
    private int batchSize = 10;
    private int epochs = 10;

    // Create the neural network
    public NeuralNetwork(int[] layerSizes, WeightInitializer weightInitializer, ActivationFunction activationFunction) {
        layers = new Layer[layerSizes.length - 1];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], activationFunction, weightInitializer);
        }
    }

    public NeuralNetwork(int[] layerSizes, WeightInitializer weightInitializer) {
        this(layerSizes, weightInitializer, new Sigmoid());
    }

    private int MaxValueIndex(double[] values) {
        double maxValue = Double.MIN_VALUE;
        int index = 0;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > maxValue) {
                maxValue = values[i];
                index = i;
            }
        }
        return index;
    }

    public int Classify(double[] inputs) {
        double[] outputs = feedForward(inputs);
        return MaxValueIndex(outputs);
    }

    public double[] feedForward(double[] inputs) {
        for (Layer layer : layers) {
            inputs = layer.CalculateOutputs(inputs);
        }
        return inputs;
    }

    public double backwordPropagation(double[] trainingData, double[] expected) {
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
        return lossFunction.calculateLoss(output, expected);
    }

    public void train(double[][] trainingData, double[][] expected) {
        int n = trainingData.length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0.0;

            for (int i = 0; i < n; i += batchSize) {
                int end = Math.min(i + batchSize, n);

                for (int j = i; j < end; j++) {
                    epochLoss += backwordPropagation(trainingData[j], expected[j]);
                }

                for (Layer layer : layers) {
                    layer.applyGradients(learningRate, end - i);
                }
            }

            epochLoss /= n;
            System.out.println("Epoch " + (epoch + 1) + " | Loss: " + epochLoss);
        }
    }

    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        for (Layer layer : layers) {
            layer.setActivationFunction(activationFunction);
        }
    }

    public void setBatchSize(int batchSize){
        this.batchSize = batchSize;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}