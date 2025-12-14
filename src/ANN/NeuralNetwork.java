package ANN;

import Activation.ActivationFunction;
import Activation.Sigmoid;
import Initialization.WeightInitializer;
import Loss.LossFunction;
import Loss.MeanSquaredError;

public class NeuralNetwork {
    private final Layer[] layers;
    NetworkLearnData[] batchLearnData;
    private LossFunction lossFunction = new MeanSquaredError();

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

    // Run the inputs through the network to predict which class they belong to.
    // Also returns the activations from the output layer.
    public int Classify(double[] inputs) {
        double[] outputs = feedForward(inputs);
        return MaxValueIndex(outputs);
    }

    // Run the inputs through the network to calculate the outputs
    public double[] feedForward(double[] inputs) {
        for (Layer layer : layers) {
            inputs = layer.CalculateOutputs(inputs);
        }
        return inputs;
    }


    public void Learn(DataPoint[] trainingData, double learnRate) {

        if (batchLearnData == null || batchLearnData.length != trainingData.length) {
            batchLearnData = new NetworkLearnData[trainingData.length];
            for (int i = 0; i < batchLearnData.length; i++) {
                batchLearnData[i] = new NetworkLearnData(layers);
            }
        }

        for (int i = 0; i < trainingData.length; i++) {
            UpdateGradients(trainingData[i], batchLearnData[i]);
        }
        // Update weights and biases based on the calculated gradients
        for (Layer layer : layers) {
            layer.ApplyGradients(learnRate / trainingData.length);
        }
    }


    void UpdateGradients(DataPoint data, NetworkLearnData learnData) {
        // Feed data through the network to calculate outputs.
        // Save all inputs/weightedInputs/activations along the way to use for backpropagation.
        double[] inputsToNextLayer = data.inputs;

        for (int i = 0; i < layers.length; i++) {
            inputsToNextLayer = layers[i].CalculateOutputs(inputsToNextLayer, learnData.layerData[i]);
        }

        // -- Backpropagation --
        int outputLayerIndex = layers.length - 1;
        Layer outputLayer = layers[outputLayerIndex];
        LayerLearnData outputLearnData = learnData.layerData[outputLayerIndex];

        // Update output layer gradients
        outputLayer.CalculateOutputLayerNodeValues(outputLearnData, data.expectedOutputs, lossFunction);
        outputLayer.UpdateGradients(outputLearnData);

        // Update all hidden layer gradients
        for (int i = outputLayerIndex - 1; i >= 0; i--) {
            LayerLearnData layerLearnData = learnData.layerData[i];
            Layer hiddenLayer = layers[i];

            hiddenLayer.CalculateHiddenLayerNodeValues(layerLearnData, layers[i + 1],
                    learnData.layerData[i + 1].nodeValues);
            hiddenLayer.UpdateGradients(layerLearnData);
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
}