package ANN;

import Activation.ActivationFunction;
import Initialization.WeightInitializer;

public class Layer {
    public final int numNodesIn;
    public final int numNodesOut;

    private final double[] weights;
    private final double[] gradients;

    private final double[] weightedInputs;

    private final double[] activations;

    private ActivationFunction activationFunction;

    // Create the layer
    public Layer(int numNodesIn, int numNodesOut, ActivationFunction activationFunction,
                 WeightInitializer weightInitializer) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;
        this.activationFunction = activationFunction;
        weights = weightInitializer.init(numNodesIn + 1, numNodesOut);
        gradients = new double[(numNodesIn + 1) * numNodesOut];

        weightedInputs = new double[numNodesOut];
        activations = new double[numNodesOut];
    }

    // Calculate layer output activations
    public double[] CalculateOutputs(double[] inputs) {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double weightedInput = getWeight(numNodesIn, nodeOut); // bias
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightedInput += inputs[nodeIn] * getWeight(nodeIn, nodeOut);
            }
            weightedInputs[nodeOut] = weightedInput;
        }
        for (int outputNode = 0; outputNode < numNodesOut; outputNode++) {
            activations[outputNode] = activationFunction.activate(weightedInputs[outputNode]);
        }
        return activations;
    }

    public double[] accumulateGradients(double[] subdeltas, double[] prevInput) {
        double[] newSubDeltas = new double[numNodesIn];
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double delta = subdeltas[nodeOut] * activationFunction.derivative(weightedInputs[nodeOut]);
            int weightIndex;
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightIndex = getFlatWeightIndex(nodeIn, nodeOut);
                newSubDeltas[nodeIn] += delta * weights[weightIndex];
                gradients[weightIndex] += delta * prevInput[nodeIn];
            }
            weightIndex = getFlatWeightIndex(numNodesIn, nodeOut);
            gradients[weightIndex] += delta; // bias
        }
        return newSubDeltas;
    }

    public void applyGradients(double learningRate, int batchSize) {
        double scale = learningRate / batchSize;

        for (int i = 0; i < weights.length; i++) {
            weights[i] -= scale * gradients[i];
            gradients[i] = 0.0;
        }
    }

    public double getWeight(int nodeIn, int nodeOut) {
        int flatIndex = nodeOut * (numNodesIn + 1) + nodeIn;
        return weights[flatIndex];
    }

    public int getFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex) {
        return outputNeuronIndex * (numNodesIn + 1) + inputNeuronIndex;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public double[] getActivations() {
        return activations;
    }
}