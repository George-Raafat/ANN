package ANN;

import Activation.ActivationFunction;
import Initialization.WeightInitializer;

import java.io.Serial;
import java.io.Serializable;

public class Layer implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;
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
    public double[] calculateOutputs(double[] inputs) {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            int base = nodeOut * (numNodesIn + 1);
            double weightedInput = weights[base + numNodesIn]; // bias
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightedInput += inputs[nodeIn] * weights[base + nodeIn];
            }
            weightedInputs[nodeOut] = weightedInput;
            activations[nodeOut] = activationFunction.activate(weightedInput);
        }
        return activations;
    }

    public double[] accumulateGradients(double[] subdeltas, double[] prevInput) {
        double[] newSubDeltas = new double[numNodesIn];
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            int base = nodeOut * (numNodesIn + 1);
            double delta = subdeltas[nodeOut] * activationFunction.derivative(weightedInputs[nodeOut]);
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                newSubDeltas[nodeIn] += delta * weights[base + nodeIn];
                gradients[base + nodeIn] += delta * prevInput[nodeIn];
            }
            gradients[base + numNodesIn] += delta; // bias
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

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public double[] getActivations() {
        return activations;
    }
}