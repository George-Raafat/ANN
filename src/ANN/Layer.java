package ANN;

import Activation.ActivationFunction;
import Initialization.WeightInitializer;
import Loss.LossFunction;

public class Layer {
    public final int numNodesIn;
    public final int numNodesOut;

    private final double[] weights;
    private final double[] biases;

    // Cost gradient with respect to weights and with respect to biases
    private final double[] costGradientW;
    private final double[] costGradientB;

    private ActivationFunction activationFunction;

    // Create the layer
    public Layer(int numNodesIn, int numNodesOut, ActivationFunction activationFunction,
                 WeightInitializer weightInitializer) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;
        this.activationFunction = activationFunction;
        weights = weightInitializer.init(numNodesIn, numNodesOut);

        costGradientW = new double[weights.length];
        biases = new double[numNodesOut];
        costGradientB = new double[biases.length];
    }

    // Calculate layer output activations
    public double[] CalculateOutputs(double[] inputs) {
        double[] output = new double[numNodesOut];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double weightedInput = biases[nodeOut];

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
            }
            output[nodeOut] = weightedInput;
        }

        for (int outputNode = 0; outputNode < numNodesOut; outputNode++) {
            output[outputNode] = activationFunction.activate(output[outputNode]);
        }

        return output;
    }

    // Calculate layer output activations and store inputs/weightedInputs/activations in the given learnData object
    public double[] CalculateOutputs(double[] inputs, LayerLearnData learnData) {
        learnData.inputs = inputs;

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double weightedInput = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightedInput += inputs[nodeIn] * GetWeight(nodeIn, nodeOut);
            }
            learnData.weightedInputs[nodeOut] = weightedInput;
        }

        // Apply activation function
        for (int i = 0; i < learnData.activations.length; i++) {
            learnData.activations[i] = activationFunction.activate(learnData.weightedInputs[i]);
        }

        return learnData.activations;
    }

    // Update weights and biases based on previously calculated gradients.
    // Also resets the gradients to zero.
    public void ApplyGradients(double learnRate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= costGradientW[i] * learnRate;
            costGradientW[i] = 0;
        }

        for (int i = 0; i < biases.length; i++) {
            biases[i] -= costGradientB[i] * learnRate;
            costGradientB[i] = 0;
        }
    }

    // Calculate the "node values" for the output layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    public void CalculateOutputLayerNodeValues(LayerLearnData layerLearnData, double[] expectedOutputs,
                                               LossFunction lossFunction) {
        for (int i = 0; i < layerLearnData.nodeValues.length; i++) {
            // Evaluate partial derivatives for current node: cost/activation & activation/weightedInput
            double costDerivative = lossFunction.derivative(layerLearnData.activations[i], expectedOutputs[i]);
            double activationDerivative = activationFunction.derivative(layerLearnData.weightedInputs[i]);
            layerLearnData.nodeValues[i] = costDerivative * activationDerivative;
        }
    }

    // Calculate the "node values" for a hidden layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    public void CalculateHiddenLayerNodeValues(LayerLearnData layerLearnData, Layer oldLayer, double[] oldNodeValues) {
        for (int newNodeIndex = 0; newNodeIndex < numNodesOut; newNodeIndex++) {
            double newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
                // Partial derivative of the weighted input with respect to the input
                double weightedInputDerivative = oldLayer.GetWeight(newNodeIndex, oldNodeIndex);
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            newNodeValue *= activationFunction.derivative(layerLearnData.weightedInputs[newNodeIndex]);
            layerLearnData.nodeValues[newNodeIndex] = newNodeValue;
        }

    }

    public void UpdateGradients(LayerLearnData layerLearnData) {
        // Update cost gradient with respect to weights (lock for multithreading)

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double nodeValue = layerLearnData.nodeValues[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                // Evaluate the partial derivative: cost / weight of current connection
                double derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue;
                // The costGradientW array stores these partial derivatives for each weight.
                // Note: the derivative is being added to the array here because ultimately we want
                // to calculate the average gradient across all the data in the training batch
                costGradientW[GetFlatWeightIndex(nodeIn, nodeOut)] += derivativeCostWrtWeight;
            }
        }


        // Update cost gradient with respect to biases (lock for multithreading)

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            // Evaluate partial derivative: cost / bias
            double derivativeCostWrtBias = 1 * layerLearnData.nodeValues[nodeOut];
            costGradientB[nodeOut] += derivativeCostWrtBias;
        }
    }

    public double GetWeight(int nodeIn, int nodeOut) {
        int flatIndex = nodeOut * numNodesIn + nodeIn;
        return weights[flatIndex];
    }

    public int GetFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex) {
        return outputNeuronIndex * numNodesIn + inputNeuronIndex;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }
}