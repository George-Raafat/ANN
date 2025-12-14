package ANN;

public class LayerLearnData {
    public double[] inputs;
    public double[] weightedInputs;
    public double[] activations;
    public double[] nodeValues;

    public LayerLearnData(Layer layer) {
        weightedInputs = new double[layer.numNodesOut];
        activations = new double[layer.numNodesOut];
        nodeValues = new double[layer.numNodesOut];
    }
}