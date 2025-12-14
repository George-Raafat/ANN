package ANN;

public class NetworkLearnData {
    public LayerLearnData[] layerData;

    public NetworkLearnData(Layer[] layers) {
        layerData = new LayerLearnData[layers.length];
        for (int i = 0; i < layers.length; i++) {
            layerData[i] = new LayerLearnData(layers[i]);
        }
    }
}