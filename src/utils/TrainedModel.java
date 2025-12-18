package utils;

import ANN.NeuralNetwork;

import java.io.Serial;
import java.io.Serializable;

public record TrainedModel(NeuralNetwork network, LabelEncoder encoder) implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

}