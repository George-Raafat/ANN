import ANN.NeuralNetwork;
import Data.DataHandler;
import Data.Dataset;
import utils.LabelEncoder;
import utils.ModelIO;
import utils.TrainedModel;

import java.io.IOException;

import static utils.ImageUtils.loadDataset;

public class TrainMain {

    public static void main(String[] args) throws IOException {

        final LabelEncoder encoder = new LabelEncoder();
        final int IMAGE_SIZE = 28;
        final int NUM_CLASSES = 10;

        // Create ANN
        NeuralNetwork nn = new NeuralNetwork(new int[]{IMAGE_SIZE * IMAGE_SIZE, 50, 25, NUM_CLASSES});

        // Load training data
        Dataset dataset = loadDataset("images", encoder, 1000);

        System.out.println("Loaded " + dataset.inputs.length + " training samples");

        Dataset[] splitDataset = DataHandler.splitData(dataset, 0.8);

        System.out.println("Data split into:");
        System.out.printf("%d training data\n", splitDataset[0].inputs.length);
        System.out.printf("%d testing data\n", splitDataset[1].inputs.length);

        nn.setBatchSize(50);
        nn.setEpochs(50);
        nn.setLearningRate(0.5);

        // Train
        nn.train(splitDataset[0].inputs, splitDataset[0].labels);

        double accuracy = nn.evaluateAccuracy(splitDataset[1].inputs, splitDataset[1].labels);

        System.out.println("Test Data Accuracy: " + accuracy);

        TrainedModel model = new TrainedModel(nn, encoder);
        ModelIO.save("models/ann.bin", model);
    }
}
