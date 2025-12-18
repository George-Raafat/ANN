import ANN.NeuralNetwork;
import Data.Dataset;
import utils.ImageUtils;
import utils.LabelEncoder;
import utils.ModelIO;
import utils.TrainedModel;

import java.io.File;
import java.io.IOException;

import static utils.ImageUtils.loadDataset;

public class PredictMain {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        TrainedModel model = ModelIO.load("models/ann.bin");
        NeuralNetwork nn = model.network();
        LabelEncoder encoder = model.encoder();

        Dataset testSet = loadDataset("testing", encoder, 100);
        double accuracy = nn.evaluateAccuracy(testSet.inputs, testSet.labels);
        System.out.println("Accuracy: " + accuracy);

        File testImage = new File("testing/hand/image_50.png");
        double[] input = ImageUtils.readGrayscaleNormalized(testImage);
        int labelIndex = nn.classify(input);
        String label = encoder.decode(labelIndex);

        System.out.println("Output Label: " + label);
    }
}
