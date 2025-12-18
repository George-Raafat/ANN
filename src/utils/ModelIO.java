package utils;

import java.io.*;

public class ModelIO {

    private ModelIO() {
    }

    public static void save(String path, TrainedModel model) throws IOException {
        try (ObjectOutputStream out =
                     new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(model);
        }
    }

    public static TrainedModel load(String path)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream in =
                     new ObjectInputStream(new FileInputStream(path))) {
            return (TrainedModel) in.readObject();
        }
    }
}