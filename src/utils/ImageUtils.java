package utils;

import Data.Dataset;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class ImageUtils {

    private ImageUtils() {
    } // Utility class

    /**
     * Reads an image, converts to grayscale, normalizes pixels to [0,1],
     * and flattens to a row-major double array.
     */
    public static double[] readGrayscaleNormalized(File file) throws IOException {
        BufferedImage color = ImageIO.read(file);
        if (color == null) {
            throw new IOException("Unsupported image format");
        }

        // Convert to grayscale
        BufferedImage gray = new BufferedImage(
                color.getWidth(),
                color.getHeight(),
                BufferedImage.TYPE_BYTE_GRAY
        );

        gray.getGraphics().drawImage(color, 0, 0, null);

        Raster raster = gray.getRaster();

        int w = gray.getWidth();
        int h = gray.getHeight();
        double[] result = new double[w * h];

        int idx = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int value = raster.getSample(x, y, 0); // 0–255
                result[idx++] = value / 255.0;
            }
        }

        return result;
    }

    public static Dataset loadDataset(String rootPath, LabelEncoder encoder, int maxLen) throws IOException {
        List<double[]> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        File root = new File(rootPath);
        if (!root.isDirectory()) {
            throw new IllegalArgumentException("Invalid dataset path: " + rootPath);
        }

        for (File classDir : Objects.requireNonNull(root.listFiles())) {
            if (!classDir.isDirectory()) continue;

            String labelName = classDir.getName();
            int labelIndex = encoder.encode(labelName);

            File[] images = Objects.requireNonNull(classDir.listFiles());
            int n = Math.min(images.length, maxLen);
            for (int i = 0; i < n; i++) {
                if (!images[i].isFile()) continue;

                double[] input = ImageUtils.readGrayscaleNormalized(images[i]);

                inputs.add(input);
                labels.add(labelIndex);
            }
        }

        return new Dataset(
                inputs.toArray(new double[0][]),
                labels.stream().mapToInt(Integer::intValue).toArray()
        );
    }
}