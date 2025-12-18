package Data;

public class Dataset {
    public double[][] inputs;
    public int[] labels;

    Dataset(double[][] inputs, int[] labels) {
        this.inputs = inputs;
        this.labels = labels;
    }
}