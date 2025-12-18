package Data;

public class Dataset {
    public double[][] inputs;
    public int[] labels;

    public Dataset(double[][] inputs, int[] labels) {
        this.inputs = inputs;
        this.labels = labels;
    }
}