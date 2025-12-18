package utils;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LabelEncoder implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private final Map<String, Integer> labelToIndex = new HashMap<>();
    private final List<String> indexToLabel = new ArrayList<>();

    public int encode(String label) {
        return labelToIndex.computeIfAbsent(label, l -> {
            indexToLabel.add(l);
            return indexToLabel.size() - 1;
        });
    }

    public String decode(int index) {
        return indexToLabel.get(index);
    }

    public int numClasses() {
        return indexToLabel.size();
    }
}