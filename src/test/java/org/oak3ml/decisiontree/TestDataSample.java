package org.oak3ml.decisiontree;

import java.util.Optional;

import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.label.Label;

public class TestDataSample implements DataSample {
    
    private Object value;
    
    private Label label;
    
    public TestDataSample(Object value, Label label) {
        super();
        this.value = value;
        this.label = label;
    }

    @Override
    public Optional<Object> getValue(String column) {
        return Optional.of(value);
    }

    @Override
    public Label getLabel() {
        return label;
    }

}
