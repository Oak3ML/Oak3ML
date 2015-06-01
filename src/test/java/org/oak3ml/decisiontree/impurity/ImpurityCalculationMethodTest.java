package org.oak3ml.decisiontree.impurity;

import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.data.SimpleDataSample;
import org.oak3ml.decisiontree.impurity.GiniIndexCalculationMethod;
import org.oak3ml.decisiontree.label.BooleanLabel;

public class ImpurityCalculationMethodTest {

    @Test
    public void testGetEmpiricalProbability50_50() {
        DataSample dataSample1 = SimpleDataSample.newSimpleDataSample("a", new String[]{"a"}, BooleanLabel.TRUE_LABEL);
        DataSample dataSample2 = SimpleDataSample.newSimpleDataSample("a", new String[]{"a"}, BooleanLabel.FALSE_LABEL);
        double p = new GiniIndexCalculationMethod().getEmpiricalProbability(Arrays.asList(dataSample1, dataSample2), BooleanLabel.TRUE_LABEL);
        Assert.assertEquals(0.5, p, 0.001);
    }
    
}
