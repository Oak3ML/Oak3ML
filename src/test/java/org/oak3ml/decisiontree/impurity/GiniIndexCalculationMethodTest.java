package org.oak3ml.decisiontree.impurity;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.data.SimpleDataSample;
import org.oak3ml.decisiontree.label.BooleanLabel;

import com.google.common.collect.Lists;

public class GiniIndexCalculationMethodTest {
    
    @Test
    public void testCalculateImpurity() {
        GiniIndexCalculationMethod giniCalculationMethod = new GiniIndexCalculationMethod();
        String[] header = {"x", "label"};
        List<DataSample> data = Lists.newArrayList(
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.TRUE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 2, BooleanLabel.TRUE_LABEL));
        
        double calculatedImpurity = giniCalculationMethod.calculateImpurity(data);
        
        Assert.assertEquals(0.00, calculatedImpurity, 0.001);

        data = Lists.newArrayList(
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.FALSE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.FALSE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.FALSE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 2, BooleanLabel.TRUE_LABEL));
        
        calculatedImpurity = giniCalculationMethod.calculateImpurity(data);

        Assert.assertEquals(0.375, calculatedImpurity, 0.001);

        data = Lists.newArrayList(
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.TRUE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.FALSE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.FALSE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 2, BooleanLabel.TRUE_LABEL));
        
        calculatedImpurity = giniCalculationMethod.calculateImpurity(data);
        
        Assert.assertEquals(0.50, calculatedImpurity, 0.01);
    }
    
}
