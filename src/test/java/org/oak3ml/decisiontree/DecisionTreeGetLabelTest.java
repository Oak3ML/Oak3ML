package org.oak3ml.decisiontree;

import static org.oak3ml.decisiontree.label.BooleanLabel.FALSE_LABEL;
import static org.oak3ml.decisiontree.label.BooleanLabel.TRUE_LABEL;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.oak3ml.decisiontree.DecisionTree;
import org.oak3ml.decisiontree.data.DataSample;

import com.google.common.collect.Lists;

public class DecisionTreeGetLabelTest {
    
    @Test
    public void testGetLabelOnEmptyList() {
        DecisionTree tree = new DecisionTree.Builder().build();
        List<DataSample> data = Lists.newArrayList();
        Assert.assertNull(tree.getLabel(data));
    }
    
    @Test
    public void testGetLabelOnSingleElement() {
        DecisionTree tree = new DecisionTree.Builder().build();
        List<DataSample> data = Lists.newArrayList();
        data.add(new TestDataSample(null, TRUE_LABEL));
        Assert.assertEquals("true", tree.getLabel(data).getName());
    }

    @Test
    public void testGetLabelOnTwoDifferent() {
        DecisionTree tree = new DecisionTree.Builder().build();
        List<DataSample> data = Lists.newArrayList();
        data.add(new TestDataSample(null, TRUE_LABEL));
        data.add(new TestDataSample(null, FALSE_LABEL));
        Assert.assertNull(tree.getLabel(data));
    }

    @Test
    public void testGetLabelOn95vs5() {
        DecisionTree tree = new DecisionTree.Builder().build();
        List<DataSample> data = Lists.newArrayList();
        for (int i = 0; i < 95; i++) {
            data.add(new TestDataSample(null, TRUE_LABEL));
        }
        for (int i = 0; i < 5; i++) {
            data.add(new TestDataSample(null, FALSE_LABEL));
        }
        Assert.assertEquals("true", tree.getLabel(data).getName());
    }

    @Test
    public void testGetLabelOn94vs6() {
        DecisionTree tree = new DecisionTree.Builder().build();
        List<DataSample> data = Lists.newArrayList();
        for (int i = 0; i < 94; i++) {
            data.add(new TestDataSample(null, TRUE_LABEL));
        }
        for (int i = 0; i < 6; i++) {
            data.add(new TestDataSample(null, FALSE_LABEL));
        }
        Assert.assertNull(tree.getLabel(data));
    }

}
