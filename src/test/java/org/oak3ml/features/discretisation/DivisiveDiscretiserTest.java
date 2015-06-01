package org.oak3ml.features.discretisation;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.data.SimpleDataSample;
import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.label.IntLabel;

import com.google.common.collect.Lists;

public class DivisiveDiscretiserTest {
    
    @Test
    public void testGetSplitPoints() {
        DivisiveDiscretiser discretiser = new DivisiveDiscretiser.Builder().build();
        List<DataSample> data = Lists.newArrayList();
        String[] header = { "a", "b" };

        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 3.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 4.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 6.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 7.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 10.0));

        List<Double> points = discretiser.getSplitPoints(data, "b", 3, 0);

        Assert.assertEquals(3, points.size());
        Assert.assertEquals(2.6, points.get(0), 0.1);
        Assert.assertEquals(3.9, points.get(1), 0.1);
        Assert.assertEquals(6.7, points.get(2), 0.1);
    }

    @Test
    public void testDiscretise() {
        DivisiveDiscretiser discretiser = new DivisiveDiscretiser.Builder().build();
        List<DataSample> data = Lists.newArrayList();
        String[] header = { "a", "b" };

        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 3.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 4.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 6.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 7.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 10.0));

        List<Feature> features = discretiser.discretise(data, "b", 4);

        Assert.assertEquals(4, features.size());
        Assert.assertEquals("b <= 2.6000000000000005", features.get(0).toString());
        Assert.assertEquals("b between 2.6000000000000005 and 3.9999999999999996", features.get(1).toString());
        Assert.assertEquals("b between 3.9999999999999996 and 6.799999999999999", features.get(2).toString());
        Assert.assertEquals("b > 6.799999999999999", features.get(3).toString());
    }

    @Test
    public void testDiscretise2Bins() {
        DivisiveDiscretiser discretiser = new DivisiveDiscretiser.Builder().build();
        List<DataSample> data = Lists.newArrayList();
        String[] header = {"a", "b"};
        
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 3.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 4.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(1), 6.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 7.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, IntLabel.newLabel(2), 10.0));
        
        List<Feature> features = discretiser.discretise(data, "b", 2);
        
        Assert.assertEquals(2, features.size());
        Assert.assertEquals("b <= 6.799999999999999", features.get(0).toString());
        Assert.assertEquals("b > 6.799999999999999", features.get(1).toString());
    }

}
