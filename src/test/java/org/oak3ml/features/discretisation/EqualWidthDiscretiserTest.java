package org.oak3ml.features.discretisation;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.data.SimpleDataSample;
import org.oak3ml.decisiontree.feature.Feature;

import com.google.common.collect.Lists;

public class EqualWidthDiscretiserTest {

    @Test
    public void testDiscretise() {
        EqualWidthDiscretiser discretiser = new EqualWidthDiscretiser();
        List<DataSample> data = Lists.newArrayList();
        String[] header = {"a", "b"};
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 3.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 4.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 6.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 7.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 10.0));
        List<Feature> features = discretiser.discretise(data, "b", 3);
        
        Assert.assertEquals(3, features.size());
        Assert.assertEquals("b <= 4.666666666666666", features.get(0).toString());
        Assert.assertEquals("b between 4.666666666666666 and 7.333333333333332", features.get(1).toString());
        Assert.assertEquals("b > 7.333333333333334", features.get(2).toString());
    }

    @Test
    public void testThresholding() {
        EqualWidthDiscretiser discretiser = new EqualWidthDiscretiser();
        List<DataSample> data = Lists.newArrayList();
        String[] header = {"a", "b"};
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 3.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 2.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 4.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 6.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 7.0));
        data.add(SimpleDataSample.newSimpleDataSample("a", header, "a", 10.0));
        List<Feature> features = discretiser.discretise(data, "b", 2);
        
        Assert.assertEquals(1, features.size());
        Assert.assertEquals("b <= 6.0", features.get(0).toString());
    }
}
