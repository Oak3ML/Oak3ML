package org.oak3ml.decisiontree.feature;

import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.data.SimpleDataSample;
import org.oak3ml.testutils.EqualsTester;

import com.google.common.collect.Lists;

public class PredicateFeatureTest {

    private static final String YES = "true";

    private static final String NO = "false";

    @Test
    public void testSplitEqualsPredicate() {
        List<DataSample> data = Lists.newArrayList();
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 3, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 4, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 5, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "red"));
        
        Feature testSplit = PredicateFeature.newFeature("color", "yellow");
        
        Map<String, List<DataSample>> split = testSplit.split(data);
        Assert.assertEquals(2, split.size());
        Assert.assertEquals(2, split.get(YES).size());
        Assert.assertEquals(6, split.get(NO).size());
        Assert.assertEquals("yellow", split.get(YES).get(0).getValue("color").get());
        Assert.assertEquals("yellow", split.get(YES).get(1).getValue("color").get());
    }

    @Test
    public void testSplitMoreThanPredicate() {
        List<DataSample> data = Lists.newArrayList();
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 3, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 4, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 5, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "red"));
        
        Feature testSplit = PredicateFeature.newFeature("label", P.moreThan(3));
        
        Map<String, List<DataSample>> split = testSplit.split(data);
        Assert.assertEquals(2, split.size());
        Assert.assertEquals(2, split.get(YES).size());
        Assert.assertEquals(6, split.get(NO).size());
        Assert.assertEquals(4, split.get(YES).get(0).getValue("label").get());
        Assert.assertEquals(5, split.get(YES).get(1).getValue("label").get());
    }

    @Test
    public void testSplitLessThanPredicate() {
        List<DataSample> data = Lists.newArrayList();
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 3, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 4, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 5, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "red"));
        
        Feature testSplit = PredicateFeature.newFeature("label", P.lessThan(2));
        
        Map<String, List<DataSample>> split = testSplit.split(data);
        Assert.assertEquals(2, split.size());
        Assert.assertEquals(3, split.get(YES).size());
        Assert.assertEquals(5, split.get(NO).size());
        Assert.assertEquals(1, split.get(YES).get(0).getValue("label").get());
        Assert.assertEquals(1, split.get(YES).get(1).getValue("label").get());
        Assert.assertEquals(1, split.get(YES).get(2).getValue("label").get());
    }

    @Test
    public void testEqualsAndHashCode() {
        EqualsTester<Feature> tester = EqualsTester.newInstance(PredicateFeature.newFeature("label", 1));
        tester.assertImplementsEqualsAndHashCode();
        tester.assertEqual(PredicateFeature.newFeature("label", 2), PredicateFeature.newFeature("label", 2));
        tester.assertEqual(PredicateFeature.newFeature("label", 2), PredicateFeature.newFeature("label", P.isEqual(2)), PredicateFeature.newFeature("label", 2));
        tester.assertNotEqual(PredicateFeature.newFeature("label", P.isEqual(2)), PredicateFeature.newFeature("label", P.moreThan(2)));
    }
    
}
