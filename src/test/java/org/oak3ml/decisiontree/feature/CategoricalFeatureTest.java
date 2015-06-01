package org.oak3ml.decisiontree.feature;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.junit.Assert;
import org.junit.Test;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.data.SimpleDataSample;
import org.oak3ml.testutils.EqualsTester;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

public class CategoricalFeatureTest {

    @Test
    public void testSplitWithAllCategories() {
        List<DataSample> data = Lists.newArrayList();
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 3, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 4, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 5, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "red"));
        
        Feature testSplit = CategoricalFeature.newFeature("color", data);
        
        Map<String, List<DataSample>> split = testSplit.split(data);
        Assert.assertEquals(4, split.size());
        
        Set<String> categories = split.values().stream().map(list -> (String)list.get(0).getValue("color").get()).collect(Collectors.toSet());
        Assert.assertEquals(4, categories.size());
        Assert.assertTrue(categories.contains("black"));
        Assert.assertTrue(categories.contains("yellow"));
        Assert.assertTrue(categories.contains("white"));
        Assert.assertTrue(categories.contains("red"));
    }

    @Test
    public void testSplitWithProvidedCategories() {
        List<DataSample> data = Lists.newArrayList();
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 3, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 4, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 5, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "red"));
        
        // feature without black color category
        Feature testSplit = CategoricalFeature.newFeature("color", Sets.newHashSet("white", "red", "yellow"));
        
        Map<String, List<DataSample>> split = testSplit.split(data);
        Assert.assertEquals(3, split.size());
        
        Set<String> categories = split.keySet();
        Assert.assertEquals(3, categories.size());
        Assert.assertTrue(categories.contains("yellow"));
        Assert.assertTrue(categories.contains("white"));
        Assert.assertTrue(categories.contains("red"));
        
    }
    
    @Test
    public void testSplitWithMissingCategories() {
        List<DataSample> data = Lists.newArrayList();
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 3, "yellow"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 1, "black"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 4, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 5, "white"));
        data.add(SimpleDataSample.newSimpleDataSample("label", new String[]{"label", "color"}, 2, "red"));
        
        // feature without black color category
        Feature testSplit = CategoricalFeature.newFeature("color", Sets.newHashSet("white", "red", "yellow", "green"));
        
        Map<String, List<DataSample>> split = testSplit.split(data);
        Assert.assertEquals(4, split.size());
        
        Set<String> categories = split.keySet();
        Assert.assertEquals(4, categories.size());
        Assert.assertTrue(categories.contains("yellow"));
        Assert.assertTrue(categories.contains("white"));
        Assert.assertTrue(categories.contains("red"));
        Assert.assertTrue(categories.contains("green"));
    }
    
    @Test
    public void testEqualsAndHashCode() {
        EqualsTester<Feature> tester = EqualsTester.newInstance(CategoricalFeature.newFeature("label", Sets.newHashSet("1", "2")));
        tester.assertImplementsEqualsAndHashCode();
        tester.assertEqual(CategoricalFeature.newFeature("label", Sets.newHashSet("1", "2")), CategoricalFeature.newFeature("label", Sets.newHashSet("1", "2")));
        tester.assertNotEqual(CategoricalFeature.newFeature("label", Sets.newHashSet("2", "3")), CategoricalFeature.newFeature("label", Sets.newHashSet("1", "2")));
    }
    
}
