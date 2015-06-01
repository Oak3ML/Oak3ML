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

public class GroupedPredicatesFeatureTest {
    
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
        
        
        Feature feature1 = PredicateFeature.newFeature("color", "black");
        Feature feature2 = PredicateFeature.newFeature("color", "yellow");
        Feature feature3 = PredicateFeature.newFeature("color", "white");
        Feature feature4 = PredicateFeature.newFeature("color", "red");
        Feature feature5 = PredicateFeature.newFeature("color", "non existant");
        
        Feature testSplit = GroupedPredicatesFeature.newFeature("color", Lists.newArrayList(feature1, feature2, feature3, feature4, feature5));
        
        Map<String, List<DataSample>> split = testSplit.split(data);
        Assert.assertEquals(4, split.size());
        
        Set<String> categories = split.values().stream().map(list -> (String)list.get(0).getValue("color").get()).collect(Collectors.toSet());
        Assert.assertEquals(4, categories.size());
        Assert.assertTrue(categories.contains("black"));
        Assert.assertTrue(categories.contains("white"));
        Assert.assertTrue(categories.contains("red"));
        Assert.assertTrue(categories.contains("yellow"));
        
    }
    
    @Test
    public void testEqualsAndHashCode() {
        
        Feature feature1 = PredicateFeature.newFeature("a", 1);
        Feature feature2 = PredicateFeature.newFeature("a", 2);
        Feature feature3 = PredicateFeature.newFeature("a", 3);
        
        EqualsTester<Feature> tester = EqualsTester.newInstance(GroupedPredicatesFeature.newFeature("a", Lists.newArrayList(feature1, feature2)));
        tester.assertImplementsEqualsAndHashCode();
        tester.assertEqual(GroupedPredicatesFeature.newFeature("a", Lists.newArrayList(feature1, feature2)), GroupedPredicatesFeature.newFeature("a", Lists.newArrayList(feature1, feature2)));
        tester.assertNotEqual(GroupedPredicatesFeature.newFeature("a", Lists.newArrayList(feature2, feature3)), GroupedPredicatesFeature.newFeature("a", Lists.newArrayList(feature1, feature2)));
    }

}
