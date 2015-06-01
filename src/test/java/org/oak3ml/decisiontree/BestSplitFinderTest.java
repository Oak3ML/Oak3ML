package org.oak3ml.decisiontree;

import static org.oak3ml.decisiontree.label.BooleanLabel.FALSE_LABEL;
import static org.oak3ml.decisiontree.label.BooleanLabel.TRUE_LABEL;

import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.data.SimpleDataSample;
import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.feature.PredicateFeature;
import org.oak3ml.decisiontree.impurity.EntropyCalculationMethod;
import org.oak3ml.decisiontree.impurity.GiniIndexCalculationMethod;
import org.oak3ml.decisiontree.impurity.MinorityClassCalculationMethod;
import org.oak3ml.decisiontree.impurity.SquareRootGiniIndexCalculationMethod;
import org.oak3ml.decisiontree.label.BooleanLabel;

import com.google.common.collect.Lists;

public class BestSplitFinderTest {
    
    private static final String YES = "true";

    private static final String NO = "false";

    @Test
    public void testBooleanSplit() {
        BestSplitFinder splitFinder = new BestSplitFinder(new EntropyCalculationMethod());
        String labelColumnName = "answer";
        
        String[] headers = {labelColumnName, "x1", "x2"};
        List<DataSample> dataSet = Lists.newArrayList();
        dataSet.add(SimpleDataSample.newSimpleDataSample(labelColumnName, headers, TRUE_LABEL, true, true));
        dataSet.add(SimpleDataSample.newSimpleDataSample(labelColumnName, headers, FALSE_LABEL, true, false));
        dataSet.add(SimpleDataSample.newSimpleDataSample(labelColumnName, headers, FALSE_LABEL, false, true));
        dataSet.add(SimpleDataSample.newSimpleDataSample(labelColumnName, headers, FALSE_LABEL, false, false));
        
        List<Feature> features = Lists.newArrayList();
        features.add(PredicateFeature.newFeature("x1", true));
        features.add(PredicateFeature.newFeature("x2", true));
        features.add(PredicateFeature.newFeature("x1", false));
        features.add(PredicateFeature.newFeature("x2", false));
        
        // test finding split
        Feature bestSplit = splitFinder.findBestSplitFeature(dataSet, features);
        Assert.assertEquals("x1 = true", bestSplit.toString());
        
        Map<String, List<DataSample>> split = bestSplit.split(dataSet);
        
        // test splitting data
        Assert.assertEquals(TRUE_LABEL, split.get(YES).get(0).getValue(labelColumnName).get());
        Assert.assertEquals(FALSE_LABEL, split.get(YES).get(1).getValue(labelColumnName).get());
        Assert.assertEquals(FALSE_LABEL, split.get(NO).get(0).getValue(labelColumnName).get());
        Assert.assertEquals(FALSE_LABEL, split.get(NO).get(1).getValue(labelColumnName).get());

        // next best split
        Feature newBestSplit = splitFinder.findBestSplitFeature(split.get(YES), features);
        Assert.assertEquals("x2 = true", newBestSplit.toString());

        Map<String, List<DataSample>> newSplit = newBestSplit.split(split.get(YES));
        Assert.assertEquals(TRUE_LABEL, newSplit.get(YES).get(0).getValue(labelColumnName).get());
        Assert.assertEquals(FALSE_LABEL, newSplit.get(NO).get(0).getValue(labelColumnName).get());
    }

    @Test
    public void testCalculateTotalImpurityWithEntropyMethod() {
        BestSplitFinder splitFinder = new BestSplitFinder(new EntropyCalculationMethod());
        Assert.assertEquals(0.72, splitFinder.calculateTotalSplitImpurity(getData()), 0.01);
    }

    @Test
    public void testCalculateTotalImpurityWithGiniIndexMethod() {
        BestSplitFinder splitFinder = new BestSplitFinder(new GiniIndexCalculationMethod());
        Assert.assertEquals(0.35, splitFinder.calculateTotalSplitImpurity(getData()), 0.01);
    }

    @Test
    public void testCalculateTotalImpurityWithSquareRootGiniIndexMethod() {
        BestSplitFinder splitFinder = new BestSplitFinder(new SquareRootGiniIndexCalculationMethod());
        Assert.assertEquals(0.52, splitFinder.calculateTotalSplitImpurity(getData()), 0.01);
    }

    @Test
    public void testCalculateTotalImpurityWithMinorityClassMethod() {
        BestSplitFinder splitFinder = new BestSplitFinder(new MinorityClassCalculationMethod());
        Assert.assertEquals(0.30, splitFinder.calculateTotalSplitImpurity(getData()), 0.01);
    }

    
    @SuppressWarnings("unchecked")
    private List<List<DataSample>> getData() {
        String[] header = {"x", "label"};
        List<DataSample> data1 = Lists.newArrayList(
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.TRUE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 2, BooleanLabel.TRUE_LABEL));
        
        List<DataSample> data2 = Lists.newArrayList(
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.FALSE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.FALSE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.FALSE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 2, BooleanLabel.TRUE_LABEL));
        
        List<DataSample> data3 = Lists.newArrayList(
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.TRUE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.FALSE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 1, BooleanLabel.FALSE_LABEL),
                SimpleDataSample.newSimpleDataSample("label", header, 2, BooleanLabel.TRUE_LABEL));
        
        return Lists.newArrayList(data1, data2, data3);
    }
}
