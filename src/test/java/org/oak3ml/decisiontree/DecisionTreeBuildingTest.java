package org.oak3ml.decisiontree;

import java.lang.reflect.Field;

import org.junit.Assert;
import org.junit.Test;
import org.oak3ml.decisiontree.impurity.MinorityClassCalculationMethod;

public class DecisionTreeBuildingTest {
    
    @Test
    public void testBuildTree() throws IllegalArgumentException, IllegalAccessException, NoSuchFieldException, SecurityException {
        DecisionTree tree = new DecisionTree.Builder()
            .findBestSplitOnClusterIfMoreThan(101l).useParalelStreamIfMoreThan(101l)
            .withHomogenityPercentage(90.0).withImpurityCalculationMethod(new MinorityClassCalculationMethod())
            .withMaxDepth(101l).build();
        
        Field field = DecisionTree.class.getDeclaredField("settings");
        field.setAccessible(true);
        
        DecisionTreeSettings settings = (DecisionTreeSettings)field.get(tree);
        
        Assert.assertEquals(101l, settings.getFindBestSplitOnClusterIfMoreThan());
        Assert.assertEquals(101l, settings.getUseParalelStreamIfMoreThan());
        Assert.assertEquals(101l, settings.getMaxDepth());
        Assert.assertEquals(90.0, settings.getHomogenityPercentage(), 0.1);
        Assert.assertTrue(settings.getImpurityCalculationMethod() instanceof MinorityClassCalculationMethod);;
    }

}
