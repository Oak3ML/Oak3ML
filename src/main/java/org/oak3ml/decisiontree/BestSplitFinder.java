package org.oak3ml.decisiontree;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.impurity.ImpurityCalculationMethod;

/**
 * Best split finder class.
 * 
 * @author Ignas
 *
 */
public class BestSplitFinder {
    
    private ImpurityCalculationMethod impurityCalculationMethod;
    
    /**
     * @param impurityCalculationMethod
     */
    public BestSplitFinder(ImpurityCalculationMethod impurityCalculationMethod) {
        super();
        this.impurityCalculationMethod = impurityCalculationMethod;
    }

    /**
     * Finds best feature to split on which is the one whose split results in lowest impurity measure.
     */
    public Feature findBestSplitFeature(List<DataSample> data, List<Feature> features) {
        double currentImpurity = Double.MAX_VALUE;
        Feature bestSplitFeature = null; // rename split to feature

        for (Feature feature : features) {
            Map<String, List<DataSample>> splitData = feature.split(data);
            double calculatedSplitImpurity = calculateTotalSplitImpurity(splitData.values());
            if (calculatedSplitImpurity < currentImpurity) {
                currentImpurity = calculatedSplitImpurity;
                bestSplitFeature = feature;
            }
        }

        return bestSplitFeature;
    }
    
    /**
     * Calculate total impurity of a split by weight averaging all impurities.
     * 
     * This formula is used:
     * totalSplitImpurity = sum(singleLeafImpurities) / nbOfLeafs
     * 
     * Without weight averaging it would be simply:
     * totalSplitImpurity = sum(singleLeafImpurities) / nbOfLeafs
     * 
     * @param splitData List of split leaves data.
     * @return Total impurity
     */
    protected double calculateTotalSplitImpurity(Collection<List<DataSample>> splitData) {
        double totalNb = splitData.stream().mapToDouble(List::size).sum();
        return splitData.stream().filter(list -> !list.isEmpty())
                .mapToDouble(list -> ((double)list.size() / totalNb) * impurityCalculationMethod.calculateImpurity(list)).sum();
    }

}
