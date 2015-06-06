package org.oak3ml.features.discretisation;

import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static org.oak3ml.decisiontree.feature.P.betweenD;
import static org.oak3ml.decisiontree.feature.P.lessThanOrEqualD;
import static org.oak3ml.decisiontree.feature.P.moreThanD;
import static org.oak3ml.decisiontree.feature.PredicateFeature.newFeature;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.oak3ml.decisiontree.BestSplitFinder;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.impurity.EntropyCalculationMethod;
import org.oak3ml.decisiontree.impurity.ImpurityCalculationMethod;
import org.oak3ml.decisiontree.label.Label;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

/**
 * Supervised discretisation method. Using top-down approach - splitting bins recursively until stopping criterion is reached. 
 * 
 * @author Ignas
 *
 */
public class DivisiveDiscretiser implements FeatureDiscretiser {
    
    /**
     * How often to check for a bin points. For example we have min = 0, max = 15. If step fraction is 0.1 (10%) then we check every (15+0)/10=1.5.
     * 0, 1.5, 3.0, 4.5, 6.0 ... 13.5, 15 for possible bins.
     */
    private double stepFraction = 0.1;
    
    /**
     * This parameter is used in stopping criteria. If we reach data which is homogeneous 90% (default) we stop and do not split anymore.
     */
    private double homogenityPercentage = 0.9;
    
    /**
     * Calculate impurity (information gain) method. Default - entropy.
     */
    private ImpurityCalculationMethod impurityCalculationMethod = new EntropyCalculationMethod();

    /**
     * Private constructor.
     */
    private DivisiveDiscretiser(Builder builder) {
        super();
        
        if (builder.impurityCalculationMethod != null) {
            impurityCalculationMethod = builder.impurityCalculationMethod;
        }
        if (builder.stepFraction != null) {
            Preconditions.checkArgument(0 < builder.stepFraction  && builder.stepFraction < 1);
            stepFraction = builder.stepFraction;
        }
        if (builder.homogenityPercentage != null) {
            Preconditions.checkArgument(0 < builder.homogenityPercentage  && builder.homogenityPercentage < 1);
            homogenityPercentage = builder.homogenityPercentage;
        }
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public List<Feature> discretise(List<DataSample> trainingData, String column, int numberOfBins) {
        Preconditions.checkArgument(numberOfBins >= 2);
        Preconditions.checkArgument((numberOfBins & (numberOfBins - 1)) == 0, "Number of bins must be power of 2 (2, 4, 8, 16...) because algorithm "
                + "recusively splits data in half when finds best point and each split produces 2 bins");
        
        int numberOfSplitPoints = numberOfBins - 1;
        // currentNumberOfSplitPoints = 1 because one split point (minimum) will always produce two features(bins)
        List<Double> binSplitPoints = getSplitPoints(trainingData, column, numberOfSplitPoints, 1);
        List<Feature> features = new ArrayList<>();
        int binNb = 0;
        Double previousBinPoint = null;
        for (Double binPoint : binSplitPoints) {
            if (previousBinPoint == null) { // if first
                features.add(newFeature(column, lessThanOrEqualD(binPoint)));
            }
            if (binNb == binSplitPoints.size() - 1) { // if last
                if (previousBinPoint != null) {
                    features.add(newFeature(column, betweenD(previousBinPoint, binPoint)));
                }
                features.add(newFeature(column, moreThanD(binPoint)));
            }
            if (previousBinPoint != null && binNb != binSplitPoints.size() - 1) { // if not first nor last
                features.add(newFeature(column, betweenD(previousBinPoint, binPoint)));
            }
            previousBinPoint = binPoint;
            binNb++;
        }
        return features;
    }

    /**
     * Return list of best points for creating bins by using scoring function with recursive partitioning algorithm.
     * 
     * @param trainingData Training data.
     * @param column Column which values are being discretised.
     * @return Points on which to to discretise numerical data to bins.
     */
    protected List<Double> getSplitPoints(List<DataSample> trainingData, String column, int numberOfSplitPoints, int currentNumberOfSplitPoints) {
        boolean stoppingCriterionReached = trainingData.size() <= 1 || getLabel(trainingData) != null;
        if (stoppingCriterionReached) {
            return Lists.newArrayList();
        }
        MinMax minMax = getMinMax(trainingData, column);
        double step = (minMax.getMax() - minMax.getMin()) * stepFraction;
        
        // create features for each point to reuse their split functionality
        Map<Feature, Double> features = Maps.newHashMap();
        double currentPoint = minMax.getMin() + step;
        while (currentPoint < minMax.getMax()) {
            features.put(newFeature(column, lessThanOrEqualD(currentPoint)), currentPoint);
            currentPoint += step;
        }
        
        // find a best split
        Feature bestSplit = new BestSplitFinder(impurityCalculationMethod).findBestSplitFeature(trainingData, new ArrayList<Feature>(features.keySet()));
        
        if (bestSplit == null) {
            return Lists.newArrayList();
        }
        
        double bestPoint = features.get(bestSplit);
        
        // split test data into 2 - below bestPoint and above (TODO make 2 best points so any number of bins can be achieved)
        Map<Boolean, List<DataSample>> partitionedData = trainingData.stream().filter(dataSample -> dataSample.getValue(column).isPresent())
                .collect(Collectors.partitioningBy(dataSample -> (double)dataSample.getValue(column).get() > bestPoint));
        
        List<Double> result = Lists.newArrayList();
        
        // add results and invoke recursion if needed
        if (currentNumberOfSplitPoints < numberOfSplitPoints) result.addAll(getSplitPoints(partitionedData.get(false), column, numberOfSplitPoints, currentNumberOfSplitPoints + 1));
        result.add(bestPoint);
        if (currentNumberOfSplitPoints < numberOfSplitPoints) result.addAll(getSplitPoints(partitionedData.get(true), column, numberOfSplitPoints, currentNumberOfSplitPoints + 1));
        return result;
    }
    
    // TODO copy paste from decision tree for testing - remove it
    protected Label getLabel(List<DataSample> data) {
        // group by to map <Label, count>
        Map<Label, Long> labelCount = data.parallelStream().collect(groupingBy(DataSample::getLabel, counting()));
        long totalCount = data.size();
        for (Label label : labelCount.keySet()) {
            long nbOfLabels = labelCount.get(label);
            if (((double) nbOfLabels / (double) totalCount) >= homogenityPercentage) {
                return label;
            }
        }
        return null;
    }

    
    /* Builder */
    public static class Builder {

        private ImpurityCalculationMethod impurityCalculationMethod;

        private Double stepFraction;

        private Double homogenityPercentage;
        
        public Builder withImpurityCalculationMethod(ImpurityCalculationMethod impurityCalculationMethod) {
            this.impurityCalculationMethod = impurityCalculationMethod;
            return this;
        }

        public Builder withStepFraction(double stepFraction) {
            this.stepFraction = stepFraction;
            return this;
        }

        public Builder withHomogenityPercentage(double homogenityPercentage) {
            this.homogenityPercentage = homogenityPercentage;
            return this;
        }
        
        public DivisiveDiscretiser build() {
            return new DivisiveDiscretiser(this);
        }
    }
}
