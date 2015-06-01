package org.oak3ml.features.discretisation;

import static org.oak3ml.decisiontree.feature.P.betweenD;
import static org.oak3ml.decisiontree.feature.P.lessThanOrEqualD;
import static org.oak3ml.decisiontree.feature.P.moreThanD;

import java.util.List;

import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.feature.PredicateFeature;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

/**
 * Splits to number of bins with equal width.
 * 
 * @author Ignas
 *
 */
public class EqualWidthDiscretiser implements FeatureDiscretiser {

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Feature> discretise(List<DataSample> trainingData, String column, int numberOfBins) {
        
        Preconditions.checkArgument(numberOfBins >= 2);
        
        MinMax minmax = getMinMax(trainingData, column);
        double max = minmax.getMax();
        double min = minmax.getMin();
        double step = (max - min) / numberOfBins;
        List<Feature> features = Lists.newArrayList();
        
        double currentPosition = min;
        int binNb = 0;
        
        if (numberOfBins == 2) { // for 2 bins only one feature is require which splits data into 2 pieces
            features.add(PredicateFeature.newFeature(column, lessThanOrEqualD(min + step)));
        } else {
            while (currentPosition <= max - step) {
                if (binNb == 0) { // if first bin use lessThan predicate
                    features.add(PredicateFeature.newFeature(column, lessThanOrEqualD(min + step)));
                } else if (binNb == numberOfBins - 1) { // if last bin use moreThan predicate
                    features.add(PredicateFeature.newFeature(column, moreThanD(max - step)));
                } else {
                    features.add(PredicateFeature.newFeature(column, betweenD(currentPosition, currentPosition + step)));
                }
                currentPosition += step;
                binNb++;
            }
        }
        return features;
    }
    
}
