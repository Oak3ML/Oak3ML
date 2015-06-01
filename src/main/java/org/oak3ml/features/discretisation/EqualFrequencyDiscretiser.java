package org.oak3ml.features.discretisation;

import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static org.oak3ml.decisiontree.feature.P.betweenD;
import static org.oak3ml.decisiontree.feature.P.lessThanOrEqualD;
import static org.oak3ml.decisiontree.feature.P.moreThanD;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.feature.PredicateFeature;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

/**
 * Tries to find bins that have approximately equal number of datasamples.
 * 
 * @author Ignas
 *
 */
public class EqualFrequencyDiscretiser implements FeatureDiscretiser {

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Feature> discretise(List<DataSample> trainingData, String column, int numberOfBins) {
        
        Preconditions.checkArgument(numberOfBins >= 2);
        Preconditions.checkArgument(trainingData.size() > 0);
//        Preconditions.checkArgument(trainingData.get(0).getValue(column).get() instanceof Double);
        
        // count each element
        Map<Double, Long> counted = trainingData.stream().filter(d -> d.getValue(column).isPresent()).collect(groupingBy(d -> (Double)d.getValue(column).get(), TreeMap::new, counting()));

        long totalElements = counted.values().stream().mapToLong(Long::longValue).sum();
        long aproxElementsPerBin = totalElements / numberOfBins;
        
        List<Feature> features = Lists.newArrayList();
        
        int binNb = 0;
        int elementsInBin = 0;
        Double lastEntry = null;
        for (Entry<Double, Long> entry : counted.entrySet()) {
            elementsInBin += entry.getValue();
            
            // if we reached number of required elements per bin lets create a feature for that range.
            if (elementsInBin >= aproxElementsPerBin) {
                if (binNb == 0) { // if first bin use lessThan predicate
                    features.add(PredicateFeature.newFeature(column, lessThanOrEqualD(entry.getKey())));
                    lastEntry = entry.getKey();
                    elementsInBin = 0;
                } else {
                    features.add(PredicateFeature.newFeature(column, betweenD(lastEntry, entry.getKey())));
                    elementsInBin = 0;
                }
                if (binNb == numberOfBins - 2) { // if we just created second last bin lets create last one too with moreThan predicate
                    if (binNb > 0) { // if its not only 2 bin case (then we need only one feature that splits data in 2 bins)
                        features.add(PredicateFeature.newFeature(column, moreThanD(entry.getKey())));
                        elementsInBin = 0;
                    }
                    break;
                }
                binNb++;
            }
        }
        
        return features;
    }

}
