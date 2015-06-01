package org.oak3ml.decisiontree.feature;

import static java.util.stream.Collectors.groupingBy;

import java.util.List;
import java.util.Map;

import org.oak3ml.decisiontree.data.DataSample;

/**
 * Feature interface. Each data sample either have or does not have a feature and it can be split based on that.
 * 
 * @author Ignas
 *
 */
public interface Feature {

    /**
     * Calculates and checks if data contains feature.
     * 
     * @param dataSample
     *            Data sample.
     * @return true if data has this feature and false otherwise.
     */
    boolean belongsTo(DataSample dataSample);
    
    /**
     * Column used by feature.
     */
    String getColumn(); // TODO multiple columns per feature or composite feature

    /**
     * Split data according to if it has this feature. This is binary split only good for Yes/No branches. For
     * categorical split look at {@link CategoricalFeature#split}.
     * 
     * @param data
     *            Data to by split by this feature.
     * @return Sublists of split data samples. Map key is name of branch (or edge).
     */
    default Map<String, List<DataSample>> split(List<DataSample> data) {
        return data.parallelStream().collect(groupingBy(dataSample -> String.valueOf(belongsTo(dataSample))));
    }

}
