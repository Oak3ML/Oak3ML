package org.oak3ml.decisiontree.data;

import java.util.Optional;

import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.label.Label;

/**
 * Labeled training data sample.
 * 
 * @author Ignas
 *
 */
public interface DataSample {
    
    /**
     * Get sample data value from specified column.
     * 
     * @return Data value.
     */
    Optional<Object> getValue(String column);
    
    /**
     * Assigned label of training data.
     * 
     * @return Label.
     */
    Label getLabel();
    
    /**
     * Syntactic sugar to check if data has feature.
     * 
     * @param feature Feature.
     * 
     * @return True if data has feature and false otherwise.
     */
    default boolean has(Feature feature) {
        return feature.belongsTo(this);
    }

}
