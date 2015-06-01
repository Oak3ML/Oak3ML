package org.oak3ml.decisiontree.classifyerrorhandler;

import org.oak3ml.decisiontree.Node;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.feature.CategoricalFeature;
import org.oak3ml.decisiontree.feature.GroupedPredicatesFeature;

/**
 * Interface which is used to describe functionality what decision tree must do during classification of data sample 
 * when no branch is found. This can happen with {@link CategoricalFeature}s or especially {@link GroupedPredicatesFeature} for example 
 * when not all branches were created during training because lack of examples. It can especially happen when those features are lower in the
 * tree and they did not have enough training data left. Implementations of this interface tells what to do in such case during classification.
 * 
 * @author Ignas
 *
 */
public interface BranchNotFoundHandler {
    
    /**
     * Implement this method to handle error when no branch is found during classification.
     */
    void handle(DataSample dataSample, Node node);

}
