package org.oak3ml.decisiontree.classifyerrorhandler;

import org.oak3ml.decisiontree.Node;
import org.oak3ml.decisiontree.data.DataSample;

/**
 * Simply throw an exception if decision tree cannot find a branch and classify data sample.
 * 
 * @author Ignas
 *
 */
public class ThrowExceptionHandler implements BranchNotFoundHandler {

    /**
     * {@inheritDoc}
     */
    @Override
    public void handle(DataSample dataSample, Node node) {
        throw new IllegalStateException("Could not classify! No branch in the tree was found for this data sample: " 
                + dataSample.toString() + " at this node: " + node.getName());
    }

}
