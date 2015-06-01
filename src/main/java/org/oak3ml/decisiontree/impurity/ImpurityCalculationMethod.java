package org.oak3ml.decisiontree.impurity;

import java.util.List;

import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.label.Label;

/**
 * Impurity calculation method of decision tree. It is used during training while trying to find best split. For example
 * one split results in 5 positive and 5 negative labels and another in 9 positive and 1 negative. Impurity calculator
 * should return high value in the first case and low value in the second. That means that during training we will
 * prefer second split as that will provide most information. Most popular methods are entropy and gini index.
 * 
 * Annotated as functional interface to allow lambda usage.
 * 
 * @author Ignas
 *
 */
@FunctionalInterface
public interface ImpurityCalculationMethod {

    /**
     * Calculates impurity value. High impurity implies low information gain and more random labels of data which in
     * turn means that split is not very good.
     * 
     * @param splitData
     *            Data subset on which impurity is calculated.
     * 
     * @return Impurity measure. 0 means it is pure and higher means it is impure. With entropy 1.0 and with Gini index 0.5 means data is totally random.
     */
    double calculateImpurity(List<DataSample> splitData);

    
    /**
     * Calculate and return empirical probability. For formula is: empiricalProbability = # of labels = Label / # of total samples.
     * 
     * @param splitData Data on which positive label probability is calculated.
     * @return Empirical probability.
     */
    default double getEmpiricalProbability(List<DataSample> splitData, Label label) {
        // TODO Performance cache calculated counts
        return (double)splitData.parallelStream().filter(d -> d.getLabel().equals(label)).count() / splitData.size();
    }
}
