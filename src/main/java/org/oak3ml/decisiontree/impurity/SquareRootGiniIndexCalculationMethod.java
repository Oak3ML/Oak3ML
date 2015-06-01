package org.oak3ml.decisiontree.impurity;

import static java.lang.Math.sqrt;

import java.util.List;
import java.util.stream.Collectors;

import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.label.Label;

/**
 * Gini index impurity calculation. Formula 2p(1 - p) - this is the expected error if we label examples in the leaf
 * randomly: positive with probability p and negative with probability 1 - p. The probability of a false positive is
 * then p(1 - p) and the probability of a false negative (1 - p)p.
 * 
 * @author Ignas
 *
 */
public class SquareRootGiniIndexCalculationMethod implements ImpurityCalculationMethod {

    /**
     * {@inheritDoc}
     */
    @Override
    public double calculateImpurity(List<DataSample> splitData) {
        List<Label> labels = splitData.parallelStream().map(data -> data.getLabel()).distinct().collect(Collectors.toList()); // TODO possible performance optimization
        if (labels.size() > 1) {
            double multiLabelImpurity = 0.0;
            for (int i = 0; i < labels.size(); i++) {
                double p = getEmpiricalProbability(splitData, labels.get(i));
                multiLabelImpurity += p * (1 - p);
            }
            return sqrt(multiLabelImpurity);
        } else if (labels.size() == 1) {
            return 0.0; // if only one label data is pure
        } else {
            throw new IllegalStateException("Split sublist is empty. This should never happen. Probably a bug.");
        }
    }

}
