package org.oak3ml.decisiontree.impurity;

import java.util.List;
import java.util.stream.Collectors;

import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.label.Label;

/**
 * Entropy calculator. -p log2 p - (1 - p)log2(1 - p) - this is the expected information, in bits, conveyed by somebody
 * telling you the class of a randomly drawn example; the purer the set of examples, the more predictable this message
 * becomes and the smaller the expected information.
 * 
 * @author Ignas
 *
 */
public class MinorityClassCalculationMethod implements ImpurityCalculationMethod {

    /**
     * This works only on 2 classes data.
     * 
     * {@inheritDoc}
     */
    @Override
    public double calculateImpurity(List<DataSample> splitData) {
        List<Label> labels = splitData.parallelStream().map(data -> data.getLabel()).distinct().collect(Collectors.toList());
        if (labels.size() > 1) {
            double p = getEmpiricalProbability(splitData, labels.get(0)); // TODO fix to multiple labels
            return Math.min(p, 1.0 - p);
        } else if (labels.size() == 1) {
            return 0.0; // if only one label data is pure
        } else {
            throw new IllegalStateException("Split sublist is empty. This should never happen. Probably a bug.");
        }
    }

}
