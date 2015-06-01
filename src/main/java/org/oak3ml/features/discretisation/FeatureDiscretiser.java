package org.oak3ml.features.discretisation;

import java.util.List;

import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.feature.Feature;

import com.google.common.base.Preconditions;

/**
 * Discretisation of numerical (quantitive) features into number of bins. For example we have person's age as a feature
 * so we can discretise it to 0-20, 20-40, 40-60 and more than 60. How it chooses ranges depends on concrete
 * implementations.
 * 
 * @author Ignas
 *
 */
public interface FeatureDiscretiser {

    /**
     * Using training data return number of predicate features where each represents a numerical bin. If numberOf bins =
     * 2 then only one feature is returned (for example age > 35). This could also be called thresholding.
     * 
     * @param trainingData
     *            Training data used for model.
     * @param column
     *            Column which contains numerical values that needs to be discretised.
     * @param numberOfBins
     *            Number of bins to create. If it is equal 2 then it can be called thresholding.
     * @return List of features where each one represents a single bin. For example feature for age between 1 and 10.
     * 
     * @see EqualWidthDiscretiser
     * @see EqualFrequencyDiscretiser
     * @see DivisiveDiscretiser
     * @see AgglomerativeDiscretiser
     */
    List<Feature> discretise(List<DataSample> trainingData, String column, int numberOfBins);

    /**
     * Return minimum and maximum values found in dataset for provided column. Data in that column must be numerical.
     * 
     * @param data Training data.
     * @param column Column.
     * @return Min and max values found in dataset.
     */
    default  MinMax getMinMax(List<DataSample> data, String column) {
        Preconditions.checkArgument(data.get(0).getValue(column).get() instanceof Double);
        Double min = data.stream().filter(d -> d.getValue(column).isPresent()).map(sample -> (Double)sample.getValue(column).get()).min(Double::compare).get();
        Double max = data.stream().filter(d -> d.getValue(column).isPresent()).map(sample -> (Double)sample.getValue(column).get()).max(Double::compare).get();
        return new MinMax(min, max);
    }
    
    /**
     * @author Ignas
     *
     */
    public static class MinMax {
        private double min;
        private double max;
        public MinMax(double min, double max) {
            super();
            this.min = min;
            this.max = max;
        }
        public double getMin() {
            return min;
        }
        public double getMax() {
            return max;
        }
    }
}
