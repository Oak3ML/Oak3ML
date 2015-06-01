package org.oak3ml.decisiontree.feature;

import static java.util.stream.Collectors.groupingBy;

import java.util.List;
import java.util.Map;

import org.oak3ml.decisiontree.data.DataSample;

import com.google.common.base.Preconditions;

/**
 * This type of feature is similar to {@link CategoricalFeature} and can have multiple splits however its categories are a group of 
 * {@link PredicateFeature}s. So instead of binary split for single predicate it has as many splits as it has predicates. For example 
 * if we have numerical data which is discretised into bins (for example age < 10, 10 - 30, 30 -60, > 60) we can have 4 {@link PredicateFeature}s 
 * with multiple binary splits somewhere in the tree or treat it as single feature with one node and 4 splits (in which case we group predicates
 * into this feature class). All PredicateFeatures in a group must belong to the same column!
 * 
 * @author Ignas
 *
 */
public class GroupedPredicatesFeature implements Feature {
    
    /** Predicate features in the group that make up the categories of the splits. */
    private List<Feature> predicateFeatures;
    
    /** Feature column. */
    private String column;
    
    /**
     * Constructor.
     */
    private GroupedPredicatesFeature(String column, List<Feature> predicateFeatures) {
        super();
        this.predicateFeatures = predicateFeatures;
        this.column = column;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean belongsTo(DataSample dataSample) {
        for (Feature predicateFeature : predicateFeatures) {
            if (predicateFeature.belongsTo(dataSample)) {
                return true;
            }
        }
        return false;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getColumn() {
        return column;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return column + " multi split with predicates"; // maybe think of better way to name?
    }
    
    /**
     * More than 2 branches per split.
     * 
     * {@inheritDoc}
     */
    @Override
    public Map<String, List<DataSample>> split(List<DataSample> data) {
        return data.parallelStream().filter(dataSample -> belongsTo(dataSample)).collect(groupingBy(dataSample -> findPredicateFeatureForDataSample(dataSample).toString()));
    }
    
    /**
     * Find predicate (or bin) to which datasample belongs. Used when grouping in a split.
     * 
     * @param dataSample Data sample.
     *
     * @return String representation of Predicate to which data sample belongs.
     */
    public Feature findPredicateFeatureForDataSample(DataSample dataSample) {
        for (Feature predicateFeature : predicateFeatures) {
            if (predicateFeature.belongsTo(dataSample)) {
                return predicateFeature;
            }
        }
        return null;
    }

    /**
     * Static factory method to create new GroupedPredicatesFeature
     */
    public static Feature newFeature(String column, List<Feature> features) {
        Preconditions.checkNotNull(column, features);
        Preconditions.checkArgument(features.size() > 0);
        
        // validate that all features are PredicateFeature and have same column
        for (Feature feature : features) {
            if (!feature.getColumn().equals(column) || !(feature instanceof PredicateFeature)) {
                throw new IllegalArgumentException("All features must be PredicateFeature and must have same column");
            }
        }
        
        return new GroupedPredicatesFeature(column, features);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((column == null) ? 0 : column.hashCode());
        result = prime * result + ((predicateFeatures == null) ? 0 : predicateFeatures.hashCode());
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        GroupedPredicatesFeature other = (GroupedPredicatesFeature) obj;
        if (column == null) {
            if (other.column != null)
                return false;
        } else if (!column.equals(other.column))
            return false;
        if (predicateFeatures == null) {
            if (other.predicateFeatures != null)
                return false;
        } else if (!predicateFeatures.equals(other.predicateFeatures))
            return false;
        return true;
    }
    
    
}
