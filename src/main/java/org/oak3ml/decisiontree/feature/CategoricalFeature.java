package org.oak3ml.decisiontree.feature;

import static java.util.stream.Collectors.groupingBy;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.oak3ml.decisiontree.data.DataSample;

/**
 * Categorical feature can split data into multiple sublists each with separate category.
 * 
 * @author Ignas
 *
 * @param <T> Feature data type (string, number)
 */
public class CategoricalFeature<T> implements Feature {

    /** Data column used by feature. */
    private String column;

    /** Feature name used for toString() method. */
    private String name;

    /**
     * List of categories that this feature recognizes. If some data sample has value with unrecodnized category then
     * that data sample will be ignored.
     */
    private Set<T> categories;

    /**
     * Contructor.
     * 
     * @param column
     *            Data column
     * @param name
     *            Feature name.
     * @param categories
     */
    private CategoricalFeature(String column, Set<T> categories, String name) {
        super();
        this.column = column;
        this.name = name;
        this.categories = categories;
    }
    
    /**
     * If feature recognized categories contains the one in data sample then we can say that feature belongs to data sample
     */
    @Override
    public boolean belongsTo(DataSample dataSample) {
        return dataSample.getValue(column).isPresent() ? categories.contains(dataSample.getValue(column).get()) : false;
    }

    /**
     * More than 2 branches per split.
     * 
     * {@inheritDoc}
     */
    @Override
    public Map<String, List<DataSample>> split(List<DataSample> data) {
        Map<String, List<DataSample>> groupedMap = data.stream()
                .filter(dataSample -> belongsTo(dataSample) && ((DataSample)dataSample).getValue(column).isPresent())
                .collect(groupingBy(dataSample -> ((DataSample)dataSample).getValue(column).get().toString()));
        
        for (T category : categories) { // if not a single datasample contains some categories - add them by hand (todo maybe possible avoid that?)
            if (groupedMap.get(category.toString()) == null) {
                groupedMap.put(category.toString(), new ArrayList<DataSample>());
            }
        }
        
        return groupedMap;
    }

    /**
     * Factory method to create new categorical feature with fixed set of categories.
     * 
     * @param column Data column.
     * @param categories Possible categories.
     * @return New CategoricalFeature.
     */
    public static <T> Feature newFeature(String column, Set<T> categories) {
        return new CategoricalFeature<T>(column, categories, String.format("%s with %s categories", column, categories.size()));
    }

    /**
     * Factory method to create new categorical feature. Because set of categories are not provided it needs full testing data
     * 
     * @param column Data column.
     * @param trainingData Training data used to extract all possible different categories from column.
     * @return New CategoricalFeature.
     */
    public static <T> Feature newFeature(String column, List<DataSample> trainingData) {
        @SuppressWarnings("unchecked")
        Set<T> categories = trainingData.parallelStream().map(d -> (T)d.getValue(column).get()).distinct().collect(Collectors.toSet());
        return new CategoricalFeature<T>(column, categories, String.format("%s with %s categories", column, categories.size()));
    }
    
    /**
     * @return All categories of this feature.
     */
    public Set<T> getCategories() {
        return categories;
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
        return name;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((categories == null) ? 0 : categories.hashCode());
        result = prime * result + ((column == null) ? 0 : column.hashCode());
        result = prime * result + ((name == null) ? 0 : name.hashCode());
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
        @SuppressWarnings("rawtypes")
        CategoricalFeature other = (CategoricalFeature) obj;
        if (categories == null) {
            if (other.categories != null)
                return false;
        } else if (!categories.equals(other.categories))
            return false;
        if (column == null) {
            if (other.column != null)
                return false;
        } else if (!column.equals(other.column))
            return false;
        if (name == null) {
            if (other.name != null)
                return false;
        } else if (!name.equals(other.name))
            return false;
        return true;
    }

}
