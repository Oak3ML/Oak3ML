package org.oak3ml.decisiontree.feature;

import java.util.Optional;
import java.util.function.Predicate;

import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.feature.P.PredicateWithName;

/**
 * Feature type that splits data into 2 sublists - one with data that has that feature and one that doesn't.
 * 
 * @author Ignas
 *
 * @param <T> Feature data type (string, number)
 */
public class PredicateFeature<T> implements Feature {
    
    /** Data column used by feature. */
    private String column; // TODO multiple columns per feature

    /** Predicate used for splitting. */
    private Predicate<T> predicate;
    
    /** 
     * Feature name used for toString() method.
     * ATTENTION! MUST represent correctly otherwise {@link equals} method might work incorrectly. If you set this field manually take care of it. 
     * @See {@link #equals} 
     */
    private String name;

    /**
     * Constructor.
     * 
     * @param column Column in data table.
     * @param predicateWithName Predicate used for splitting. For example if value is equal to some value, or is more/less. Also its name in string representation.
     * @param name Feature name.
     */
    private PredicateFeature(String column, PredicateWithName<T> predicateWithName) {
        super();
        this.column = column;
        this.predicate = predicateWithName.getPredicate();
        this.name = String.format("%s %s", column, predicateWithName.getPredicateName());
    }

    /**
     * {@inheritDoc}
     */
    @SuppressWarnings("unchecked")
    @Override
    public boolean belongsTo(DataSample dataSample) { // TODO implement other splits (in different type of feature)
        Optional<Object> optionalValue = dataSample.getValue(column);
        return optionalValue.isPresent() ? predicate.test((T)optionalValue.get()) : false;
    }

    /**
     * Default static factory method which creates a feature. Default feature splits data whose column value is equal provided feature value.
     * For example PredicateFeature.newFeature("name", "john") will split data into 2 sublists - one where all entries has name = john and another one with different names.
     * 
     * @param column Column to use in data.
     * @param featureValue Feature value.
     * @return New feature.
     */
    public static <T> Feature newFeature(String column, T featureValue) {
        return new PredicateFeature<T>(column, P.isEqual(featureValue));
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public String getColumn() {
        return column;
    }

    /**
     * Create feature. Accepts {@link PredicateWithName} from {@link P} class which contains various predicates already created.
     * Example usage:
     * <code>
     *  PredicateFeature.newFeature("age", P.moreThan(10));
     * </code>
     * 
     * @param column Column to use in data.
     * @param predicateWithName Predicate to use for splitting and its string representation with name. Check out {@link #equals}.
     * 
     * @return @return New feature.
     */
    public static <T> Feature newFeature(String column, PredicateWithName<T> predicateWithName) {
        return new PredicateFeature<T>(column, predicateWithName);
    }

    /**
     * Static factory method to create a new feature. This one accepts any predicate.
     * 
     * ATTENTION! predicateString MUST represent predicate correctly otherwise {@link equals} method might work incorrectly. If you set this field manually take care of it. 
     * @See {@link #equals} 
     * 
     * @param column Column to use in data.
     * @param predicate Predicate to use for splitting.
     * @return New feature.
     */
    public static <T> Feature newFeature(String column, Predicate<T> predicate, String predicateString) {
        return new PredicateFeature<T>(column, new PredicateWithName<T>(predicate, String.format("%s %s", column, predicateString)));
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
        result = prime * result + ((column == null) ? 0 : column.hashCode());
        result = prime * result + ((name == null) ? 0 : name.hashCode());
        return result;
    }

    /**
     * Because there are no clear way to compare lambdas (or {@link Predicate} see here: http://stackoverflow.com/questions/24095875/is-there-a-way-to-compare-lambdas) 
     * we cannot use predicate in equals method thats why name (string representation of predicate) must be used instead. This is a walk on a thin ice as there is no
     * way to force that name string represents feature's predicate correctly so its up to developer to make sure of that.
     * 
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
        PredicateFeature other = (PredicateFeature) obj;
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
