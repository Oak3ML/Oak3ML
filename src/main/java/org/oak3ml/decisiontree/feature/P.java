package org.oak3ml.decisiontree.feature;

import java.util.function.Predicate;

/**
 * Convenience class for various predicates.
 * 
 * @author Ignas
 *
 */
public class P {
    
    public static <T> PredicateWithName<T> isEqual(T value) {
        return new PredicateWithName<T>(p -> p.equals(value), String.format("= %s", value));
    }

    public static PredicateWithName<Double> moreThanD(double value) {
        return new PredicateWithName<Double>(p -> p > value, String.format("> %s", value));
    }

    public static PredicateWithName<Double> lessThanD(double value) {
        return new PredicateWithName<Double>(p -> p < value, String.format("< %s", value));
    }

    public static PredicateWithName<Double> moreThanOrEqualD(double value) {
        return new PredicateWithName<Double>(p -> p >= value, String.format(">= %s", value));
    }
    
    public static PredicateWithName<Double> lessThanOrEqualD(double value) {
        return new PredicateWithName<Double>(p -> p <= value, String.format("<= %s", value));
    }

    public static PredicateWithName<Integer> moreThan(int value) {
        return new PredicateWithName<Integer>(p -> p > value, String.format("> %s", value));
    }

    public static PredicateWithName<Integer> lessThan(int value) {
        return new PredicateWithName<Integer>(p -> p < value, String.format("< %s", value));
    }

    public static PredicateWithName<Integer> moreThanOrEqual(int value) {
        return new PredicateWithName<Integer>(p -> p >= value, String.format(">= %s", value));
    }
    
    public static PredicateWithName<Integer> lessThanOrEqual(int value) {
        return new PredicateWithName<Integer>(p -> p <= value, String.format("<= %s", value));
    }

    public static PredicateWithName<Integer> between(int from, int to) {
        return new PredicateWithName<Integer>(moreThan(from).getPredicate().and(lessThanOrEqual(to).getPredicate()), String.format("between %s and %s", from, to));
    }

    public static PredicateWithName<Double> betweenD(double from, double to) {
        return new PredicateWithName<Double>(moreThanD(from).getPredicate().and(lessThanOrEqualD(to).getPredicate()), String.format("between %s and %s", from, to));
    }

    public static PredicateWithName<String> startsWith(String prefix) {
        return new PredicateWithName<String>(p -> p != null && p.startsWith(prefix), String.format("startsWith %s", prefix));
    }

    public static PredicateWithName<String> endsWith(String ending) {
        return new PredicateWithName<String>(p -> p != null && p.endsWith(ending), String.format("endsWith %s", ending));
    }

    public static PredicateWithName<String> containsString(String string) {
        return new PredicateWithName<String>(p -> p != null && p.contains(string), String.format("containsString %s", string));
    }
    
    /**
     * Helper class which pairs predicate and its string representation.
     * 
     * @author Ignas
     *
     * @param <T> Predicate type.
     */
    public static class PredicateWithName<T> {
        
        private Predicate<T> predicate;
        
        private String predicateName;
        
        public PredicateWithName(Predicate<T> predicate, String predicateName) {
            super();
            this.predicate = predicate;
            this.predicateName = predicateName;
        }
        
        public Predicate<T> getPredicate() {
            return predicate;
        }
        public void setPredicate(Predicate<T> predicate) {
            this.predicate = predicate;
        }
        public String getPredicateName() {
            return predicateName;
        }
        public void setPredicateName(String predicateName) {
            this.predicateName = predicateName;
        }
    }
    
}
