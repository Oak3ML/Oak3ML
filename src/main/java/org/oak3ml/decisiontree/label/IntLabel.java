package org.oak3ml.decisiontree.label;

/**
 * Simplest possible label. Simply labels data as true or false.
 * 
 * @author Ignas
 *
 */
public class IntLabel extends Label {

    /** Label. */
    private int label;

    /**
     * Constructor.
     */
    private IntLabel(int label) {
        super();
        this.label = label;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getPrintValue() {
        return getName();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getName() {
        return String.valueOf(label);
    }

    /**
     * Static factory method.
     */
    public static Label newLabel(Integer label) {
        return new IntLabel(label);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + label;
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
        IntLabel other = (IntLabel) obj;
        if (label != other.label)
            return false;
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return "IntLabel [label=" + label + "]";
    }

}
