package org.oak3ml.decisiontree;

import java.util.List;
import java.util.Map;

import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.label.Label;

import com.google.common.collect.Lists;

/**
 * Node of the decision tree. Leaf nodes has {@link Label}s set. It can have list of children nodes. It also contains a
 * feature on which it will split futher if it is not leaf node.
 * 
 * @author Ignas
 *
 */
public class Node {

    private static final String LEAF_NODE_NAME = "Leaf";

    /** Node's feature used to split it further. */
    private Feature feature;

    /** Value of parent node split. It is tree's branch (or edge). */
    private Object branchValue;

    /** Label if it is leaf node. */
    private Label label;
    
    /** Number of datasamples for each label seen by this node. */
    private Map<Label, Long> countedSamples;

    /** Node's children. */
    private List<Node> children = Lists.newArrayList();

    /**
     * Protected private constructor.
     */
    private Node(Feature feature, Object branchValue, Map<Label, Long> countedSamples) {
        this.feature = feature;
        this.branchValue = branchValue;
        this.countedSamples = countedSamples;
    }

    /**
     * Protected private constructor with Label.
     */
    private Node(Feature feature, Label label, Object branchValue, Map<Label, Long> countedSamples) {
        this.label = label;
        this.feature = feature;
        this.branchValue = branchValue;
        this.countedSamples = countedSamples;
    }

    /**
     * Static factory method.
     */
    public static Node newNode(Feature feature, Object branchValue, Map<Label, Long> countedSamples) {
        return new Node(feature, branchValue, countedSamples);
    }

    /**
     * Static factory method for a leaf node.
     */
    public static Node newLeafNode(Label label, Object branchValue, Map<Label, Long> countedSamples) {
        return new Node(null, label, branchValue, countedSamples);
    }

    public void addChild(Node child) {
        children.add(child);
    }

    public List<Node> getChildren() {
        return children;
    }

    public Label getLabel() {
        return label;
    }

    public boolean isLeaf() {
        return label != null;
    }

    public Object getBranchValue() {
        return branchValue;
    }

    public Feature getFeature() {
        return feature;
    }

    public String getName() {
        return feature != null ? feature.toString() : LEAF_NODE_NAME;
    }

    public Map<Label, Long> getCountedSamples() {
        return countedSamples;
    }

}
