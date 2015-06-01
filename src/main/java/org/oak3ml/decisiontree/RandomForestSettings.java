package org.oak3ml.decisiontree;

import java.util.List;
import java.util.stream.IntStream;

import com.google.common.collect.Lists;

/**
 * This class holds all the settings for a decision tree.
 * 
 * @author Ignas
 *
 */
public class RandomForestSettings {
    
    /**
     * What percentage of original features to use for building each model. One column = one feature even if it has more
     * features associated (in case of discretisation for example.
     */
    private double randomFeaturesPercentage = 0.2;

    /** What part of original training data to use for building boostrap sample. */
    private double bootstrapingPercentage = 0.8;
    
    /** Number of trees in the random forest. */
    private int numberOfTrees = 20;

    /** Trees in the forest. */
    private List<DecisionTree> trees = Lists.newArrayList();

    /**
     * Constructor. Creates default number of trees for the forest. It can be overriden by setting trees manually.
     */
    public RandomForestSettings() {
        super();
        IntStream.range(0, numberOfTrees).forEach(i -> trees.add(new DecisionTree.Builder().build()));
    }

    public double getRandomFeaturesPercentage() {
        return randomFeaturesPercentage;
    }

    public void setRandomFeaturesPercentage(double randomFeaturesPercentage) {
        this.randomFeaturesPercentage = randomFeaturesPercentage;
    }

    public double getBootstrapingPercentage() {
        return bootstrapingPercentage;
    }

    public void setBootstrapingPercentage(double bootstrapingPercentage) {
        this.bootstrapingPercentage = bootstrapingPercentage;
    }

    public int getNumberOfTrees() {
        return numberOfTrees;
    }

    public void setNumberOfTrees(int numberOfTrees) {
        this.numberOfTrees = numberOfTrees;
    }

    public List<DecisionTree> getTrees() {
        return trees;
    }

    public void setTrees(List<DecisionTree> trees) {
        this.trees = trees;
    }
    
}
