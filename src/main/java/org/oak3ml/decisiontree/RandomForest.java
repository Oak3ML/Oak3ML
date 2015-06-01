package org.oak3ml.decisiontree;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toSet;

import java.util.List;
import java.util.Random;
import java.util.Set;

import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.label.Label;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multisets;

/**
 * Random forest implementation.
 * 
 * @author Ignas
 *
 */
public class RandomForest {

    /** Random number generator. */ // TODO replace?
    private Random random = new Random(System.nanoTime());

    private RandomForestSettings settings;
    
    /**
     * Private constructor.
     */
    private RandomForest(Builder builder) {
        super();
        settings = new RandomForestSettings();
        if (builder.trees != null)
            settings.setTrees(builder.trees);
        if (builder.bootstrapingPercentage != null)
            settings.setBootstrapingPercentage(builder.bootstrapingPercentage);
        if (builder.randomFeaturesPercentage != null)
            settings.setRandomFeaturesPercentage(builder.randomFeaturesPercentage);
        if (builder.numberOfTrees != null)
            settings.setNumberOfTrees(builder.numberOfTrees);
    }

    /**
     * Trains each {@link DecisionTree} in an ensemble (forest).
     * 
     * @param trainingData Training data.
     * @param features Features.
     */
    public void train(List<DataSample> trainingData, List<Feature> features) {
        
        checkArgument(settings.getBootstrapingPercentage() < 1.0 && settings.getBootstrapingPercentage() > 0);
        checkArgument(settings.getRandomFeaturesPercentage() < 1.0 && settings.getRandomFeaturesPercentage() > 0);
        checkNotNull(settings.getTrees());
        checkArgument(settings.getTrees().size() >= 2);
        
        for (DecisionTree tree : settings.getTrees()) {
            // filter datasamples randomly (using approximately bootstrapingPercentage amount of samples)
            List<DataSample> bootstrapSamples = trainingData.stream().filter(d -> random.nextInt(100) < 100 * settings.getBootstrapingPercentage()).collect(toList());
            Set<String> randomColumnsToUse = features.stream().map(f -> f.getColumn()).filter(f -> random.nextInt(100) < 100 * settings.getRandomFeaturesPercentage()).collect(toSet());
            List<Feature> randomFeatures = features.stream().filter(f -> randomColumnsToUse.contains(f.getColumn())).collect(toList());
            
            tree.train(bootstrapSamples, randomFeatures);
        }
        
    }

    /**
     * Classify data sample by using all DecisionTrees in ensemble. Most accepted Label after vote is returned.
     * 
     * @param dataSample Data sample to classify.
     * @return Classification label.
     */
    public Label classify(DataSample dataSample) {

        Multiset<Label> countedLabels = HashMultiset.create();

        // check classification of each tree and count labels
        for (DecisionTree tree : settings.getTrees()) {
            Label label = tree.classify(dataSample);
            countedLabels.add(label);
        }

        // return most common label
        return Multisets.copyHighestCountFirst(countedLabels).iterator().next();
    }
    
    /* Builder */
    public static class Builder {

        private List<DecisionTree> trees;

        private Double randomFeaturesPercentage;
        
        private Double bootstrapingPercentage;

        private Integer numberOfTrees;
        
        public Builder withTrees(List<DecisionTree> trees) {
            this.trees = trees;
            return this;
        }

        public Builder withNumberOfTrees(int numberOfTrees) {
            this.numberOfTrees = numberOfTrees;
            return this;
        }
        
        public Builder withRandomFeaturesPercentage(double randomFeaturesPercentage) {
            this.randomFeaturesPercentage = randomFeaturesPercentage;
            return this;
        }

        public Builder withBootstrapingPercentage(double bootstrapingPercentage) {
            this.bootstrapingPercentage = bootstrapingPercentage;
            return this;
        }
        
        public RandomForest build() {
            return new RandomForest(this);
        }

    }
}
