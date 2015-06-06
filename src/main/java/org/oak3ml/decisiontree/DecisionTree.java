package org.oak3ml.decisiontree;

import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toList;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.ignite.IgniteCompute;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.feature.CategoricalFeature;
import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.feature.GroupedPredicatesFeature;
import org.oak3ml.decisiontree.feature.PredicateFeature;
import org.oak3ml.decisiontree.impurity.ImpurityCalculationMethod;
import org.oak3ml.decisiontree.label.Label;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Maps;

/**
 * Decision tree implementation.
 * 
 * @author Ignas
 *
 */
public class DecisionTree {

    /** Logger. */
    private Logger log = LoggerFactory.getLogger(DecisionTree.class);

    /** Root node. */
    private Node root;
    
    /** Various settings. */
    private DecisionTreeSettings settings;
    
    /**
     * Private constructor user by builder to return new decision tree instance.
     * 
     * @param builder Builder with decision tree parameters.
     */
    private DecisionTree(Builder builder) {
        super();
        settings = new DecisionTreeSettings();
        if (builder.calculationMethod != null)
            settings.setImpurityCalculationMethod(builder.calculationMethod);
        if (builder.homogenityPercentage != null)
            settings.setHomogenityPercentage(builder.homogenityPercentage);
        if (builder.findBestSplitOnClusterIfMoreThan != null)
            settings.setFindBestSplitOnClusterIfMoreThan(builder.findBestSplitOnClusterIfMoreThan);
        if (builder.computeGrid != null)
            settings.setCompute(builder.computeGrid);
        if (builder.maxDepth != null)
            settings.setMaxDepth(builder.maxDepth);
        if (builder.useParalelStreamIfMoreThan != null)
            settings.setUseParalelStreamIfMoreThan(builder.useParalelStreamIfMoreThan);
    }

    /**
     * Get root.
     */
    public Node getRoot() {
        return root;
    }

    /**
     * Trains tree on training data for provided features.
     * 
     * @param trainingData
     *            List of training data samples.
     * @param features
     *            List of possible features.
     */
    public void train(List<DataSample> trainingData, List<Feature> features) {
        root = growTree(trainingData, features, 1, "root");
    }
    
    /**
     * Grow tree during training by splitting data recusively on best feature.
     * 
     * Pseudocode:
     * 
     * GrowTree(D, F) – grow a feature tree from training data.
     *
     *   Input : data D; set of features F.
     *   Output : feature tree T with labelled leaves.
     *    if Homogeneous(D) then return Label(D) ;
     *    S = BestSplit(D, F) ; 
     *    split D into subsets Di according to the literals in S;
     *    for each i do
     *        if Di not empty then Ti = GrowTree(Di, F) else Ti is a leaf labelled with Label(D);
     *    end
     *    return a tree whose root is labelled with S and whose children are Ti
     * 
     * @param trainingData
     *            List of training data samples.
     * @param features
     *            List of possible features.
     * @param branchFromParent
     *            Branch name from parent so children nodes know on what condition their split was.
     * 
     * @return Node after split. For a first invocation it returns tree root node.
     */
    protected Node growTree(List<DataSample> trainingData, List<Feature> features, int currentDepth, Object branchFromParent) {
        Map<Label, Long> countedSamples = countNbOfSamples(trainingData);

        Label currentNodeLabel = null;
        // if dataset already homogeneous enough (has label assigned) make this node a leaf
        if ((currentNodeLabel = getLabel(trainingData)) != null) {
            log.debug("New leaf is created because data is homogeneous: {}", currentNodeLabel.getName());
            return Node.newLeafNode(currentNodeLabel, branchFromParent, countedSamples);
        }
        
        // check if there are more features and tree is not too deep before splitting
        boolean stoppingCriteriaReached = features.isEmpty() || currentDepth >= settings.getMaxDepth();
        if (stoppingCriteriaReached) {
            Label majorityLabel = getMajorityLabel(countedSamples);
            log.debug("New leaf is created because stopping criteria reached: {}", majorityLabel.getName());
            return Node.newLeafNode(majorityLabel, branchFromParent, countedSamples);
        }

        Feature bestSplit = settings.getBestSplitFinder().findBestSplitFeature(trainingData, features);
        log.debug("Best split found: {}", bestSplit.toString());
        Map<String, List<DataSample>> splitData = bestSplit.split(trainingData);

        // remove best split from features (TODO check if it is not slow)
        List<Feature> featuresWithoutSplitFeature = features.stream().filter(f -> !f.equals(bestSplit)).collect(toList());
        Node node = Node.newNode(bestSplit, branchFromParent, countedSamples);
        
        // check for another stopping criteria after we calculated a split
        boolean stoppingCriteriaWithInformationFromSplitReached = splitData.keySet().size() < settings.getMinimumNumberOfSplits();
        if (stoppingCriteriaWithInformationFromSplitReached) {
            Label majorityLabel = getMajorityLabel(countedSamples);
            log.debug("New leaf is created because stopping criteria after split reached: {}", majorityLabel.getName());
            return Node.newLeafNode(majorityLabel, branchFromParent, countedSamples);
        }
        
        Set<Entry<String, List<DataSample>>> treeBranches = splitData.entrySet();
        for (Entry<String, List<DataSample>> branch : treeBranches) {
            // branch name passed to children is kept as a key of the split map
            Object branchName = branch.getKey();
            // all data that belongs to that branch
            List<DataSample> subsetTrainingData = branch.getValue();
            
            if (subsetTrainingData == null || subsetTrainingData.isEmpty()) {
                // if subset data is empty add a leaf with label calculated from initial data
                // it has no counted data samples on the leaf, so empty map
                node.addChild(Node.newLeafNode(getMajorityLabel(countedSamples), branchName, Maps.newHashMap()));
            } else {
                // if we have clusters - calculate branches on other machines
                if (settings.getCompute() != null) {
                    // grow tree further recursively with cluster
                    node.addChild(settings.getCompute().call(() -> growTree(subsetTrainingData, featuresWithoutSplitFeature, currentDepth + 1, branchName)));
                } else {
                    node.addChild(growTree(subsetTrainingData, featuresWithoutSplitFeature, currentDepth + 1, branchName));
                }
            }
        }

        return node;
    }

    /**
     * Classify dataSample.
     * 
     * @param dataSample
     *            Data sample
     * @return Return label of class.
     */
    public Label classify(DataSample dataSample) {
        Node node = root;
        boolean branchFound = true;
        while (!node.isLeaf()) { // go through tree until leaf is reached
            if (!branchFound) {
                return getMajorityLabel(node.getCountedSamples());
            }
            branchFound = false;
            for (Node child : node.getChildren()) {
                Feature feature = node.getFeature(); // TODO refactor
                if (feature instanceof PredicateFeature) { // moving through predicate binary splits and categorical multisplits are different
                    if (dataSample.has(feature) && child.getBranchValue().equals("true") || 
                            !dataSample.has(feature) && child.getBranchValue().equals("false")) {
                        node = child;
                        branchFound = true;
                        break;
                    }
                } else if (feature instanceof CategoricalFeature) {
                    if (child.getBranchValue().equals(dataSample.getValue(feature.getColumn()).get().toString())) {
                        node = child;
                        branchFound = true;
                        break;
                    }
                } else if (feature instanceof GroupedPredicatesFeature) {
                    Feature predicateFeature = ((GroupedPredicatesFeature)feature).findPredicateFeatureForDataSample(dataSample);
                    if (predicateFeature != null && child.getBranchValue().equals(predicateFeature.toString())) {
                        node = child;
                        branchFound = true;
                        break;
                    }
                }
            }
        }
        return node.getLabel();
    }

    /**
     * Returns Label if data is homogeneous.
     */
    protected Label getLabel(List<DataSample> data) {
        // group by to map <Label, count>
        Map<Label, Long> labelCount = countNbOfSamples(data);
        long totalCount = data.size();
        for (Label label : labelCount.keySet()) {
            long nbOfLabels = labelCount.get(label);
            if (((double) nbOfLabels / (double) totalCount) >= settings.getHomogenityPercentage()) {
                return label;
            }
        }
        return null;
    }

    /**
     * Differs from getLabel() that it always return some label and does not look at homogenityPercentage parameter. It
     * is used when tree growth is stopped and everything what is left must be classified so it returns majority label for the data.
     */
    protected Label getMajorityLabel(Map<Label, Long> countedSamples) {
        // group by to map <Label, count> like in getLabels() but return Label with most counts
        return countedSamples.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }

    /**
     * Count number of datasamples for each Label.
     */
    protected Map<Label, Long> countNbOfSamples(List<DataSample> data) {
        // group by to map <Label, count>
        return data.parallelStream().collect(groupingBy(DataSample::getLabel, counting()));
    }
    
    /**
     * Export decision tree as JSON string.
     */
    public String exportAsJson() {
        return printNode(root);
    }
        
    
    private String printNode(Node node) {
        if (node.isLeaf()) {
            throw new RuntimeException("Invalid node");
        }
        
        String nodeString = "{\"error\":\"1.0\", \"value\":[\"" + node.getBranchValue() + "\"], \"samples\":\"1\", \"label\":\""  + node.getName() + "\", \"type\":\"split\"";
        
        boolean firstChild = true;
        nodeString += ", \"children\": [";
        for (Node child : node.getChildren()) {
            if (!firstChild) {
                nodeString += ",";
            }
            firstChild = false;
            if (!child.isLeaf()) {
                nodeString += printNode(child);
            } else {
                nodeString += "{\"error\":\"1.0\", \"value\":[" + child.getLabel().getPrintValue() + "], \"samples\":\"1\", \"label\":\""  + node.getName() + "\", \"type\":\"leaf\"}";
            }
        }
        nodeString += "]}";
        return nodeString;
    }

    /* Builder */
    public static class Builder {

        private ImpurityCalculationMethod calculationMethod;
        
        private Double homogenityPercentage;

        private Long findBestSplitOnClusterIfMoreThan;
        
        private Long useParalelStreamIfMoreThan;

        private Long maxDepth;
        
        private IgniteCompute computeGrid;
        
        public Builder withImpurityCalculationMethod(ImpurityCalculationMethod calculationMethod) {
            this.calculationMethod = calculationMethod;
            return this;
        }

        public Builder withHomogenityPercentage(double homogenityPercentage) {
            this.homogenityPercentage = homogenityPercentage;
            return this;
        }

        public Builder withMaxDepth(long maxDepth) {
            this.maxDepth = maxDepth;
            return this;
        }

        public Builder withComputeGrid(IgniteCompute computeGrid) {
            this.computeGrid = computeGrid;
            return this;
        }

        public Builder findBestSplitOnClusterIfMoreThan(long findBestSplitOnClusterIfMoreThan) {
            this.findBestSplitOnClusterIfMoreThan = findBestSplitOnClusterIfMoreThan;
            return this;
        }

        public Builder useParalelStreamIfMoreThan(long useParalelStreamIfMoreThan) {
            this.useParalelStreamIfMoreThan = useParalelStreamIfMoreThan;
            return this;
        }
        
        public DecisionTree build() {
            return new DecisionTree(this);
        }
    }
}
