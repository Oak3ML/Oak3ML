package org.oak3ml.decisiontree;

import org.apache.ignite.IgniteCompute;
import org.oak3ml.decisiontree.impurity.GiniIndexCalculationMethod;
import org.oak3ml.decisiontree.impurity.ImpurityCalculationMethod;

/**
 * This class holds all the settings for a decision tree.
 * 
 * @author Ignas
 *
 */
public class DecisionTreeSettings {
    
    // ***** Unimplemented stopping criterions
    /** Minimum number of data samples that tree leaf must have. */
    private long minimumNumberOfInstancesPerLeaf = 20; // TODO
    
    /** If there are less than 2 branches on split then there is no point in further growing a tree. 
     * This parameter will probably rarely have to be changed.
     */
    private int minimumNumberOfSplits = 2;
    
    /** 
     * Tree growth stoppping parameter. If splitting criterion is not better than this parameter tree growth is stopped at that branch. 
     * This parameter does not have default value, because different impurity methods have different max values so it must be different 
     * for different {@link ImpurityCalculationMethod}s.
     */
    private Long impurityThreshold = null; // TODO
    
    /** If decision tree runs on Apache Ignite cluster this must be set. */
    private IgniteCompute compute;
    
    /** If {@link IgniteCompute} and this flag is set then not only tree is built using help from ignite cluster but also.  */
    private long findBestSplitOnClusterIfMoreThan = 10000000l;  // TODO
    
    /** Use paralel stream if there are more than 100k datasamples in the list. This parameter can also be set by builder. */
    private long useParalelStreamIfMoreThan = 100000l; // TODO

    /** Impurity calculation method. */
    private ImpurityCalculationMethod impurityCalculationMethod = new GiniIndexCalculationMethod();

    /**
     * When data is considered homogeneous and node becomes leaf and is labeled. If it is equal 1.0 then absolutely all
     * data must be of the same label that node would be considered a leaf.
     */
    private double homogenityPercentage = 0.95;

    /**
     * Max depth parameter. Growth of the tree is stopped once this depth is reached. Limiting depth of the tree can
     * help with overfitting, however if depth will be set too low tree will not be acurate.
     */
    private long maxDepth = 100;
    
    /** Best split finder class. */
    private BestSplitFinder bestSplitFinder = new BestSplitFinder(impurityCalculationMethod);
    
    public IgniteCompute getCompute() {
        return compute;
    }

    public void setCompute(IgniteCompute compute) {
        this.compute = compute;
    }

    public long getFindBestSplitOnClusterIfMoreThan() {
        return findBestSplitOnClusterIfMoreThan;
    }

    public void setFindBestSplitOnClusterIfMoreThan(long findBestSplitOnClusterIfMoreThan) {
        this.findBestSplitOnClusterIfMoreThan = findBestSplitOnClusterIfMoreThan;
    }

    public long getUseParalelStreamIfMoreThan() {
        return useParalelStreamIfMoreThan;
    }

    public void setUseParalelStreamIfMoreThan(long useParalelStreamIfMoreThan) {
        this.useParalelStreamIfMoreThan = useParalelStreamIfMoreThan;
    }

    public ImpurityCalculationMethod getImpurityCalculationMethod() {
        return impurityCalculationMethod;
    }

    public void setImpurityCalculationMethod(ImpurityCalculationMethod impurityCalculationMethod) {
        this.impurityCalculationMethod = impurityCalculationMethod;
        this.bestSplitFinder = new BestSplitFinder(impurityCalculationMethod);
    }

    public double getHomogenityPercentage() {
        return homogenityPercentage;
    }

    public void setHomogenityPercentage(double homogenityPercentage) {
        this.homogenityPercentage = homogenityPercentage;
    }

    public long getMaxDepth() {
        return maxDepth;
    }

    public void setMaxDepth(long maxDepth) {
        this.maxDepth = maxDepth;
    }

    public long getMinimumNumberOfInstancesPerLeaf() {
        return minimumNumberOfInstancesPerLeaf;
    }

    public void setMinimumNumberOfInstancesPerLeaf(long minimumNumberOfInstancesPerLeaf) {
        this.minimumNumberOfInstancesPerLeaf = minimumNumberOfInstancesPerLeaf;
    }

    public Long getImpurityThreshold() {
        return impurityThreshold;
    }

    public void setImpurityThreshold(Long impurityThreshold) {
        this.impurityThreshold = impurityThreshold;
    }

    public int getMinimumNumberOfSplits() {
        return minimumNumberOfSplits;
    }

    public void setMinimumNumberOfSplits(int minimumNumberOfSplits) {
        this.minimumNumberOfSplits = minimumNumberOfSplits;
    }

    public BestSplitFinder getBestSplitFinder() {
        return bestSplitFinder;
    }

}
