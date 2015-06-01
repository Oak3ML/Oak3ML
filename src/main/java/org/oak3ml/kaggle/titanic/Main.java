package org.oak3ml.kaggle.titanic;

import static org.oak3ml.decisiontree.feature.P.between;
import static org.oak3ml.decisiontree.feature.P.moreThan;
import static org.oak3ml.decisiontree.feature.P.startsWith;
import static org.oak3ml.decisiontree.feature.PredicateFeature.newFeature;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.oak3ml.decisiontree.DecisionTree;
import org.oak3ml.decisiontree.data.DataSample;
import org.oak3ml.decisiontree.data.SimpleDataSample;
import org.oak3ml.decisiontree.feature.CategoricalFeature;
import org.oak3ml.decisiontree.feature.Feature;
import org.oak3ml.decisiontree.feature.GroupedPredicatesFeature;
import org.oak3ml.decisiontree.impurity.EntropyCalculationMethod;
import org.oak3ml.decisiontree.label.BooleanLabel;
import org.oak3ml.features.discretisation.DivisiveDiscretiser;
import org.supercsv.cellprocessor.Optional;
import org.supercsv.cellprocessor.ParseBool;
import org.supercsv.cellprocessor.ParseDouble;
import org.supercsv.cellprocessor.ParseInt;
import org.supercsv.cellprocessor.ift.CellProcessor;
import org.supercsv.io.CsvListReader;
import org.supercsv.io.ICsvListReader;
import org.supercsv.prefs.CsvPreference;
import org.supercsv.util.CsvContext;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

public class Main {

    public static void main(String[] args) throws FileNotFoundException, IOException {
        
//        try (Ignite ignite = Ignition.start()) {
//            IgniteCompute compute = ignite.compute();
            
            List<DataSample> trainingData = readData(true);
            DecisionTree tree = new DecisionTree.Builder().withMaxDepth(15).withHomogenityPercentage(0.9)
                    .withImpurityCalculationMethod(new EntropyCalculationMethod()).build();
//            RandomForest tree = new RandomForest.Builder()
//                                        .withNumberOfTrees(150)
//                                        .withBootstrapingPercentage(0.6)
//                                        .withRandomFeaturesPercentage(0.3)
//                                        .build();
            
            List<Feature> features = getFeatures(trainingData);
            
            tree.train(trainingData, features);
            
            // print tree after training
            String treeJson = tree.exportAsJson();
            
            FileWriter jsonFileWriter = new FileWriter(new File("boston.json"));
            jsonFileWriter.write(treeJson);
            jsonFileWriter.flush();
            jsonFileWriter.close();
            
            // read test data
            List<DataSample> testingData = readData(false);
            List<String> predictions = Lists.newArrayList();
            // classify all test data
            for (DataSample dataSample : testingData) {
                predictions.add(dataSample.getValue("PassengerId").get() + "," + tree.classify(dataSample).getPrintValue());
            }
            
            // write predictions to file
            FileWriter fileWriter = new FileWriter(new File("predictions.csv"));
            fileWriter.append("PassengerId,Survived").append("\n");
            for (String prediction : predictions) {
                fileWriter.append(prediction).append("\n");
            }
            fileWriter.flush();
            fileWriter.close();
//        }
        
    }
    
    private static List<Feature> getFeatures(List<DataSample> trainingData) {
        Feature passengerClass = CategoricalFeature.newFeature("Pclass", Sets.newHashSet(1, 2, 3));
        Feature sex = CategoricalFeature.newFeature("Sex", Sets.newHashSet("male", "female"));
        
        Feature ageFeatures = GroupedPredicatesFeature.newFeature("Age", new DivisiveDiscretiser.Builder().build().discretise(trainingData, "Age", 3));
        List<Feature> fareFeatures = new DivisiveDiscretiser.Builder().build().discretise(trainingData, "Fare", 3);
        
       
        
        Feature zeroSiblings = newFeature("SibSp", 0);
        Feature hasSiblings = newFeature("SibSp", between(0, 2));
        Feature moreThan2Siblings = newFeature("SibSp", moreThan(2));
        Feature zeroParentsChildren = newFeature("Parch", 0);
        Feature hasParentsChildren = newFeature("Parch", between(0, 2));
        Feature moreThan2Children = newFeature("Parch", moreThan(2));
        Feature cabinA = newFeature("Cabin", startsWith("A"));
        Feature cabinB = newFeature("Cabin", startsWith("B"));
        Feature cabinC = newFeature("Cabin", startsWith("C"));
        Feature cabinD = newFeature("Cabin", startsWith("D"));
        Feature cabinE = newFeature("Cabin", startsWith("E"));
        Feature cabinF = newFeature("Cabin", startsWith("F"));
        Feature embarked = CategoricalFeature.newFeature("Embarked", Sets.newHashSet("C", "S", "Q"));
        
        List<Feature> featureList = Arrays.asList(passengerClass, sex, zeroSiblings, hasSiblings, moreThan2Siblings,
                zeroParentsChildren, hasParentsChildren, moreThan2Children,
                cabinA, cabinB, cabinC, ageFeatures,
                cabinD, cabinE, cabinF, embarked);
        
        List<Feature> editableFeatures = new ArrayList<Feature>();
        editableFeatures.addAll(fareFeatures);
        editableFeatures.addAll(featureList);
        return editableFeatures;
    }
    
    private static List<DataSample> readData(boolean training) throws IOException {
        List<DataSample> data = Lists.newArrayList();
        String filename = training ? "train.csv" : "test.csv";
        
        InputStreamReader stream = new InputStreamReader(Test.class.getResourceAsStream(filename));
        try (ICsvListReader listReader = new CsvListReader(stream, CsvPreference.STANDARD_PREFERENCE);) {
            
            // the header elements are used to map the values to the bean (names must match)
            final String[] header = listReader.getHeader(true);
            
            List<Object> values;
            while ((values = listReader.read(getProcessors(training))) != null) {
//                System.out.println(String.format("lineNo=%s, rowNo=%s, data=%s", listReader.getLineNumber(), listReader.getRowNumber(), values));
                data.add(SimpleDataSample.newSimpleDataSample("Survived", header, values.toArray()));
            }
        }
        return data;
    }
    
    private static CellProcessor[] getProcessors(boolean training) {
        // TODO fix this is ugly
        if (training) {
            final CellProcessor[] processors = new CellProcessor[] { 
                    new Optional(new ParseInt()),
                    new Optional(new ParseBooleanLabel()),
                    new Optional(new ParseInt()),
                    new Optional(),
                    new Optional(),
                    new Optional(new ParseDouble()),
                    new Optional(new ParseInt()),
                    new Optional(new ParseInt()),
                    new Optional(),
                    new Optional(new ParseDouble()),
                    new Optional(),
                    new Optional()
            };
            return processors;
        } else {
            final CellProcessor[] processors = new CellProcessor[] { 
                    new Optional(new ParseInt()),
                    new Optional(new ParseInt()),
                    new Optional(),
                    new Optional(),
                    new Optional(new ParseDouble()),
                    new Optional(new ParseInt()),
                    new Optional(new ParseInt()),
                    new Optional(),
                    new Optional(new ParseDouble()),
                    new Optional(),
                    new Optional()
            };
            return processors;
        }
    }
    
    private static class ParseBooleanLabel extends ParseBool {
        
        public Object execute(final Object value, final CsvContext context) {
            Boolean parsed = (Boolean)super.execute(value, context);
            return parsed ? BooleanLabel.TRUE_LABEL : BooleanLabel.FALSE_LABEL;
        }
        
    }
    
}
