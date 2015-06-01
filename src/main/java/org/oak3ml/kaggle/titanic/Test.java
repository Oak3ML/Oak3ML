package org.oak3ml.kaggle.titanic;

import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCluster;
import org.apache.ignite.IgniteCompute;
import org.apache.ignite.Ignition;

public class Test {

    public static void main(String[] args) {
        try (Ignite ignite = Ignition.start()) {

            IgniteCluster cluster = ignite.cluster();

         // Compute instance over remote nodes.
         IgniteCompute compute = ignite.compute(cluster.forRemotes());

         // Print hello message on all remote nodes.
         compute.broadcast(() -> System.out.println("Hello node: " + cluster.localNode().id()));
        }
    }
}
