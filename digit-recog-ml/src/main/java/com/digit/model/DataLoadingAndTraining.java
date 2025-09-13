package com.digit.model;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * DataLoadingAndTraining
 * ..................................
 * This class loads the MNIST dataset, builds a Convolutional Neural Network (CNN),
 * trains it on both normal and inverted digit images, evaluates accuracy, and saves
 * the trained model to a .zip file.
 */
public class DataLoadingAndTraining {

    public static void main(String[] args) throws Exception {
        int batchSize = 64;   // Number of samples per mini-batch
        int epochs = 7;       // Number of training epochs
        int seed = 123;       // Random seed for reproducibility

        // =======================
        // Load MNIST dataset
        // =======================
        // mnistTrain → training set (60k images)
        // mnistTest  → testing set (10k images)
        MnistDataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        MnistDataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        // =======================
        // Build CNN model
        // =======================
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-3)) // Adam optimizer with learning rate = 0.001
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(1) // Input: 1 channel (grayscale image)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}).build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}).build())
                .layer(new DenseLayer.Builder()
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10) // 10 output classes (digits 0–9)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutionalFlat(28, 28, 1))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100)); // Print score every 100 iterations

        // =======================
        // Training Loop
        // =======================
        System.out.println("Training CNN on normal + inverted MNIST...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            while (mnistTrain.hasNext()) {
                DataSet ds = mnistTrain.next();

                // Train on original (normal) batch
                model.fit(ds);

                // Train again on inverted batch (1 - pixelValue)
                INDArray features = ds.getFeatures();
                INDArray inverted = features.rsub(1.0);
                DataSet invertedDs = new DataSet(inverted, ds.getLabels());
                model.fit(invertedDs);
            }
            mnistTrain.reset();
            System.out.println("Epoch " + (epoch + 1) + " complete.");
        }

        // =======================
        // Evaluation
        // =======================
        System.out.println("Evaluating model...");
        var eval = model.evaluate(mnistTest);
        System.out.println(eval.stats()); // Print accuracy, precision, recall, etc.

        // =======================
        // Save trained model
        // =======================
        File modelFile = new File("digit-model-cnn-both.zip");
        ModelSerializer.writeModel(model, modelFile, true);
        System.out.println("Model saved to " + modelFile.getAbsolutePath());
    }
}
