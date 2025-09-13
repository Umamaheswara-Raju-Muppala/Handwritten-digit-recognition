package com.digit.utility;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * GetMnistPictures
 * ----------------
 * Utility class to export sample MNIST digits (0–9) from the dataset.
 * 
 * Features:
 * - Iterates over MNIST dataset (one image at a time).
 * - Saves one normal and one inverted PNG for each digit (0–9).
 * - Output files are named: mnist_digit_{label}.png and mnist_digit_{label}_inverted.png
 */
public class GetMnistPictures {

    public static void main(String[] args) throws Exception {
        // Load MNIST training set, one image at a time
        MnistDataSetIterator mnist = new MnistDataSetIterator(1, true, 12345);

        boolean[] saved = new boolean[10]; // Track if digit (0–9) already saved
        int savedCount = 0;

        // Loop until all 10 digits are saved
        while (mnist.hasNext() && savedCount < 10) {
            DataSet ds = mnist.next();
            INDArray feature = ds.getFeatures().reshape(28, 28); // 28x28 pixel array
            int label = ds.getLabels().argMax(1).getInt(0);       // digit label (0–9)

            if (!saved[label]) {
                // --------------------------------------------
                // Create normal MNIST image
                // (black background, white digit)
                // --------------------------------------------
                BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        int v = (int) (feature.getDouble(y, x) * 255); // normalize [0,1] → [0,255]
                        int rgb = (v << 16) | (v << 8) | v;
                        img.setRGB(x, y, rgb);
                    }
                }
                File outFile = new File("mnist_digit_" + label + ".png");
                ImageIO.write(img, "png", outFile);
                System.out.println("Saved normal: " + outFile.getAbsolutePath());

                // --------------------------------------------
                // Create inverted MNIST image
                // (white background, black digit)
                // --------------------------------------------
                BufferedImage inverted = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        int gray = img.getRGB(x, y) & 0xFF;   // extract grayscale value
                        int inv = 255 - gray;                 // invert pixel
                        int newRgb = (inv << 16) | (inv << 8) | inv;
                        inverted.setRGB(x, y, newRgb);
                    }
                }
                File invFile = new File("mnist_digit_" + label + "_inverted.png");
                ImageIO.write(inverted, "png", invFile);
                System.out.println("Saved inverted: " + invFile.getAbsolutePath());

                // Mark digit as saved
                saved[label] = true;
                savedCount++;
            }
        }

        System.out.println("Exported " + savedCount + " digits with normal and inverted images.");
    }
}
