
package com.digit.predict;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Path;
import java.text.DecimalFormat;

/**
 * PredictBothCNN --------------------- This class loads a trained CNN model
 * (digit-model-cnn-both.zip), preprocesses an input image into MNIST-style
 * (28x28 grayscale), and predicts the digit.
 *
 * It tries both normal and inverted images and selects the one with higher
 * confidence for prediction.
 */
public class Prediction {

	public static void main(String[] args) throws Exception {
		// Default resource (used if no argument is passed)
		String defaultResource = "/model/mnist_digit_9_inverted.png";

		BufferedImage inputImage = null;
		String usedPathDesc = null;

		// ---------------------------------------------------------
		// 1. Load image (from args → filesystem or classpath)
		// ---------------------------------------------------------
		if (args.length > 0) {
			File f = new File(args[0]);
			if (f.exists() && f.canRead()) {
				inputImage = ImageIO.read(f);
				usedPathDesc = "file:" + f.getAbsolutePath();
				System.out.println("Using filesystem image: " + f.getAbsolutePath());
			} else {
				// try loading from resources
				String rname = args[0].startsWith("/") ? args[0] : ("/" + args[0]);
				InputStream ris = Prediction.class.getResourceAsStream(rname);
				if (ris != null) {
					inputImage = ImageIO.read(ris);
					usedPathDesc = "classpath:" + rname;
					System.out.println("Using classpath resource: " + rname);
				} else {
					System.err.println("Argument not found as file or classpath resource: " + args[0]);
				}
			}
		}

		// Fallback to default resource if none provided
		if (inputImage == null) {
			InputStream ris = Prediction.class.getResourceAsStream(defaultResource);
			if (ris == null) {
				File fallbackFile = new File("src/main/resources" + defaultResource);
				if (fallbackFile.exists() && fallbackFile.canRead()) {
					inputImage = ImageIO.read(fallbackFile);
					usedPathDesc = "file:" + fallbackFile.getAbsolutePath();
					System.out.println("Using fallback filesystem image: " + fallbackFile.getAbsolutePath());
				} else {
					throw new FileNotFoundException("Default resource not found: " + defaultResource);
				}
			} else {
				inputImage = ImageIO.read(ris);
				usedPathDesc = "classpath:" + defaultResource;
				System.out.println("Using default classpath resource: " + defaultResource);
			}
		}

		if (inputImage == null) {
			throw new IOException("Failed to load input image (ImageIO returned null).");
		}

		// ---------------------------------------------------------
		// 2. Load trained CNN model
		// ---------------------------------------------------------
		MultiLayerNetwork model = null;
		InputStream modelIs = Prediction.class.getResourceAsStream("digit-model-cnn-both.zip");
		if (modelIs != null) {
			System.out.println("Loading model from classpath /model/digit-model-cnn-both.zip");
			model = ModelSerializer.restoreMultiLayerNetwork(modelIs);
		} else {
			File modelFile = Path.of("digit-model-cnn-both.zip").toFile();
			if (modelFile.exists()) {
				System.out.println("Loading model from file: " + modelFile.getAbsolutePath());
				model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
			} else {
				File alt = new File("src/main/resources/model/digit-model-cnn-both.zip");
				if (alt.exists()) {
					System.out.println("Loading model from src/main/resources/model/digit-model-cnn-both.zip");
					model = ModelSerializer.restoreMultiLayerNetwork(alt);
				} else {
					throw new FileNotFoundException(
							"Model not found. Put digit-model-cnn-both.zip in resources or project root.");
				}
			}
		}
		System.out.println("Model loaded.");

		// ---------------------------------------------------------
		// 3. Preprocess image (normal + inverted)
		// ---------------------------------------------------------
		double[] normal = preprocessToFlattened(inputImage, false);
		double[] inverted = preprocessToFlattened(inputImage, true);

		// Save debug images (for verification)
		BufferedImage debug = flattenedToBuffered(normal);
		ImageIO.write(debug, "png", new File("debug_normal.png"));
		BufferedImage debugInv = flattenedToBuffered(inverted);
		ImageIO.write(debugInv, "png", new File("debug_inverted.png"));
		System.out.println("Saved debug_normal.png and debug_inverted.png");

		// ---------------------------------------------------------
		// 4. Run prediction on both normal & inverted inputs
		// ---------------------------------------------------------
		INDArray arr = Nd4j.create(normal).reshape(1, 784);
		INDArray arrInv = Nd4j.create(inverted).reshape(1, 784);

		INDArray out = model.output(arr, false);
		INDArray outInv = model.output(arrInv, false);

		int pred = out.argMax(1).getInt(0);
		double predProb = out.getDouble(0, pred);

		int predInv = outInv.argMax(1).getInt(0);
		double predInvProb = outInv.getDouble(0, predInv);

		DecimalFormat df = new DecimalFormat("#0.000");

		// Choose prediction with higher confidence
		if (predInvProb > predProb) {
			System.out.println("Used inverted image (higher confidence).");
			printPrediction(outInv, predInv, df, usedPathDesc);
		} else {
			System.out.println("Used normal image.");
			printPrediction(out, pred, df, usedPathDesc);
		}
	}

	/**
	 * Print predictions with probabilities for all classes.
	 */
	private static void printPrediction(INDArray output, int predictedLabel, DecimalFormat df, String imagePath) {
		System.out.println("Image: " + imagePath);
		System.out.println("Predicted digit: " + predictedLabel);
		System.out.println("Top probability: " + df.format(output.getDouble(0, predictedLabel)));
		System.out.println("All probabilities:");
		double[] probs = output.toDoubleVector();
		for (int i = 0; i < probs.length; i++) {
			System.out.println(i + ": " + df.format(probs[i]));
		}
	}

	/**
	 * Convert BufferedImage → MNIST-style 28x28 normalized array. Steps: 1. Convert
	 * to grayscale 2. Apply threshold and find bounding box 3. Resize digit to
	 * 20x20 while preserving aspect ratio 4. Center into 28x28 canvas 5. Normalize
	 * pixel values (0–1)
	 */
	private static double[] preprocessToFlattened(BufferedImage src, boolean invert) throws IOException {
		if (src == null)
			throw new IllegalArgumentException("Source image is null");

		int width = src.getWidth();
		int height = src.getHeight();

		// Step 1: Grayscale conversion
		double[][] gray = new double[height][width];
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int rgb = src.getRGB(x, y);
				int r = (rgb >> 16) & 0xFF;
				int g = (rgb >> 8) & 0xFF;
				int b = rgb & 0xFF;
				double lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
				double p = invert ? 1.0 - lum : lum;
				gray[y][x] = p;
			}
		}

		// Step 2: Bounding box detection
		double threshold = 0.08;
		int minX = width, minY = height, maxX = -1, maxY = -1;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (gray[y][x] > threshold) {
					if (x < minX)
						minX = x;
					if (x > maxX)
						maxX = x;
					if (y < minY)
						minY = y;
					if (y > maxY)
						maxY = y;
				}
			}
		}

		if (maxX < minX || maxY < minY) {
			return new double[28 * 28]; // blank
		}

		int bw = maxX - minX + 1;
		int bh = maxY - minY + 1;

		// Step 3: Crop digit
		BufferedImage digitBox = new BufferedImage(bw, bh, BufferedImage.TYPE_BYTE_GRAY);
		for (int y = 0; y < bh; y++) {
			for (int x = 0; x < bw; x++) {
				int v = (int) Math.round(gray[minY + y][minX + x] * 255.0);
				int rgb = (v << 16) | (v << 8) | v;
				digitBox.setRGB(x, y, rgb);
			}
		}

		// Step 4: Resize to 20x20
		final int boxSize = 20;
		int newW, newH;
		if (bw >= bh) {
			newW = boxSize;
			newH = Math.max(1, (int) Math.round((double) bh / bw * boxSize));
		} else {
			newH = boxSize;
			newW = Math.max(1, (int) Math.round((double) bw / bh * boxSize));
		}

		BufferedImage resized = new BufferedImage(newW, newH, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g2 = resized.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g2.drawImage(digitBox, 0, 0, newW, newH, null);
		g2.dispose();

		// Step 5: Center into 28x28
		BufferedImage final28 = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g3 = final28.createGraphics();
		g3.setColor(Color.BLACK);
		g3.fillRect(0, 0, 28, 28);
		int offsetX = (28 - newW) / 2;
		int offsetY = (28 - newH) / 2;
		g3.drawImage(resized, offsetX, offsetY, null);
		g3.dispose();

		// Step 6: Flatten into double[]
		double[] flat = new double[28 * 28];
		for (int y = 0; y < 28; y++) {
			for (int x = 0; x < 28; x++) {
				int rgb = final28.getRGB(x, y);
				int v = rgb & 0xFF;
				flat[y * 28 + x] = v / 255.0;
			}
		}
		return flat;
	}

	/**
	 * Convert flattened double array back to BufferedImage (for debugging).
	 */
	private static BufferedImage flattenedToBuffered(double[] flat) {
		BufferedImage out = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
		for (int y = 0; y < 28; y++) {
			for (int x = 0; x < 28; x++) {
				int v = (int) Math.round(Math.max(0, Math.min(1, flat[y * 28 + x])) * 255);
				int rgb = (v << 16) | (v << 8) | v;
				out.setRGB(x, y, rgb);
			}
		}
		return out;
	}
}
