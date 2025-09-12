package com.digit.service;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import com.digit.dtos.PredictionResponse;

import jakarta.annotation.PostConstruct;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Path;
import java.text.DecimalFormat;

@Service
public class PredictionService {

	private MultiLayerNetwork model;

	@PostConstruct
	public void loadModel() throws IOException {
		InputStream modelIs = getClass().getResourceAsStream("/model/digit-model-cnn-both.zip");
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
	}

	public PredictionResponse predict(MultipartFile file) throws Exception {
		BufferedImage inputImage = ImageIO.read(file.getInputStream());
		if (inputImage == null) {
			throw new IOException("Failed to load input image.");
		}

		// Preprocess (normal + inverted)
		double[] normal = preprocessToFlattened(inputImage, false);
		double[] inverted = preprocessToFlattened(inputImage, true);

		INDArray arr = Nd4j.create(normal).reshape(1, 784);
		INDArray arrInv = Nd4j.create(inverted).reshape(1, 784);

		INDArray out = model.output(arr, false);
		INDArray outInv = model.output(arrInv, false);

		int pred = out.argMax(1).getInt(0);
		double predProb = out.getDouble(0, pred);

		int predInv = outInv.argMax(1).getInt(0);
		double predInvProb = outInv.getDouble(0, predInv);

		DecimalFormat df = new DecimalFormat("#0.000");

		if (predInvProb > predProb) {

			printPrediction(outInv, predInv, df, file.getOriginalFilename());
			return new PredictionResponse(predInv, predInvProb);
		} else {
			printPrediction(out, pred, df, file.getOriginalFilename());
			return new PredictionResponse(pred, predProb);
		}
	}

	private void printPrediction(INDArray output, int predictedLabel, DecimalFormat df, String imagePath) {

		System.out.println("Predicted digit: " + predictedLabel);
		System.out.println("Top probability: " + df.format(output.getDouble(0, predictedLabel)));

	}

	private double[] preprocessToFlattened(BufferedImage src, boolean invert) throws IOException {
		if (src == null)
			throw new IllegalArgumentException("Source image is null");

		int width = src.getWidth();
		int height = src.getHeight();

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
			return new double[28 * 28];
		}

		int bw = maxX - minX + 1;
		int bh = maxY - minY + 1;

		BufferedImage digitBox = new BufferedImage(bw, bh, BufferedImage.TYPE_BYTE_GRAY);
		for (int y = 0; y < bh; y++) {
			for (int x = 0; x < bw; x++) {
				int v = (int) Math.round(gray[minY + y][minX + x] * 255.0);
				int rgb = (v << 16) | (v << 8) | v;
				digitBox.setRGB(x, y, rgb);
			}
		}

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

		BufferedImage final28 = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g3 = final28.createGraphics();
		g3.setColor(Color.BLACK);
		g3.fillRect(0, 0, 28, 28);
		int offsetX = (28 - newW) / 2;
		int offsetY = (28 - newH) / 2;
		g3.drawImage(resized, offsetX, offsetY, null);
		g3.dispose();

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

	private BufferedImage flattenedToBuffered(double[] flat) {
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
