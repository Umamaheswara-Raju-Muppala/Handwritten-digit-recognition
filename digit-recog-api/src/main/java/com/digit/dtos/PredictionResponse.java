package com.digit.dtos;

public class PredictionResponse {
	private int predictedDigit;
	private double confidence;

	public PredictionResponse(int predictedDigit, double confidence) {
		this.predictedDigit = predictedDigit;
		this.confidence = confidence;
	}

	public int getPredictedDigit() {
		return predictedDigit;
	}

	public double getConfidence() {
		return confidence;
	}
}
