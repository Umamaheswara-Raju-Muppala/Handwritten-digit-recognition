package com.digit.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.digit.dtos.PredictionResponse;
import com.digit.exceptions.FileMissingException;
import com.digit.service.PredictionService;

@RestController
public class PredictController {

	private final PredictionService predictionService;

	public PredictController(PredictionService predictionService) {
		this.predictionService = predictionService;
	}

	@PostMapping("/predict")
	public ResponseEntity<PredictionResponse> predictDigit(@RequestParam MultipartFile file) {
		try {
			PredictionResponse result = predictionService.predict(file);
			return ResponseEntity.ok(result);
		} catch (Exception e) {
			throw new FileMissingException("Image file not found try agian with valid image file");
		}
	}
}
