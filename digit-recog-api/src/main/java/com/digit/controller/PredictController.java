package com.digit.controller;

import com.digit.dtos.PredictionResponse;
import com.digit.exceptions.FileMissingException;
import com.digit.service.PredictionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

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
