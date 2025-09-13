package com.digit.dtos;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

public class SimpleErrorStructure {
	private int status;
	private String error;
	private String message;

	public SimpleErrorStructure(int status, String error, String message) {
		this.status = status;
		this.error = error;
		this.message = message;
	}

	public int getStatus() {
		return status;
	}

	public void setStatus(int status) {
		this.status = status;
	}

	public String getError() {
		return error;
	}

	public void setError(String error) {
		this.error = error;
	}

	public String getMessage() {
		return message;
	}

	public void setMessage(String message) {
		this.message = message;
	}

	public static ResponseEntity<SimpleErrorStructure> createSimpleErrorStructure(HttpStatus status, String error,
			String message) {
		SimpleErrorStructure response = new SimpleErrorStructure(status.value(), error, message);
		return ResponseEntity.status(status).body(response);
	}
}
