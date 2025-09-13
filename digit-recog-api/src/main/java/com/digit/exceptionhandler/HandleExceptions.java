package com.digit.exceptionhandler;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MultipartException;

import com.digit.dtos.SimpleErrorStructure;
import com.digit.exceptions.FileMissingException;

import jakarta.servlet.http.HttpServletRequest;

@RestControllerAdvice
public class HandleExceptions {
	@ExceptionHandler(FileMissingException.class)
	public ResponseEntity<SimpleErrorStructure> handleFileMissing(FileMissingException ex, HttpServletRequest request) {

		return SimpleErrorStructure.createSimpleErrorStructure(HttpStatus.BAD_REQUEST, "File Missing", ex.getMessage());
	}

	@ExceptionHandler(FileNotFoundException.class)
	public ResponseEntity<SimpleErrorStructure> handleFileNotFound(FileNotFoundException ex,
			HttpServletRequest request) {
		return SimpleErrorStructure.createSimpleErrorStructure(HttpStatus.NOT_FOUND, "File Not Found", ex.getMessage());
	}

	@ExceptionHandler(IOException.class)
	public ResponseEntity<SimpleErrorStructure> handleIOException(IOException ex, HttpServletRequest request) {

		return SimpleErrorStructure.createSimpleErrorStructure(HttpStatus.BAD_REQUEST, "Invalid Image",
				"Failed to read the uploaded image");
	}

	@ExceptionHandler(MultipartException.class)
	public ResponseEntity<SimpleErrorStructure> handleMultipartException(MultipartException ex,
			HttpServletRequest request) {
		return SimpleErrorStructure.createSimpleErrorStructure(HttpStatus.BAD_REQUEST, "Multipart Error",
				"File upload failed: " + ex.getMessage());
	}

	@ExceptionHandler(IllegalArgumentException.class)
	public ResponseEntity<SimpleErrorStructure> handleIllegalArgument(IllegalArgumentException ex,
			HttpServletRequest request) {
		return SimpleErrorStructure.createSimpleErrorStructure(HttpStatus.BAD_REQUEST, "Invalid Argument",
				ex.getMessage());
	}

	@ExceptionHandler(Exception.class)
	public ResponseEntity<SimpleErrorStructure> handleGeneralException(Exception ex, HttpServletRequest request) {
		return SimpleErrorStructure.createSimpleErrorStructure(HttpStatus.INTERNAL_SERVER_ERROR,
				"Internal Server Error", ex.getMessage());
	}
}
