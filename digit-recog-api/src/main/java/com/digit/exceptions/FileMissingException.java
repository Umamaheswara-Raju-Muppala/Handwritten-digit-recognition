package com.digit.exceptions;

@SuppressWarnings("serial")
public class FileMissingException extends RuntimeException {

	public FileMissingException(String message) {
		super(message);
	}
}