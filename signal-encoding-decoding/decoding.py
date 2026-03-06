import numpy as np

def denormalize_signal(normalized_signal, reference_max):
    """
    Convert normalized values back to approximate original scale.
    """
    normalized_signal = np.array(normalized_signal)
    return normalized_signal * reference_max


def decode_message(encoded_values):
    """
    Decode ASCII numeric values back into a string.
    """
    decoded_chars = []

    for value in encoded_values:
        decoded_chars.append(chr(int(round(value))))

    return "".join(decoded_chars)


if __name__ == "__main__":

    normalized_signal = [0.75, 0.71875, 0.84375, 0.84375, 0.875]
    reference_max = 80

    print("Normalized Signal:", normalized_signal)

    denormalized = denormalize_signal(normalized_signal, reference_max)
    print("Denormalized Values:", denormalized)

    decoded = decode_message(denormalized)
    print("Decoded Message:", decoded)
