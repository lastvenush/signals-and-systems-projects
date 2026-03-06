import numpy as np

def encode_message(message):
    """
    Encode a string message into ASCII numeric values.
    """
    encoded = []

    for char in message:
        ascii_value = ord(char)
        encoded.append(ascii_value)

    return encoded


def normalize_signal(signal):
    """
    Normalize the encoded signal between -1 and 1.
    """
    signal = np.array(signal)
    max_val = np.max(np.abs(signal))

    if max_val == 0:
        return signal

    return signal / max_val


if __name__ == "__main__":

    message = "HELLO"

    print("Original Message:", message)

    encoded = encode_message(message)
    print("Encoded ASCII:", encoded)

    normalized = normalize_signal(encoded)
    print("Normalized Signal:", normalized)
