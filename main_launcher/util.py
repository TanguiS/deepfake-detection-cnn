def decode_shape(shape: int, grayscale: bool):
    channels = 3
    if grayscale:
        channels = 1
        raise NotImplementedError(f"Grayscale is not implemented yet.")
    input_shape = (shape, shape, channels)

    return input_shape
