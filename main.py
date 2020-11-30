import argparse


def generate_mobilenet_model(width_multiplier=1., input_resolution=224, verbose=False):
    import tensorflow.keras.layers as layers

    avgpool_size = {
        224: (4, 4),
        192: (3, 3),
        160: (3, 3),
        128: (2, 2),
    }

    model = tf.keras.Sequential([
        layers.Input(shape=(input_resolution, input_resolution, 3)),

        layers.Conv2D(filters=int(32 * width_multiplier), kernel_size=(3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(64 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(128 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(128 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(256 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(256 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(512 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # X5
        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(512 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(512 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(512 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(512 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(512 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        # X5 End

        layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(1024 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(filters=int(1024 * width_multiplier), kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.AveragePooling2D(pool_size=avgpool_size[input_resolution]),
        layers.Flatten(),
        layers.Dense(units=1024),
        layers.Dense(units=1000),
        layers.Softmax()
    ])

    model.compile(optimizer='rmsprop')
    if verbose:
        model.summary()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width-multiplier', '-w', type=float, default=1.,
                        help='Width multiplier of MobileNet (could be one of 1.0, 0.75, 0.5, 0.24)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging (Model summary and Tensorflow logics, etc)')
    parser.add_argument('--input-resolution', '-i', type=int, default=224,
                        help='Input resolution (applied with resolution multiplier, one of 224, 192, 160, 128)')
    args = parser.parse_args()

    import os

    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print("Begin TF Library Load")
    import tensorflow as tf
    print("End TF Library Load")

    if args.width_multiplier not in [1., 0.75, 0.5, 0.25]:
        print(f"Invalid argument: args.width_multiplier={args.width_multiplier}")
        exit(-1)

    if args.input_resolution not in [224, 192, 160, 128]:
        print(f"Invalid argument: args.input_resolution={args.input_resolution}")
        exit(-1)

    print("Model Init")
    model = generate_mobilenet_model(args.width_multiplier, args.input_resolution, args.verbose)
    print("Done Model Init")

    model.summary()
