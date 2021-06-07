import common


def darknet53(inputs):
    inputs = common.convolution(inputs, (3, 3, 3, 32))
    inputs = common.convolution(inputs, (3, 3, 32, 64), downsample=True)

    for i in range(1):
        inputs = common.residual_block(inputs, 64, 32, 64)

    inputs = common.convolution(inputs, (3, 3, 64, 128), downsample=True)

    for i in range(2):
        inputs = common.residual_block(inputs, 128, 64, 128)

    inputs = common.convolution(inputs, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        inputs = common.residual_block(inputs, 256, 128, 256)

    route_1 = inputs
    inputs = common.convolution(inputs, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        inputs = common.residual_block(inputs, 512, 256, 512)

    route_2 = inputs
    inputs = common.convolution(inputs, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        inputs = common.residual_block(inputs, 1024, 512, 1024)

    return route_1, route_2, inputs
