import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def vis_sparse_flow(flow, X=None, Y=None, path="flow.png"):
    flow = flow.copy()
    flow[:, :, 0] = -flow[:, :, 0]
    if X is None:
        height, width, _ = flow.shape
        xx = np.arange(0, height)
        yy = np.arange(0, width)
        X, Y = np.meshgrid(yy, xx)
        X = X.flatten()
        Y = Y.flatten()

        # sample
        sample_x = flow[:, :, 0]
        sample_y = flow[:, :, 1]
        sample_x = sample_x[:, :, np.newaxis]
        sample_y = sample_y[:, :, np.newaxis]

        new_flow = np.concatenate([sample_x, sample_y], axis=2)
    flow_x = new_flow[:, :, 0].flatten()
    flow_y = new_flow[:, :, 1].flatten()

    # display
    plt.cla()  # Clean up the canvas left in the previous step, or it, otherwise overlapping areas will appear.
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.axis('off')
    ax.quiver(X, Y, flow_x, flow_y, color="#666666")
    ax.grid()
    # ax.legend()
    plt.draw()
    plt.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()


def gen_flow_circle(center, height, width):
    x0, y0 = center
    if x0 >= height or y0 >= width:
        raise AttributeError('ERROR')
    flow = np.zeros((height, width, 2), dtype=np.float32)

    grid_x = np.tile(np.expand_dims(np.arange(width), 0), [height, 1])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])

    grid_x0 = np.tile(np.array([x0]), [height, width])
    grid_y0 = np.tile(np.array([y0]), [height, width])

    flow[:, :, 0] = grid_x0 - grid_x
    flow[:, :, 1] = grid_y0 - grid_y

    return flow


if __name__ == '__main__':
    center = [5, 5]
    flow = gen_flow_circle(center, height=11, width=11)
    flow = flow / 2
    vis_sparse_flow(flow)
