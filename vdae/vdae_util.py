import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import ot
import annoy
import scipy

# for each point in x, determine neighborhood in x, and then compute upper and lower constants
def bilip(x, y, k, eps=1e-7):
    # find nearest neighbors
    a = annoy.AnnoyIndex(x.shape[1], 'euclidean')
    for i, x_ in enumerate(x):
        a.add_item(i, x_)

    # build nn tree
    a.build(-1)

    # compute K
    maxs = []
    for i in range(len(x)):
        x_neighbs, dists = a.get_nns_by_item(i, k+1, include_distances=True)
        x_neighbs, dists = x_neighbs[1:], dists[1:]
        ratio1 = [np.linalg.norm(x[j] - x[i])/np.linalg.norm(y[j] - y[i]) for j in x_neighbs]
        ratio2 = [np.linalg.norm(y[j] - y[i])/np.linalg.norm(x[j] - x[i]) for j in x_neighbs]
        max1, max2 = np.max(ratio1), np.max(ratio2)
        max_ = max(max1, max2) + eps
        assert np.all(ratio1 <= max_)
        assert np.all(ratio2 >= 1/max_)
        maxs.append(max_)

    return maxs

def predict_with_K_fn(K_fn, x, bs=1000):
    '''
    Convenience function: evaluates x by K_fn(x), where K_fn is
    a Keras function, by batches of size 1000.
    '''
    if not isinstance(x, list):
        x = [x]
    num_outs = len(K_fn.outputs)
    shapes = [list(output_.get_shape()) for output_ in K_fn.outputs]
    shapes = [[len(x[0])] + s[1:] for s in shapes]
    y = [np.empty(s) for s in shapes]
    recon_means = []
    for i in range(int((x[0].shape[0]-1)/bs + 1)):
        x_batch = []
        for x_ in x:
            x_batch.append(x_[i*bs:(i+1)*bs])
        temp = K_fn(x_batch)
        for j in range(num_outs):
            y[j][i*bs:(i+1)*bs] = temp[j]

    return y

def imscatter(x, y, samples, shape, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    m=samples.shape[0]
    i=0
    for x0, y0  in zip(x, y):
        if len(shape) == 3:
            im = OffsetImage(samples[i].reshape(shape), zoom=zoom)
        else:
            im = OffsetImage(samples[i].reshape(shape), zoom=zoom, cmap=plt.get_cmap('gray'))
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
        i=i+1
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    ax.axis('off')
    return artists

def plot(x, y=None, x2=None, y2=None, s=10, s2=None, alpha=0.5, alpha2=None, label1=None, label2=None, cmap1=None, cmap2=None, n_imgs=4, shuffle=True):
    s2 = s if s2 is None else s2
    alpha2 = alpha if alpha2 is None else alpha2

    if x.shape[1:] == (28, 28, 1) or x.shape[1:] == (32, 32, 3) or x.shape[1:] == (32, 32, 1) or x.shape[1:] == (128, 128, 1):
        x = x.reshape(len(x), -1)

    n = x.shape[1]

    if n == 1:
        g = plt.figure()
        plt.scatter(np.zeros((n,)), x[:,1], c=y, s=s, alpha=alpha, label=label1, cmap=cmap1)
        if x2 is not None:
            plt.scatter(np.zeros((n,)), x2[:,1], c=y2, s=s2, alpha=alpha2, label=label2, cmap=cmap2)
    if n == 3:
        g = plt.figure()
        ax = g.add_subplot(111, projection='3d')
        ax.set_axis_off()
        ax.scatter(x[:,0], x[:,1], x[:,2], c=y, s=s, alpha=alpha, label=label1)
        if x2 is not None:
            ax.scatter(x2[:,0], x2[:,1], x2[:,2], c=y2, s=s2, alpha=alpha2, label=label2)
        g = (g, ax)
    elif n == 784 or n == 3072 or n == 24000 or n == 1024 or n == 16384:
        n_imgs = min(n_imgs, len(x))
        if n == 784:
            img_shape, wdist = (28, 28), -.7
        elif n == 1024:
            img_shape, wdist = (32, 32), -.7
        elif n == 3072:
            img_shape, wdist = (32, 32, 3), -.7
        elif n == 24000:
            img_shape, wdist = (100, 80, 3), -.785
        elif n == 16384:
            img_shape, wdist = (128, 128), -.685

        if x2 is None:
            g, axarr = plt.subplots(n_imgs, n_imgs)
            if shuffle:
                p = np.random.permutation(len(x))
            else:
                p = np.arange(len(x))
            n = 0
            while n < n_imgs * n_imgs:
                i, j = int(n / n_imgs), n % n_imgs
                axarr[i,j].axis('off')
                if n < len(x):
                    axarr[i,j].imshow(x[p[n]].reshape(img_shape))
                n += 1
        elif x2 is not None:
            p = np.random.permutation(len(x))[:n_imgs]
            for i in range(n_imgs):
                idx = p[i]
                plt.subplot(1,2,1)
                plt.imshow(x2[idx].reshape(img_shape))
                plt.subplot(1,2,2)
                plt.imshow(x[idx].reshape(img_shape))
                g = plt.figure()
        g.subplots_adjust(wspace=wdist, hspace=0)
    else:
        g = plt.figure()
        plt.scatter(x[:,0], x[:,1], c=y, s=s, alpha=alpha, label=label1, cmap=cmap1)
        if x2 is not None:
            plt.scatter(x2[:,0], x2[:,1], c=y2, s=s2, alpha=alpha2, label=label2, cmap=cmap2)

    if label1 is not None or label2 is not None:
        plt.legend()

    return g

def generate_torus(n=1000, train_set_fraction=.8):
    n_ = int(np.sqrt(n) + 1)
    theta = np.linspace(0, 2.*np.pi, num=n_)
    phi = np.linspace(0, 2.*np.pi, num=n_)
    theta, phi = np.meshgrid(theta, phi)
    c, a = 2, 1
    x = np.empty((n_ ** 2, 3))
    print(x.shape)
    print(np.cos(theta).shape, np.cos(phi).shape, theta.shape, phi.shape)
    x[:,0] = ((c + a*np.cos(theta)) * np.cos(phi)).reshape(-1)
    x[:,1] = ((c + a*np.cos(theta)) * np.sin(phi)).reshape(-1)
    x[:,2] = (a * np.sin(theta)).reshape(-1)

    # y is just theta
    y = theta.reshape(-1) + phi.reshape(-1)

    return shuffle_and_return_n(x, y, train_set_fraction)
def generate_bunny(n=2000, train_set_fraction=.8):
    import bunny
    a = [np.expand_dims(np.array(bunny.trace2[c]), axis=-1) for c in ['x', 'y', 'z']]
    x = np.concatenate(a, axis=-1)
    x = x.astype(np.float32)
    x = x[np.logical_not(np.any(np.isnan(x), axis=1))]
    y = np.arange(len(x))

    return shuffle_and_return_n(x, y, train_set_fraction, n=n)

def generate_sphere(n=1200, train_set_fraction=.8):
    r = 1
    alpha = 4.0*np.pi*r*r/(n+1)
    d = np.sqrt(alpha)
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alpha/d_nu
    count = 0
    coords = [[], [], []]
    y = []
    for i in range(0, m_nu):
        nu = np.pi*(i+0.5)/m_nu
        m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
        for j in range(0, m_phi):
            phi = 2*np.pi*j/m_phi
            xp = r*np.sin(nu)*np.cos(phi)
            yp = r*np.sin(nu)*np.sin(phi)
            zp = r*np.cos(nu)
            coords[0].append(xp)
            coords[1].append(yp)
            coords[2].append(zp)
            y.append(i + j)
            count = count +1

    x = np.array(coords).T
    y = np.array(y).T

    return shuffle_and_return_n(x, y, train_set_fraction, n=n)

def generate_plane(n=1200, train_set_fraction=.8):
    # compute number of points in each dimension
    n_i = np.int(np.sqrt(n))
    n = n_i ** 2

    # compute points on this grid
    t = np.mgrid[0:1:1/n_i, 0:1:1/n_i].reshape(2,-1).T
    t = np.concatenate([t, np.zeros(shape=(len(t),1))], axis=1)

    # compute rotation
    A = np.random.normal(size=(3, 3))
    A, _ = np.linalg.qr(A)

    x = np.dot(A, t.T).T

    # y is the sum of the ts
    y = t[:,0] + t[:,1]

    return shuffle_and_return_n(x, y, train_set_fraction)

def generate_loop(n=1200, train_set_fraction=.8):
    t = np.linspace(0, 2*np.pi, num=n+1)[:-1]

    # generate all three coordinates
    x = np.empty((n, 3))
    x[:,0] = np.cos(t)
    x[:,1] = np.sin(2*t)
    x[:,2] = np.sin(3*t)

    # y is just t
    y = t

    return shuffle_and_return_n(x, y, train_set_fraction)

def generate_circle(n=1000, train_set_fraction=.8, alpha=4):
    t = np.linspace(0, 2*np.pi, num=n+1)[:-1]
#     t = np.log(np.linspace(1, alpha, num=n))
    t = t / np.max(t) * 2 * np.pi

    # generate all three coordinates
    x = np.empty((n, 2))
    x[:,0] = np.cos(t)
    x[:,1] = np.sin(t)

    # y is just t
    y = t

    return shuffle_and_return_n(x, y, train_set_fraction)

def generate_line(n=1200, train_set_fraction=.8):
    pts_per_cluster = int(n / 2)
    x1 = np.linspace(0, 1, num=n).reshape((-1, 1))
    x2 = np.linspace(0, 1, num=n).reshape((-1, 1))
    x = np.concatenate([x1, x2], axis=1)

    # generate labels
#     y = np.concatenate((np.zeros(shape=(pts_per_cluster, 1)), np.ones(shape=(pts_per_cluster, 1))), axis=0)
    y = x1

    return shuffle_and_return_n(x, y, train_set_fraction)

def generate_gaussians(n=1200, n_clusters=2, noise_sigma=0.1, train_set_fraction=1.):
    '''
    Generates and returns the nested 'C' example dataset (as seen in the leftmost
    graph in Fig. 1)
    '''
    pts_per_cluster = int(n / n_clusters)
    r = 1

    clusters = []

    for x in np.linspace(0, 1, num=n_clusters):
        clusters.append(np.random.normal(x, noise_sigma, size=(pts_per_cluster, 2)))

    # combine clusters
    x = np.concatenate(clusters, axis=0)
    print(np.max(x), np.min(x))
    x /= (np.max(x) - np.min(x))
    print(np.max(x), np.min(x))
    x -= np.min(x)
    print(np.max(x), np.min(x))

    # generate labels
    y = np.concatenate((np.zeros(shape=(pts_per_cluster, 1)), np.ones(shape=(pts_per_cluster, 1))), axis=0)

    return shuffle_and_return_n(x, y, train_set_fraction)

def generate_cc(n=1200, noise_sigma=0.1, train_set_fraction=1.):
    '''
    Generates and returns the nested 'C' example dataset (as seen in the leftmost
    graph in Fig. 1)
    '''
    pts_per_cluster = int(n / 2)
    r = 1

    # generate clusters
    theta1 = (np.random.uniform(0, 1, pts_per_cluster) * r * np.pi - np.pi / 2).reshape(pts_per_cluster, 1)
    theta2 = (np.random.uniform(0, 1, pts_per_cluster) * r * np.pi - np.pi / 2).reshape(pts_per_cluster, 1)

    cluster1 = np.concatenate((np.cos(theta1) * r, np.sin(theta1) * r), axis=1)
    cluster2 = np.concatenate((np.cos(theta2) * r, np.sin(theta2) * r), axis=1)

    # shift and reverse cluster 2
    cluster2[:, 0] = -cluster2[:, 0] + 0.5
    cluster2[:, 1] = -cluster2[:, 1] - 1

    # combine clusters
    x = np.concatenate((cluster1, cluster2), axis=0)

    # add noise to x
    x = x + np.random.randn(x.shape[0], 2) * noise_sigma
    print(np.max(x), np.min(x))
    x /= (np.max(x) - np.min(x))
    print(np.max(x), np.min(x))
    x -= np.min(x)
    print(np.max(x), np.min(x))

    # generate labels
    y = np.concatenate((np.zeros(shape=(pts_per_cluster, 1)), np.ones(shape=(pts_per_cluster, 1))), axis=0)

    return shuffle_and_return_n(x, y, train_set_fraction)

def generate_faces(train_set_fraction=.8):
    x = np.array(loadmat('frey_rawface.mat')['ff']).T
    y = np.arange(len(x)) / len(x)
    x = (x - np.min(x))/(np.max(x) - np.min(x))

    return shuffle_and_return_n(x, y, train_set_fraction)

def generate_bulldog(train_set_fraction=.8):
    x = np.load('bulldog.npy').T
    y = np.arange(len(x))
    print(x.shape)

    x = x / np.max(x)

    return shuffle_and_return_n(x, y, train_set_fraction)

def generate_gaussian_grid(train_set_fraction=.8):
    dim = 2
    n_per_gaussian = 300
    scale = .1
    xs = []
    ys = []
    for i in range(5):
        for j in range(5):
            loc = i - 2, j - 2
            xs.append(np.random.normal(loc=loc, scale=scale, size=(n_per_gaussian, dim)))
            ys.append([(i * 5) + j] * n_per_gaussian)
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    return shuffle_and_return_n(x, y, train_set_fraction)

def shuffle_and_return_n(x, y, train_set_fraction, n=None):
    if n is None:
        n = len(x)
    # shuffle
    p = np.random.permutation(len(x))[:n]
    y = y[p]
    x = x[p]

    # make train and test splits
    n_train = int(n * train_set_fraction)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train].flatten(), y[n_train:].flatten()

    return (x_train, y_train), (x_test, y_test)

def gromorov_wasserstein_d(x_in,x_out):
    C1 = scipy.spatial.distance.cdist(x_in,x_in)
    C2 = scipy.spatial.distance.cdist(x_out,x_out)
    n_samples=x_in.shape[0]
    C1 /= C1.max()
    C2 /= C2.max()
    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    gw0, log0 = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, 'square_loss', verbose=False, log=True)
    return log0['gw_dist']
