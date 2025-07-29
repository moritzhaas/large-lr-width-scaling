import pickle, os, fnmatch
import numpy as np

def find(pattern, path, nosub = False):
    '''
    Returns list of filenames containing pattern in path.
    '''
    result = []
    if nosub:
        all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for name in all_files:
            if fnmatch.fnmatch(name,pattern):
                result.append(os.path.join(path, name))
    else:
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
    return result


def mysave(path,name,data):
    if os.path.exists(path):
        if os.path.exists(path+name):
            os.remove(path+name)
        with open(path+name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)
    else:
        os.makedirs(path)
        with open(path + name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)

def myload(name, device = None):
    with open(name, "rb") as fp:   # Unpickling
        if device is None:
            all_stats = pickle.load(fp)
        else:
            all_stats = pickle.load(name,map_location=device)
    return all_stats



def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def get_lr(max_lr, iter, num_iter, warmup=0, schedule=None):
    warmup = 0 if warmup is None else warmup
    if iter<warmup*num_iter:
        return max_lr * iter/(num_iter*warmup)
    else:
        if schedule is None:
            return max_lr
        elif schedule == 'cos':
            time_frac = (iter-warmup*num_iter)/((1-warmup)*num_iter)
            return max_lr * 0.5 * (1 + np.cos(np.pi * time_frac))
        
def update(f_t, lambda_t, x, y, eta, param='mup', loss='mse',width=1, c_sp=1, weight_decay = 0):
    if loss == 'mse':
        chi = f_t-y
    else:
        raise NotImplementedError
    try:
        if param == 'mup':
            f_t = f_t * ((1-weight_decay*eta)**2 + eta ** 2 * chi ** 2 * x**2) - eta * (1-weight_decay*eta) * lambda_t * chi
            lambda_t = (1-weight_decay*eta)**2 * lambda_t + x**2 * eta * chi * (eta * chi * lambda_t  - 4 * (1-weight_decay*eta) * f_t)
        elif param == 'ntp': # this implementation directly assumes the correct scaling of weight decay to achieve a width-independent effect
            factor = 1/width
            f_t = f_t * ((1-weight_decay*eta)**2 + eta ** 2 * chi ** 2 * x**2*factor) - eta * (1-weight_decay*eta) * lambda_t * chi
            lambda_t = (1-weight_decay*eta)**2 * lambda_t + x**2 * eta * chi * factor * (eta * chi * lambda_t - 4 * (1-weight_decay*eta) * f_t)
        elif param== 'sp':
            factor = width**(-c_sp)
            f_t = f_t * ((1-weight_decay*eta)**2 + factor**2 * eta ** 2 * chi ** 2 * x**2) - factor*width * eta * (1-weight_decay*eta) * lambda_t * chi
            lambda_t = (1-weight_decay*eta)**2 * lambda_t + x**2 * eta * chi * factor * (eta * chi * lambda_t - 4 * (1-weight_decay*eta) * f_t/width)
    except OverflowError:
        f_t, lambda_t = np.nan, np.nan
    return f_t, lambda_t


def update_uv(u, v, x, y, eta, param='mup', loss='mse',width=1, c_sp=0, weight_decay = 0):
    # x should be one-dimensional!
    f_t = x * v @ u
    if loss == 'mse':
        chi = (f_t.flatten()-y).reshape(-1,1)
    elif loss=='celoss':
        chi = (softmax(f_t).flatten()-y).reshape(-1,1)
    else:
        raise NotImplementedError
    if param == 'mup':
        eta_u = eta * width
        eta_v = eta * width**(-1)
    elif param == 'ntp':
        raise NotImplementedError
    elif param == 'sp':
        eta_u = eta * width**(-c_sp)
        eta_v = eta * width**(-c_sp)
    u = (1-eta*weight_decay)*u - eta_u * x * v.T @ chi # n x 1
    v = (1-eta*weight_decay)*v - eta_v * x * chi @ u.T # c x n
    return u, v


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
