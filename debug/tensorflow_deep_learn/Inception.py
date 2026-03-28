# import numpy as np
# from tensorflow.keras import layers, Model, Input

# inp = Input(shape=(8,1))                      # (timesteps=8, channels=1)
# conv = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inp)
# model = Model(inp, conv)
# model.summary()                               # 查看输出 shape 与参数量

# x = np.ones((4,8,1), dtype=np.float32)        # batch_size=4 的示例输入
# y = model.predict(x)
# print('input shape:', x.shape)
# print('output shape:', y.shape)               # 期望 (4, 8, 32)

import numpy as np

def conv2d_valid(inp, kernel):
    H, W = inp.shape
    kH, kW = kernel.shape
    outH = H - kH + 1
    outW = W - kW + 1
    out = np.zeros((outH, outW), dtype=np.float32)
    for i in range(outH):
        for j in range(outW):
            window = inp[i:i+kH, j:j+kW]
            out[i, j] = np.sum(window * kernel)
    return out

# 示例输入（4x4）
X = np.array([
    [1,2,3,0],
    [0,1,2,1],
    [1,0,1,2],
    [2,1,0,1]
], dtype=np.float32)

# 卷积核（2x2）
K = np.array([
    [1,0],
    [0,1]
], dtype=np.float32)

out = conv2d_valid(X, K)
print("输入 X:\n", X)
print("卷积核 K:\n", K)
print("输出特征图 out:\n", out)
# 显示左上位置计算细节
window00 = X[0:2, 0:2]
print("左上窗口:\n", window00)
print("逐元素相乘并求和:", window00 * K, "=>", np.sum(window00 * K))