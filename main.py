from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy import ndimage
from numpy import any as np_any
from numpy import dot, inf, isinf, isnan, isreal, real, sqrt

#########################LEGAL DISCLAIMER########################################
#The implementation in this repository is based on the original public source:

#- **Original repository:** `hantek/zlinnet`
#- **Source URL:** `https://github.com/hantek/zlinnet/tree/master`

#The original code is licensed under the BSD-3-Clause license:

#Copyright (c) 2015, Zhouhan LIN
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this
#list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice,
#this list of conditions and the following disclaimer in the documentation
#and/or other materials provided with the distribution.

#Neither the name of ZLINNet nor the names of its
#contributors may be used to endorse or promote products derived from
#this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#POSSIBILITY OF SUCH DAMAGE.


#This adaptation **does not claim ownership of the original code**. Any redistributed or derivative work must continue to comply with the above license conditions.

#######################################################################################################################################

def minimize(
#############################################################################
#This function contains a Python version of Carl Rasmussen's Matlab-function 
#minimize.m
#
#minimize.m is copyright (C) 1999 - 2006, Carl Edward Rasmussen.
#Python adaptation by Roland Memisevic 2008.
#
#
#The following is the original copyright notice that comes with the 
#function minimize.m
#(from http://www.kyb.tuebingen.mpg.de/bs/people/carl/code/minimize/Copyright):
#
#
#"(C) Copyright 1999 - 2006, Carl Edward Rasmussen
#############################################################################
    X: np.ndarray,
    f: Callable[..., float],
    grad: Callable[..., np.ndarray],
    args: tuple,
    maxnumlinesearch: Optional[int] = None,
    maxnumfuneval: Optional[int] = None,
    red: float = 1.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, list, int]:
    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 10
    SIG = 0.1
    RHO = SIG / 2
    SMALL = 10.0**-16

    if maxnumlinesearch is None:
        if maxnumfuneval is None:
            raise ValueError("Specify maxnumlinesearch or maxnumfuneval")
        S = "Function evaluation"
        length = int(maxnumfuneval)
    else:
        if maxnumfuneval is not None:
            raise ValueError("Specify either maxnumlinesearch or maxnumfuneval (not both)")
        S = "Linesearch"
        length = int(maxnumlinesearch)

    i = 0
    ls_failed = 0
    f0 = f(X, *args)
    df0 = grad(X, *args)
    fX = [f0]

    i = i + (length < 0)
    s = -df0
    d0 = -dot(s, s)
    x3 = red / (1.0 - d0)

    while i < abs(length):
        i = i + (length > 0)

        X0 = X
        F0 = f0
        dF0 = df0

        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)

        while True:
            x2 = 0.0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = 0

            while (not success) and (M > 0):
                try:
                    M = M - 1
                    i = i + (length < 0)

                    f3 = f(X + x3 * s, *args)
                    df3 = grad(X + x3 * s, *args)

                    if isnan(f3) or isinf(f3) or np_any(isnan(df3) + isinf(df3)):
                        return X, fX, i

                    success = 1
                except Exception:
                    x3 = (x2 + x3) / 2.0

            if f3 < F0:
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3

            d3 = dot(df3, s)

            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:
                break

            x1 = x2
            f1 = f2
            d1 = d2

            x2 = x3
            f2 = f3
            d2 = d3

            A = 6.0 * (f1 - f2) + 3.0 * (d2 + d1) * (x2 - x1)
            B = 3.0 * (f2 - f1) - (2.0 * d1 + d2) * (x2 - x1)
            Z = B + sqrt(complex(B * B - A * d1 * (x2 - x1)))

            if Z != 0.0:
                x3 = x1 - d1 * (x2 - x1) ** 2 / Z
            else:
                x3 = inf

            if (not isreal(x3)) or isnan(x3) or isinf(x3) or (x3 < 0):
                x3 = x2 * EXT
            elif x3 > x2 * EXT:
                x3 = x2 * EXT
            elif x3 < x2 + INT * (x2 - x1):
                x3 = x2 + INT * (x2 - x1)

            x3 = real(x3)

        while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:
            if (d3 > 0) or (f3 > f0 + x3 * RHO * d0):
                x4 = x3
                f4 = f3
                d4 = d3
            else:
                x2 = x3
                f2 = f3
                d2 = d3

            if f4 > f0:
                x3 = x2 - (0.5 * d2 * (x4 - x2) ** 2) / (f4 - f2 - d2 * (x4 - x2))
            else:
                A = 6.0 * (f2 - f4) / (x4 - x2) + 3.0 * (d4 + d2)
                B = 3.0 * (f4 - f2) - (2.0 * d2 + d4) * (x4 - x2)
                if A != 0:
                    x3 = x2 + (sqrt(B * B - A * d2 * (x4 - x2) ** 2) - B) / A
                else:
                    x3 = inf

            if isnan(x3) or isinf(x3):
                x3 = (x2 + x4) / 2.0

            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))

            f3 = f(X + x3 * s, *args)
            df3 = grad(X + x3 * s, *args)

            if f3 < F0:
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3

            M = M - 1
            i = i + (length < 0)

            d3 = dot(df3, s)

        if abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:
            X = X + x3 * s
            f0 = f3
            fX.append(f0)

            if verbose:
                print(f"{S} {i:6d};  Value {f0:4.6e}\r", end="")

            s = (dot(df3, df3) - dot(df0, df3)) / dot(df0, df0) * s - df3
            df0 = df3
            d3 = d0
            d0 = dot(df0, s)

            if d0 > 0:
                s = -df0
                d0 = -dot(s, s)

            x3 = x3 * min(RATIO, d3 / (d0 - SMALL))
            ls_failed = 0
        else:
            X = X0
            f0 = F0
            df0 = dF0

            if ls_failed or (i > abs(length)):
                break

            s = -df0
            d0 = -dot(s, s)
            x3 = 1.0 / (1.0 - d0)
            ls_failed = 1

    if verbose:
        print("\n")
    return X, fX, i


pca_retain = 800
hid_layer_sizes = [4000, 1000, 4000] # for deep architecture [4000, 1000, 4000, 1000, 4000, 1000, 4000]
batchsize = 100
zae_threshold = 1.0

momentum = 0.9
pretrain_lr_zae = 1e-3
pretrain_lr_lin = 1e-4
weightdecay = 1.0 # set to 0.001 for deep architecture
pretrain_epc = 800 # set to 600 for deep architecture

logreg_epc = 1000

finetune_lr = 5e-3
finetune_epc = 1000


def _print_hparams() -> None:
    print(" ")
    print("pca_retain =", pca_retain)
    print("hid_layer_sizes =", hid_layer_sizes)
    print("batchsize =", batchsize)
    print("zae_threshold =", zae_threshold)
    print("momentum =", momentum)
    print(f"pretrain, zae:       lr = {pretrain_lr_zae:.6f}, epc = {pretrain_epc:d}")
    print(f"pretrain, lin:       lr = {pretrain_lr_lin:.6f}, epc = {pretrain_epc:d}, wd = {weightdecay:.3f}")
    print(f"logistic regression: epc = {logreg_epc:d}")
    print(f"finetune:            lr = {finetune_lr:.6f}, epc = {finetune_epc:d}")


def subtract_mean_and_normalize_h_numpy(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(x), axis=1, keepdims=True) + eps)
    return (x / rms).astype(np.float32, copy=False)


class PCAWhiten:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.whiten_scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, retain: int, whiten: bool = True, eps: float = 1e-5) -> "PCAWhiten":
        x = x.astype(np.float32, copy=False)
        self.mean_ = x.mean(axis=0, keepdims=True)
        xc = x - self.mean_
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        k = int(retain)
        self.components_ = vt[:k].T
        if whiten:
            n = x.shape[0]
            eig_sqrt = s[:k] / np.sqrt(max(n - 1, 1))
            self.whiten_scale_ = 1.0 / (eig_sqrt + eps)
        else:
            self.whiten_scale_ = None
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAWhiten.transform called before fit().")
        x = x.astype(np.float32, copy=False)
        xc = x - self.mean_
        z = xc @ self.components_
        if self.whiten_scale_ is not None:
            z = z * self.whiten_scale_[None, :]
        return z.astype(np.float32, copy=False)


class SubtractMeanAndNormalizeH(tf.keras.layers.Layer):
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        m = tf.reduce_mean(x, axis=1, keepdims=True)
        xc = x - m
        rms = tf.sqrt(tf.reduce_mean(tf.square(xc), axis=1, keepdims=True) + self.eps)
        return xc / rms


class TiedAutoencoder(tf.keras.Model):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        kind: str,
        threshold: float,
        init_w: np.ndarray,
        input_drop: float,
        hidden_drop: float,
        name: str,
    ) -> None:
        super().__init__(name=name)

        if kind not in ("zae", "lin"):
            raise ValueError("kind must be 'zae' or 'lin'")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.kind = kind
        self.threshold = float(threshold)
        self.input_drop = float(input_drop)
        self.hidden_drop = float(hidden_drop)

        init_w = init_w.astype(np.float32, copy=False)
        if init_w.shape != (self.input_dim, self.hidden_dim):
            raise ValueError(f"init_w has shape {init_w.shape}, expected {(self.input_dim, self.hidden_dim)}")

        self.W = self.add_weight(
            name="W",
            shape=(self.input_dim, self.hidden_dim),
            initializer=tf.constant_initializer(init_w),
            trainable=True,
        )
        self.b_dec = self.add_weight(
            name="b_dec",
            shape=(self.input_dim,),
            initializer="zeros",
            trainable=True,
        )

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        z = tf.linalg.matmul(x, self.W)
        if self.kind == "zae":
            return tf.nn.relu(z - self.threshold)
        return z

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if training and self.input_drop > 0.0:
            x = tf.nn.dropout(x, rate=self.input_drop)
        h = self.encode(x)
        if training and self.hidden_drop > 0.0:
            h = tf.nn.dropout(h, rate=self.hidden_drop)
        x_hat = tf.linalg.matmul(h, tf.transpose(self.W)) + self.b_dec
        return x_hat


def _init_w_first_layer(train_x: np.ndarray, hidden_dim: int) -> np.ndarray:
    n, d = train_x.shape
    scale = 0.01
    if hidden_dim <= n:
        return (scale * train_x[:hidden_dim, :].T).astype(np.float32, copy=False)
    return _init_w_by_tiling_data(train_x, in_dim=d, out_dim=hidden_dim)


def _init_w_by_tiling_data(train_x: np.ndarray, in_dim: int, out_dim: int) -> np.ndarray:
    scale = 0.01
    feature_num = int(train_x.shape[0] * train_x.shape[1])
    needed = int(in_dim * out_dim)
    reps = needed // feature_num + 1
    flat = np.tile(scale * train_x, (reps, 1)).ravel()
    return flat[:needed].reshape(in_dim, out_dim).astype(np.float32, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z, dtype=np.float64)
    return (ez / ez.sum(axis=1, keepdims=True)).astype(np.float64, copy=False)


def logreg_cost(theta: np.ndarray, x: np.ndarray, y: np.ndarray, wd: float) -> float:
    d = x.shape[1]
    k = int(np.max(y) + 1)
    w = theta[: d * k].reshape(d, k)
    b = theta[d * k :].reshape(k)
    logits = x @ w + b[None, :]
    p = _softmax(logits)
    n = x.shape[0]
    nll = -np.log(p[np.arange(n), y] + 1e-12).mean()
    reg = 0.5 * wd * float(np.sum(w * w))
    return float(nll + reg)


def logreg_grad(theta: np.ndarray, x: np.ndarray, y: np.ndarray, wd: float) -> np.ndarray:
    d = x.shape[1]
    k = int(np.max(y) + 1)
    w = theta[: d * k].reshape(d, k)
    b = theta[d * k :].reshape(k)
    logits = x @ w + b[None, :]
    p = _softmax(logits)
    n = x.shape[0]
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(n), y] = 1.0
    dlogits = (p - y_onehot) / float(n)
    gw = x.T @ dlogits + wd * w
    gb = dlogits.sum(axis=0)
    return np.concatenate([gw.ravel(), gb.ravel()]).astype(np.float64, copy=False)


def _augment_norm_images(batch_norm_img: np.ndarray, variant: str, rng: np.random.RandomState) -> np.ndarray:
    if variant == "orig":
        return batch_norm_img
    if variant == "flip":
        return batch_norm_img[:, :, ::-1, :].astype(np.float32, copy=False)
    if variant == "rot":
        out = np.empty_like(batch_norm_img, dtype=np.float32)
        for i in range(batch_norm_img.shape[0]):
            angle = (rng.vonmises(0.0, 1.0) / (4.0 * np.pi)) * 180.0
            out[i] = ndimage.rotate(
                batch_norm_img[i],
                angle,
                axes=(0, 1),
                reshape=False,
                mode="wrap",
                order=1,
            ).astype(np.float32, copy=False)
        return out
    if variant == "shift":
        out = np.empty_like(batch_norm_img, dtype=np.float32)
        for i in range(batch_norm_img.shape[0]):
            dy = int(rng.randint(-4, 4))
            dx = int(rng.randint(-4, 4))
            out[i] = ndimage.shift(
                batch_norm_img[i],
                shift=(dy, dx, 0),
                mode="reflect",
                order=1,
            ).astype(np.float32, copy=False)
        return out
    raise ValueError("variant must be one of: orig, flip, rot, shift")


def _pca_from_raw_images_batch(
    batch_img: np.ndarray,
    pca: PCAWhiten,
    *,
    apply_norm_before_pca: bool,
) -> np.ndarray:
    flat = batch_img.reshape(batch_img.shape[0], -1).astype(np.float32, copy=False)
    if apply_norm_before_pca:
        flat = subtract_mean_and_normalize_h_numpy(flat)
    return pca.transform(flat)


def _forward_to_layer_input(
    x_pca: np.ndarray,
    pretrained_W: list[np.ndarray],
    layer_kinds: list[str],
    thresholds: list[float],
    upto_layer: int,
) -> np.ndarray:
    x = x_pca
    for j in range(upto_layer):
        x = x @ pretrained_W[j]
        if layer_kinds[j] == "zae":
            thr = float(thresholds[j])
            x = np.maximum(0.0, x - thr)
        if j < len(layer_kinds) - 1:
            x = subtract_mean_and_normalize_h_numpy(x)
    return x.astype(np.float32, copy=False)


def _pretrain_one_layer_with_aug(
    x_train_pca: np.ndarray,
    x_train_img: np.ndarray,
    pca: PCAWhiten,
    pretrained_W: list[np.ndarray],
    layer_kinds: list[str],
    thresholds: list[float],
    *,
    li: int,
    input_dim: int,
    hidden_dim: int,
    kind: str,
    init_w: np.ndarray,
    lr: float,
    wd: Optional[float],
    momentum: float,
    batchsize: int,
    epochs: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    ae = TiedAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kind=kind,
        threshold=zae_threshold if kind == "zae" else 0.0,
        init_w=init_w,
        input_drop=0.2,
        hidden_drop=0.5,
        name=f"ae_{li}",
    )

    opt = tf.keras.optimizers.SGD(learning_rate=float(lr), momentum=float(momentum))

    @tf.function
    def train_step(xb: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            x_hat = ae(xb, training=True)
            loss = tf.reduce_mean(tf.square(xb - x_hat))
            if wd is not None and wd > 0.0:
                loss = loss + float(wd) * tf.nn.l2_loss(ae.W)
        grads = tape.gradient(loss, ae.trainable_variables)
        opt.apply_gradients(zip(grads, ae.trainable_variables))
        return loss

    n = x_train_pca.shape[0]
    prev_cost = np.inf
    patience = 0

    for _ in range(int(epochs)):
        losses = []

        for variant in ("orig", "flip", "rot", "shift"):
            idx = rng.permutation(n)
            for s in range(0, n, int(batchsize)):
                bidx = idx[s : s + int(batchsize)]
                if variant == "orig":
                    xb_pca = x_train_pca[bidx]
                else:
                    batch_raw = x_train_img[bidx]
                    batch_pca_input = _pca_from_raw_images_batch(
                        batch_raw,
                        pca,
                        apply_norm_before_pca=True,
                    )
                    batch_norm_flat = subtract_mean_and_normalize_h_numpy(
                        batch_raw.reshape(batch_raw.shape[0], -1).astype(np.float32, copy=False)
                    )
                    batch_norm_img = batch_norm_flat.reshape(batch_raw.shape[0], 32, 32, 3)
                    aug_norm_img = _augment_norm_images(batch_norm_img, variant, rng)
                    aug_flat = aug_norm_img.reshape(aug_norm_img.shape[0], -1).astype(np.float32, copy=False)
                    xb_pca = pca.transform(aug_flat)

                if li > 0:
                    xb_in = _forward_to_layer_input(
                        xb_pca,
                        pretrained_W=pretrained_W,
                        layer_kinds=layer_kinds,
                        thresholds=thresholds,
                        upto_layer=li,
                    )
                else:
                    xb_in = xb_pca.astype(np.float32, copy=False)

                l = train_step(tf.convert_to_tensor(xb_in, dtype=tf.float32))
                losses.append(float(l.numpy()))

        cost = float(np.mean(losses)) if losses else float("inf")

        if prev_cost <= cost:
            patience += 1
            if patience > 10:
                patience = 0
                opt.learning_rate.assign(float(opt.learning_rate.numpy()) * 0.9)
            if float(opt.learning_rate.numpy()) < 1e-10:
                break
        prev_cost = cost

    return ae.W.numpy().astype(np.float32, copy=False)


def _build_encoder_sequential(input_dim: int, hid_sizes: list[int], kinds: list[str], eps: float = 1e-8) -> tf.keras.Sequential:
    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.InputLayer(input_shape=(int(input_dim),), dtype=tf.float32)]
    for i, (h, k) in enumerate(zip(hid_sizes, kinds)):
        layers.append(tf.keras.layers.Dense(int(h), use_bias=False, name=f"enc_{i}"))
        if k == "zae":
            layers.append(tf.keras.layers.ReLU(name=f"relu_{i}"))
        if i < len(hid_sizes) - 1:
            layers.append(SubtractMeanAndNormalizeH(eps=eps, name=f"norm_{i}"))
    return tf.keras.Sequential(layers, name="encoder")


def _set_encoder_weights(encoder: tf.keras.Sequential, pretrained_W: list[np.ndarray]) -> None:
    for i, w in enumerate(pretrained_W):
        layer = encoder.get_layer(f"enc_{i}")
        layer.set_weights([w.astype(np.float32, copy=False)])


def _error_rate(model: tf.keras.Model, x_np: np.ndarray, y_np: np.ndarray, batchsize: int) -> float:
    ds = tf.data.Dataset.from_tensor_slices((x_np.astype(np.float32, copy=False), y_np.astype(np.int64, copy=False))).batch(
        int(batchsize)
    )
    wrong = 0
    total = 0
    for xb, yb in ds:
        pred = tf.argmax(model(xb, training=False), axis=1, output_type=tf.int64)
        wrong += int(tf.reduce_sum(tf.cast(pred != yb, tf.int32)).numpy())
        total += int(yb.shape[0])
    return float(wrong / max(total, 1))


def main() -> None:
    _print_hparams()

    (x_train_img_u8, y_train), (x_test_img_u8, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze().astype(np.int64)
    y_test = y_test.squeeze().astype(np.int64)

    x_train_img = (x_train_img_u8.astype(np.float32) / 255.0).astype(np.float32, copy=False)
    x_test_img = (x_test_img_u8.astype(np.float32) / 255.0).astype(np.float32, copy=False)

    x_train_flat = x_train_img.reshape(x_train_img.shape[0], -1).astype(np.float32, copy=False)
    x_test_flat = x_test_img.reshape(x_test_img.shape[0], -1).astype(np.float32, copy=False)

    print("\n... pre-processing")
    x_train_norm = subtract_mean_and_normalize_h_numpy(x_train_flat)
    x_test_norm = subtract_mean_and_normalize_h_numpy(x_test_flat)

    pca = PCAWhiten().fit(x_train_norm, retain=pca_retain, whiten=True)
    x_train_pca = pca.transform(x_train_norm)
    x_test_pca = pca.transform(x_test_norm)
    print("Done.")

    print("... building pre-train model (with data augmentation)")

    npy_rng = np.random.RandomState(123)
    layer_kinds = ["zae" if (i % 2 == 0) else "lin" for i in range(len(hid_layer_sizes))]

    thresholds = [zae_threshold if k == "zae" else 0.0 for k in layer_kinds]
    pretrained_W: list[np.ndarray] = []

    for li, (hsize, kind) in enumerate(zip(hid_layer_sizes, layer_kinds)):
        if li == 0:
            input_dim = int(x_train_pca.shape[1])
        else:
            input_dim = int(hid_layer_sizes[li - 1])
        hidden_dim = int(hsize)

        if kind == "lin" and li - 1 >= 0 and layer_kinds[li - 1] == "zae":
            thresholds[li - 1] = 0.0

        if li == 0:
            init_w = _init_w_first_layer(x_train_pca, hidden_dim=hidden_dim)
        else:
            init_w = _init_w_by_tiling_data(x_train_pca, in_dim=input_dim, out_dim=hidden_dim)

        if kind == "lin":
            lr = pretrain_lr_lin
            wd = weightdecay
        else:
            lr = pretrain_lr_zae
            wd = None

        print(f"\n\nPre-training layer {li} ({kind.upper()}): {input_dim} -> {hidden_dim}")

        w = _pretrain_one_layer_with_aug(
            x_train_pca=x_train_pca,
            x_train_img=x_train_img,
            pca=pca,
            pretrained_W=pretrained_W,
            layer_kinds=layer_kinds,
            thresholds=thresholds,
            li=li,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kind=kind,
            init_w=init_w,
            lr=lr,
            wd=wd,
            momentum=momentum,
            batchsize=batchsize,
            epochs=pretrain_epc,
            rng=npy_rng,
        )
        pretrained_W.append(w)

        np.savez(
            f"ZLIN_pretrain_layer_{li}_dtagmt.npz",
            W=w,
            kind=np.array([kind]),
            input_dim=np.array([input_dim]),
            hidden_dim=np.array([hidden_dim]),
            thresholds=np.array(thresholds, dtype=np.float32),
        )

    print("Done.")
    print("\n\n... building fine-tune model (Sequential, thresholds=0)")

    encoder = _build_encoder_sequential(pca_retain, hid_layer_sizes, layer_kinds)
    _ = encoder(tf.zeros((1, pca_retain), dtype=tf.float32), training=False)
    _set_encoder_weights(encoder, pretrained_W)

    clf = tf.keras.layers.Dense(10, use_bias=True, name="logreg")
    model_ft = tf.keras.Sequential([encoder, clf], name="ZLIN_FT")
    _ = model_ft(tf.zeros((1, pca_retain), dtype=tf.float32), training=False)

    print("Done.")
    print("... training classifier with conjugate gradient: minimize.py")

    train_feat = encoder.predict(x_train_pca, batch_size=batchsize, verbose=0).astype(np.float64, copy=False)
    d = train_feat.shape[1]
    k = 10
    theta0 = np.zeros(d * k + k, dtype=np.float64)

    def _cost(th: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        return logreg_cost(th, X, y, wd=weightdecay)

    def _grad(th: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return logreg_grad(th, X, y, wd=weightdecay)

    theta_opt, _, _ = minimize(
        theta0,
        _cost,
        _grad,
        args=(train_feat, y_train),
        maxnumlinesearch=logreg_epc,
        verbose=False,
    )

    w_opt = theta_opt[: d * k].reshape(d, k).astype(np.float32, copy=False)
    b_opt = theta_opt[d * k :].reshape(k).astype(np.float32, copy=False)
    clf.set_weights([w_opt, b_opt])

    train_err = _error_rate(model_ft, x_train_pca, y_train, batchsize=batchsize)
    test_err = _error_rate(model_ft, x_test_pca, y_test, batchsize=batchsize)
    np.savez(
        "ZLIN_after_logreg_cg_dtagmt.npz",
        pretrained_W=np.array(pretrained_W, dtype=object),
        thresholds=np.array(thresholds, dtype=np.float32),
        logreg_W=w_opt,
        logreg_b=b_opt,
    )
    print(f"***error rate: train: {train_err:.6f}, test: {test_err:.6f}")

    print("\n\n... fine-tuning the whole network, with dropout + data augmentation")

    dp_layers: list[tf.keras.layers.Layer] = [tf.keras.layers.InputLayer(input_shape=(pca_retain,), dtype=tf.float32)]
    dp_layers.append(tf.keras.layers.Dropout(0.1, name="dp_in"))

    for i, (h, knd) in enumerate(zip(hid_layer_sizes, layer_kinds)):
        dp_layers.append(tf.keras.layers.Dense(int(h), use_bias=False, name=f"enc_{i}"))
        if knd == "zae":
            dp_layers.append(tf.keras.layers.ReLU(name=f"relu_{i}"))
        if i < len(hid_layer_sizes) - 1:
            dp_layers.append(SubtractMeanAndNormalizeH(eps=1e-8, name=f"norm_{i}"))
        dp_layers.append(tf.keras.layers.Dropout(0.1, name=f"dp_{i}"))

    dp_layers.append(tf.keras.layers.Dense(10, use_bias=True, name="logreg"))

    dropout_ft = tf.keras.Sequential(dp_layers, name="ZLIN_FT_DROPOUT")
    _ = dropout_ft(tf.zeros((1, pca_retain), dtype=tf.float32), training=False)

    for i, w in enumerate(pretrained_W):
        dropout_ft.get_layer(f"enc_{i}").set_weights([w.astype(np.float32, copy=False)])
    dropout_ft.get_layer("logreg").set_weights([w_opt, b_opt])

    opt = tf.keras.optimizers.SGD(learning_rate=float(finetune_lr), momentum=float(momentum))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    @tf.function
    def sup_train_step(xb: tf.Tensor, yb: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            logits = dropout_ft(xb, training=True)
            loss = tf.reduce_mean(loss_fn(yb, logits))
        grads = tape.gradient(loss, dropout_ft.trainable_variables)
        opt.apply_gradients(zip(grads, dropout_ft.trainable_variables))
        return loss

    n = x_train_img.shape[0]
    prev_cost = np.inf
    patience = 0

    for _ in range(int(finetune_epc)):
        losses = []

        for variant in ("orig", "flip", "rot", "shift"):
            idx = npy_rng.permutation(n)
            for s in range(0, n, int(batchsize)):
                bidx = idx[s : s + int(batchsize)]
                yb = y_train[bidx].astype(np.int64, copy=False)

                if variant == "orig":
                    xb_pca = x_train_pca[bidx]
                else:
                    batch_raw = x_train_img[bidx]
                    batch_norm_flat = subtract_mean_and_normalize_h_numpy(
                        batch_raw.reshape(batch_raw.shape[0], -1).astype(np.float32, copy=False)
                    )
                    batch_norm_img = batch_norm_flat.reshape(batch_raw.shape[0], 32, 32, 3)
                    aug_norm_img = _augment_norm_images(batch_norm_img, variant, npy_rng)
                    aug_flat = aug_norm_img.reshape(aug_norm_img.shape[0], -1).astype(np.float32, copy=False)
                    xb_pca = pca.transform(aug_flat)

                l = sup_train_step(
                    tf.convert_to_tensor(xb_pca.astype(np.float32, copy=False), dtype=tf.float32),
                    tf.convert_to_tensor(yb, dtype=tf.int64),
                )
                losses.append(float(l.numpy()))

        cost = float(np.mean(losses)) if losses else float("inf")

        if prev_cost <= cost:
            patience += 1
            if patience > 5:
                patience = 0
                opt.learning_rate.assign(float(opt.learning_rate.numpy()) * 0.9)
            if float(opt.learning_rate.numpy()) < 1e-10:
                break
        prev_cost = cost

        tr = _error_rate(dropout_ft, x_train_pca, y_train, batchsize=batchsize)
        te = _error_rate(dropout_ft, x_test_pca, y_test, batchsize=batchsize)
        print(f"***error rate: train: {tr:.6f}, test: {te:.6f}")

    final_train_err = _error_rate(dropout_ft, x_train_pca, y_train, batchsize=batchsize)
    final_test_err = _error_rate(dropout_ft, x_test_pca, y_test, batchsize=batchsize)
    print(f"***FINAL error rate, train: {final_train_err:.6f}, test: {final_test_err:.6f}")

    dropout_ft.save_weights("ZLIN_final_weights_tfkeras212_dtagmt.weights.h5")
    np.savez(
        "ZLIN_final_snapshot_tfkeras212_dtagmt.npz",
        pretrained_W=np.array([dropout_ft.get_layer(f"enc_{i}").get_weights()[0] for i in range(len(hid_layer_sizes))], dtype=object),
        logreg_W=dropout_ft.get_layer("logreg").get_weights()[0],
        logreg_b=dropout_ft.get_layer("logreg").get_weights()[1],
    )
    print("Done.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    tf.random.set_seed(123)
    np.random.seed(123)
    main()
