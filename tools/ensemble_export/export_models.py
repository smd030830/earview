"""Export official pretrained audio networks with waveform frontends to LiteRT.

Build-only requirements: tensorflow, torch, torchvision, torchaudio, torchlibrosa.
Run from this directory after fetching the reference repositories/checkpoints.
No microphone data is read or written. Validation uses deterministic synthetic inputs.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import sys
import types

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchaudio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ASSETS = ROOT / "app/src/main/assets/ensemble"
SAMPLES = 31200
LABEL_GROUPS = [
    ["Smoke detector, smoke alarm", "Fire alarm"],
    ["Siren", "Civil defense siren", "Emergency vehicle", "Police car (siren)",
     "Ambulance (siren)", "Fire engine, fire truck (siren)"],
    ["Shatter"], ["Doorbell", "Ding-dong"], ["Knock"],
    ["Vehicle horn, car horn, honking"],
]


def array(t):
    return t.detach().cpu().numpy().astype(np.float32)


def constant(t):
    return tf.constant(array(t))


def network(module, x):
    """Faithful eval-mode module translation, retaining NHWC internally."""
    if isinstance(module, nn.Conv2d):
        ph, pw = module.padding
        if ph or pw:
            x = tf.pad(x, [[0, 0], [ph, ph], [pw, pw], [0, 0]])
        w = array(module.weight)
        sh, sw = module.stride
        dh, dw = module.dilation
        if module.groups == module.in_channels:
            w = w.reshape(module.in_channels, -1, *w.shape[2:]).transpose(2, 3, 0, 1)
            x = tf.nn.depthwise_conv2d(x, tf.constant(w), [1, sh, sw, 1], "VALID",
                                       dilations=[dh, dw])
        else:
            assert module.groups == 1
            x = tf.nn.conv2d(x, tf.constant(w.transpose(2, 3, 1, 0)),
                             [1, sh, sw, 1], "VALID", dilations=[1, dh, dw, 1])
        return x if module.bias is None else tf.nn.bias_add(x, constant(module.bias))
    if isinstance(module, nn.BatchNorm2d):
        return tf.nn.batch_normalization(x, constant(module.running_mean),
            constant(module.running_var), constant(module.bias), constant(module.weight), module.eps)
    if isinstance(module, nn.Linear):
        return tf.linalg.matmul(x, tf.transpose(constant(module.weight))) + constant(module.bias)
    if isinstance(module, nn.Hardswish):
        return x * tf.nn.relu6(x + 3.) / 6.
    if isinstance(module, nn.ReLU6):
        return tf.nn.relu6(x)
    if isinstance(module, nn.ReLU):
        return tf.nn.relu(x)
    if isinstance(module, nn.Sigmoid):
        return tf.sigmoid(x)
    if isinstance(module, (nn.Dropout, nn.Identity)):
        return x
    if isinstance(module, nn.AdaptiveAvgPool2d):
        assert module.output_size in (1, (1, 1))
        return tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    if isinstance(module, nn.Flatten):
        return tf.reshape(x, [1, -1])
    if isinstance(module, nn.AvgPool2d):
        assert module.padding == 0 and not module.ceil_mode
        return tf.nn.avg_pool2d(x, module.kernel_size, module.stride, "VALID")
    if isinstance(module, nn.Sequential):
        for child in module:
            x = network(child, x)
        return x
    if type(module).__name__ == "InvertedResidual":
        result = network(module.block if hasattr(module, "block") else module.conv, x)
        return result + x if module.use_res_connect else result
    if type(module).__name__ == "ConcurrentSEBlock":
        assert len(module.conc_se_layers) == 1
        return network(module.conc_se_layers[0], x)
    if type(module).__name__ == "SqueezeExcitation":
        assert module.se_dim == [2, 3], "Only official channel SE is supported"
        scale = tf.reduce_mean(x, axis=[1, 2])
        scale = network(module.fc2, network(module.activation, network(module.fc1, scale)))
        return x * tf.reshape(tf.sigmoid(scale), [1, 1, 1, -1])
    raise TypeError(f"Unsupported module: {module}")


class Export(tf.Module):
    def __init__(self, model, kind, mel, window, groups, resampler):
        super().__init__()
        self.model = model
        self.kind = kind
        self.mel = tf.constant(mel)
        self.window = tf.constant(window)
        self.groups = groups
        # torchaudio's official fixed-ratio sinc resampling kernel, copied exactly.
        self.resample_kernel = tf.constant(array(resampler.kernel).transpose(2, 1, 0))
        self.resample_width = resampler.width

    @tf.function(input_signature=[tf.TensorSpec([SAMPLES], tf.float32, name="waveform")])
    def __call__(self, waveform):
        x = tf.reshape(waveform, [1, SAMPLES, 1])
        x = tf.pad(x, [[0, 0], [self.resample_width, self.resample_width + 1], [0, 0]])
        x = tf.nn.conv1d(x, self.resample_kernel, stride=1, padding="VALID")
        x = tf.reshape(x, [1, -1])[:, :SAMPLES * 2]
        if self.kind == "efficientat":
            x = x[:, 1:] - .97 * x[:, :-1]
        x = tf.pad(x, [[0, 0], [512, 512]], mode="REFLECT")
        z = tf.signal.stft(x, 1024, 320, 1024,
            window_fn=lambda length, dtype: self.window, pad_end=False)
        power = tf.square(tf.math.real(z)) + tf.square(tf.math.imag(z))
        mel = tf.matmul(power, self.mel)
        if self.kind == "efficientat":
            mel = (tf.math.log(mel + .00001) + 4.5) / 5.
            x = tf.expand_dims(tf.transpose(mel, [0, 2, 1]), -1)
            x = network(self.model.features, x)
            x = network(self.model.classifier, x)
            scores = tf.sigmoid(x)
        else:
            mel = 10. / np.log(10.) * tf.math.log(tf.maximum(mel, 1e-10))
            bn = self.model.bn0
            mel = tf.nn.batch_normalization(mel, constant(bn.running_mean),
                constant(bn.running_var), constant(bn.bias), constant(bn.weight), bn.eps)
            x = network(self.model.features, tf.expand_dims(mel, -1))
            x = tf.reduce_mean(x, axis=2)
            x = tf.reduce_max(x, axis=1) + tf.reduce_mean(x, axis=1)
            x = tf.nn.relu(network(self.model.fc1, x))
            scores = tf.sigmoid(network(self.model.fc_audioset, x))
        return tf.stack([tf.reduce_max(tf.gather(scores, g, axis=1), axis=1)
                         for g in self.groups], axis=1, name="danger_scores")


def main():
    torch.set_num_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for name, expected in {
        "mn04_as.pt": "899a8c6217063f6941793102a2780b3dd788ab11fce468d9cf5f0577c453321c",
        "panns_mobilenetv2.pth": "22437af9eb5ed1854b5ad285b4a17d5fc566f1a3b23df40db5541f5705ca340f",
    }.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    repo = HERE / "EfficientAT-main"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(HERE))
    os.chdir(repo)
    from models.mn.model import get_model
    from models.preprocess import AugmentMelSTFT
    with (repo / "metadata/class_labels_indices.csv").open() as f:
        labels = [r["display_name"] for r in csv.DictReader(f)]
    groups = [[labels.index(label) for label in g] for g in LABEL_GROUPS]
    pcheckpoint = torch.load(HERE / "panns_mobilenetv2.pth", map_location="cpu", weights_only=True)
    # torchlibrosa uses librosa only to initialize window/filter tensors. The
    # checkpoint supplies all those tensors and strict load_state_dict below
    # replaces them. Avoid bringing optional JIT/audio-file packages into this
    # export-only process; the official torchlibrosa forward remains unchanged.
    librosa_init = types.ModuleType("librosa")
    librosa_init.filters = types.SimpleNamespace(
        get_window=lambda window, length, fftbins=True: np.hanning(length+1)[:-1] if fftbins else np.hanning(length),
        mel=lambda **kw: array(pcheckpoint["model"]["logmel_extractor.melW"]).T,
    )
    librosa_init.util = types.SimpleNamespace(
        pad_center=lambda x, size: np.pad(x, ((size-len(x))//2, size-len(x)-(size-len(x))//2)),
    )
    sys.modules["librosa"] = librosa_init
    spec = importlib.util.spec_from_file_location("panns_models", HERE / "panns_model_reference.py")
    panns = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panns)
    efficient = get_model(width_mult=.4).eval()
    efficient.load_state_dict(torch.load(HERE / "mn04_as.pt", map_location="cpu", weights_only=True))
    pmodel = panns.MobileNetV2(32000, 1024, 320, 64, 50, 14000, 527).eval()
    pmodel.load_state_dict(pcheckpoint["model"])
    pre = AugmentMelSTFT().eval()
    resampler = torchaudio.transforms.Resample(16000, 32000)
    emel, _ = torchaudio.compliance.kaldi.get_mel_banks(128, 1024, 32000,
        0., 15000., 100., -500., 1.)
    emel = np.pad(array(emel), ((0, 0), (0, 1))).T
    ewindow = np.pad(array(torch.hann_window(800, periodic=False)), (112, 112))
    pmel = array(pmodel.logmel_extractor.melW)
    pwindow = array(torch.hann_window(1024))
    rng = np.random.default_rng(20261029)
    t = np.arange(SAMPLES, dtype=np.float32) / 16000
    cases = {
        "silence": np.zeros(SAMPLES, np.float32),
        "noise": rng.normal(0, .03, SAMPLES).astype(np.float32),
        "multitone": (.1 * np.sin(2*np.pi*440*t) + .05*np.sin(2*np.pi*1700*t)).astype(np.float32),
        "chirp": (.1*np.sin(2*np.pi*(200*t+1000*t*t))).astype(np.float32),
    }
    report = {"input_samples": SAMPLES, "sample_rate": 16000,
              "output_order": ["FIRE_ALARM", "SIREN", "GLASS_BREAKING", "DOORBELL", "DOOR_KNOCK", "CAR_HORN"],
              "groups": groups, "models": {}}
    ASSETS.mkdir(parents=True, exist_ok=True)
    for kind, name, model, mel, window in [
        ("efficientat", "efficientat_mn04_as", efficient, emel, ewindow),
        ("panns", "panns_mobilenetv2", pmodel, pmel, pwindow),
    ]:
        exported = Export(model, kind, mel, window, groups, resampler)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([exported.__call__.get_concrete_function()], exported)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        blob = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=blob, num_threads=2)
        interpreter.allocate_tensors()
        checks = []
        for case, waveform in cases.items():
            with torch.no_grad():
                input32 = resampler(torch.from_numpy(waveform)[None, :])
                if kind == "efficientat":
                    original = torch.sigmoid(model(pre(input32).unsqueeze(1))[0])
                else:
                    original = model(input32)["clipwise_output"]
                expected = np.stack([array(original[:, g].max(dim=1).values) for g in groups], axis=1)
            tfscore = exported(waveform).numpy()
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], waveform)
            interpreter.invoke()
            actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
            error = float(np.max(np.abs(expected - actual)))
            checks.append({"case": case, "max_abs_error": error,
                "tensorflow_max_abs_error": float(np.max(np.abs(expected-tfscore)))})
            print(name, checks[-1], flush=True)
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-3)
        target = ASSETS / f"{name}.tflite"
        target.write_bytes(blob)
        report["models"][name] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(), "parity": checks}
    (ASSETS / "validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
