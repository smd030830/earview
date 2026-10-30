"""BL-006: expose Shatter/Knock from existing pretrained weights, without retraining.

Replaces only final TFLite label grouping. Full 527-score sigmoid outputs already
exist in both bundled graphs. Requires TensorFlow, NumPy and FlatBuffers.
"""
import copy
import hashlib
import json
from pathlib import Path

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ASSETS = ROOT / 'app/src/main/assets/ensemble'
BACKUP = HERE / '.cache/five-sounds'
LABELS = ['FIRE_ALARM', 'SIREN', 'GLASS_BREAKING', 'DOORBELL', 'DOOR_KNOCK', 'CAR_HORN']
GROUPS = [[399, 400], [396, 397, 322, 323, 324, 325], [443], [355, 356], [359], [308]]
SOURCE_HASHES = {
    'efficientat_mn04_as': 'de5a85179498f7832e670eea9332ad08bbb57e9398b934a7cc28b01a287ac332',
    'panns_mobilenetv2': 'ae03954df5d25db9a44d9f4c900fefc7601c3eec21db9e09a063c9c918363e5f',
}


def unpack(blob):
    return schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(blob, 0))


def pack(model):
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output())


def retarget(blob):
    model = unpack(blob)
    graph = model.subgraphs[0]
    tail = graph.operators[-11:]
    codes = [model.operatorCodes[op.opcodeIndex].builtinCode for op in tail]
    if codes != [schema.BuiltinOperator.GATHER, schema.BuiltinOperator.REDUCE_MAX] * 5 + [schema.BuiltinOperator.PACK]:
        raise ValueError('Unexpected grouping graph; refusing to modify')
    full_scores = int(tail[0].inputs[0])
    if list(graph.tensors[full_scores].shape) != [1, 527]:
        raise ValueError('Expected full AudioSet output')
    # Check source mapping rather than trusting tensor positions in another export.
    original_groups = [[399, 400], [396, 397, 322, 323, 324, 325], [23], [355, 356], [308]]
    for op, expected in zip(tail[::2], original_groups):
        tensor = graph.tensors[op.inputs[1]]
        indices = np.frombuffer(model.buffers[tensor.buffer].data, dtype='<i4')
        np.testing.assert_array_equal(indices, expected)

    # Baby-cry group becomes Shatter. No generic Glass/Breaking scores are used.
    shatter_indices = graph.tensors[tail[4].inputs[1]]
    if sum(int(shatter_indices.buffer == t.buffer) for t in graph.tensors) != 1:
        raise ValueError('Group constant unexpectedly shared')
    model.buffers[shatter_indices.buffer].data = np.frombuffer(np.array([443], dtype='<i4').tobytes(), dtype=np.uint8)

    def append_tensor(template, name, buffer_index=0):
        tensor = copy.deepcopy(template)
        tensor.name = name.encode()
        tensor.buffer = buffer_index
        graph.tensors.append(tensor)
        return len(graph.tensors) - 1

    indices_buffer = schema.BufferT()
    indices_buffer.data = np.frombuffer(np.array([359], dtype='<i4').tobytes(), dtype=np.uint8)
    model.buffers.append(indices_buffer)
    index_tensor = append_tensor(shatter_indices, 'door_knock_indices', len(model.buffers) - 1)
    gathered = append_tensor(graph.tensors[tail[4].outputs[0]], 'door_knock_gather')
    reduced = append_tensor(graph.tensors[tail[5].outputs[0]], 'door_knock_score')
    gather = copy.deepcopy(tail[4])
    gather.inputs = np.array([full_scores, index_tensor], dtype=np.int32)
    gather.outputs = np.array([gathered], dtype=np.int32)
    reduce = copy.deepcopy(tail[5])
    reduce.inputs = np.array([gathered, tail[5].inputs[1]], dtype=np.int32)
    reduce.outputs = np.array([reduced], dtype=np.int32)
    output_pack = graph.operators.pop()
    output_pack.inputs = np.array([tail[1].outputs[0], tail[3].outputs[0], tail[5].outputs[0],
                                  tail[7].outputs[0], reduced, tail[9].outputs[0]], dtype=np.int32)
    output_pack.builtinOptions.valuesCount = len(LABELS)
    graph.tensors[graph.outputs[0]].shape = np.array([1, len(LABELS)], dtype=np.int32)
    graph.operators.extend([gather, reduce, output_pack])
    return pack(model), full_scores


def runner(blob):
    runtime = tf.lite.Interpreter(model_content=blob, num_threads=2)
    runtime.allocate_tensors()
    def run(audio):
        runtime.set_tensor(runtime.get_input_details()[0]['index'], audio)
        runtime.invoke()
        return runtime.get_tensor(runtime.get_output_details()[0]['index'])
    return run


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261030)
    t = np.arange(31200, dtype=np.float32) / 16000
    cases = {
        'silence': np.zeros(31200, dtype=np.float32),
        'noise': rng.normal(0, .03, 31200).astype(np.float32),
        'multitone': (.1*np.sin(2*np.pi*440*t) + .05*np.sin(2*np.pi*1700*t)).astype(np.float32),
        'chirp': (.1*np.sin(2*np.pi*(200*t+1000*t*t))).astype(np.float32),
    }
    # Also check actual external fixtures, not only low-scoring synthetic signals.
    import sys
    sys.path.insert(0, str(ROOT / 'tools/model_evaluation'))
    from evaluate import load_audio
    fixture_paths = sorted((ROOT / 'tools/model_evaluation/.cache').glob('*/*.wav'))
    for path in fixture_paths:
        audio = load_audio(path)
        if len(audio) >= 31200:
            cases['fixture:' + path.name] = audio[:31200]
    generated = {}
    validations = {}
    for name, source_hash in SOURCE_HASHES.items():
        target = ASSETS / (name + '.tflite')
        backup = BACKUP / target.name
        source = (backup if backup.exists() else target).read_bytes()
        if hashlib.sha256(source).hexdigest() != source_hash:
            raise ValueError('Unexpected source asset hash')
        if not backup.exists():
            backup.write_bytes(source)
        updated, score_index = retarget(source)
        reference = unpack(source)
        reference.subgraphs[0].outputs = np.array([score_index], dtype=np.int32)
        reference.signatureDefs = []
        reference_run = runner(pack(reference))
        original_run = runner(source)
        new_run = runner(updated)
        checks = []
        for case, audio in cases.items():
            full = reference_run(audio)
            expected = np.stack([np.max(full[:, group], axis=1) for group in GROUPS], axis=1)
            actual = new_run(audio)
            if actual.shape != (1, 6) or not np.all(np.isfinite(actual)) or not np.all((actual >= 0) & (actual <= 1)):
                raise ValueError('Invalid six-score output')
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
            old = original_run(audio)
            np.testing.assert_allclose(actual[:, [0, 1, 3, 5]], old[:, [0, 1, 3, 4]], atol=1e-7, rtol=1e-6)
            checks.append(dict(case=case, max_abs_error=float(np.max(np.abs(actual-expected)))))
        generated[name] = updated
        validations[name] = dict(bytes=len(updated), sha256=hashlib.sha256(updated).hexdigest(),
                                 source_asset_sha256=source_hash, parity=checks)
        print(name, len(checks), 'reference cases passed', flush=True)
    # Write only once both candidates have passed.
    previous_validation = ASSETS / 'validation.json'
    if not (BACKUP / 'validation.json').exists():
        (BACKUP / 'validation.json').write_bytes(previous_validation.read_bytes())
    for name, blob in generated.items():
        (ASSETS / (name + '.tflite')).write_bytes(blob)
    report = dict(input_samples=31200, sample_rate=16000, output_order=LABELS, groups=GROUPS,
                  method='Only final label grouping retargeted; pretrained parameters/frontend unchanged',
                  reference='Full 527 sigmoid scores of hash-pinned previously validated TFLite graphs',
                  limits='Output parity is not detection accuracy or Android runtime validation', models=validations)
    previous_validation.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    provenance_path = ASSETS / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['output'] = dict(shape=[1, 6], labels=LABELS)
    modification = '2026-10-30: replace Baby cry with Shatter and add Knock by retargeting final grouping; pretrained weights unchanged'
    if modification not in provenance['modifications']:
        provenance['modifications'].append(modification)
    provenance['output_retargeting'] = dict(tool='tools/ensemble_export/retarget_outputs.py', tensorflow=tf.__version__,
                                           source_asset_sha256=SOURCE_HASHES, groups=GROUPS)
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Both six-output assets and validation/provenance updated', flush=True)


if __name__ == '__main__':
    main()
