import argparse
import os
import json
from pathlib import Path

import tensorflow.compat.v1 as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
tf.disable_v2_behavior()

import train


def strip(input_graph, drop_scope, input_before, output_after, pl_name):
    input_nodes = input_graph.node
    nodes_after_strip = []
    for node in input_nodes:
        print("{0} : {1} ( {2} )".format(node.name, node.op, node.input))

        if node.name.startswith(drop_scope + '/'):
            continue

        if node.name == pl_name:
            continue

        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        if new_node.name == output_after:
            new_input = []
            for node_name in new_node.input:
                if node_name == drop_scope:

                    new_input.append(input_before)
                else:
                    new_input.append(node_name)
            del new_node.input[:]
            new_node.input.extend(new_input)
        nodes_after_strip.append(new_node)

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_strip)
    return output_graph


if __name__ == '__main__':
    # which experiment we want to evaluate?
    # Use the -l functionality to ensamble models: python arg.py -l 1234 2345 3456 4567
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir')

    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)

    config = json.load(open((exp_dir / 'config_whole.json').resolve()))

    # get the architecture
    if config['config_train']['model_number'] == 0:
        arch = 'mlp'
    elif config['config_train']['model_number'] == 2:
        arch = 'vgg'
    elif config['config_train']['model_number'] == 11:
        arch = 'musicnn'
    elif config['config_train']['model_number'] == 20:
        arch = 'vggish'
    else:
        raise ValueError('Model number not found ({})'.format(config['config_train']['model_number']))

    config_train = config['config_train']
    feature_combination = 'audio_representation_dirs' in config_train

    # get the source task from the model name
    if config['config_train']['load_model']:
        if 'MSD' in config['config_train']['load_model']:
            source_task = 'msd'
        elif 'MTT' in config['config_train']['load_model']:
            source_task = 'mtt'
        elif 'audioset' in config['config_train']['load_model']:
            source_task = 'audioset'
        else:
            print('"{}" does not contain the name of a known source task.'.format(config['config_train']['load_model']))
            source_task = 'unknown'
    else:
        if feature_combination:
            source_task = '__'.join(config['config_train']['features_type'])
        else:
            source_task = config['config_train']['feature_type']

    output_graph = os.path.join(exp_dir, '{}-{}-{}.pb'.format(config['dataset'], arch, source_task))
    print(output_graph)

    # tensorflow: define model and cost
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()

        config_train['xInput'] = config_train['feature_params']['xInput']

        if feature_combination:
            config_train['yInput'] = sum([i['yInput'] for i in config_train['features_params']])
        else:
            config_train['yInput'] = config_train['feature_params']['yInput']

        [x, y_, is_train, y, normalized_y, cost, model_vars] = train.tf_define_model_and_cost_freeze(config_train)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        # call training script
        with open(exp_dir / 'experiment_id_whole') as f:
            model = f.read().rstrip()
        results_folder = str(exp_dir / 'experiments' / model) + '/'
        saver.restore(sess, results_folder)
        tf_vars = [sess, normalized_y, cost, x, y_, is_train]

        gd = sess.graph.as_graph_def()

        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                if 'validate_shape' in node.attr: del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]

        if arch == 'vgg':
            for i in range(1, 4):
                drop_scope = 'model/dropout/Identity' if i == 1 else 'model/dropout_{}/Identity'.format(i)
                input_before = 'model/max_pooling2d/MaxPool' if i == 1 else 'model/max_pooling2d_{}/MaxPool'.format(i)

                gd = strip(gd, drop_scope, input_before,
                           'model/{}CNN/Conv2D'.format(i + 1),
                           'Placeholder_1')
            gd = strip(gd, 'model/dropout_4/Identity', 'model/flatten/Reshape', 'model/dense/MatMul', 'Placeholder_1')

        elif arch == 'musicnn':
            for i in range(0, 2):
                drop_scope = 'model/dropout/Identity' if i == 0 else 'model/dropout_{}/Identity'.format(i)

                # the dropout layers are connected after the 9th and 19th batch normalization layers.
                bn_i = i + 9
                if bn_i == 0:
                    input_before = 'model/batch_normalization/batchnorm/add_1'
                else:
                    input_before = 'model/batch_normalization_{}/batchnorm/add_1'.format(bn_i)
                output_after = 'model/dense/MatMul' if i == 0 else 'model/dense_{}/MatMul'.format(i)
                gd = strip(gd, drop_scope, input_before, output_after, 'Placeholder_1')

        elif arch == 'vggish':
            pass

        elif arch == 'mlp':
            pass

        else:
            raise ValueError('Source task not found in "{}"'.format(config['config_train']['load_model']))

        # Remove unnecessary nodes
        # 'model/Placeholder_1' was used on train time to specify train/eval status
        blacklist = ['model/Placeholder_1']
        node_names = [n.name for n in gd.node if ('model' in n.name and n.name not in blacklist)]

        # discard dropout nodes, as they have already been disconnected.
        node_names = [n for n in node_names if 'dropout' not in n]

        subgraph = tf.graph_util.extract_sub_graph(gd, node_names)
        # tf.reset_default_graph()
        tf.import_graph_def(subgraph)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            gd,  # The graph_def is used to retrieve the nodes
            node_names  # .split(",")   # The output node names are used to select the usefull nodes
        )
        tf.io.write_graph(output_graph_def, '.', output_graph, as_text=False)
        sess.close()
