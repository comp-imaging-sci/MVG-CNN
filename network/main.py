import argparse
import tensorflow as tf
import importlib
import os, random
import json
import sys
sys.path.append('../../')

random.seed(42)
class project_params:
    def __init__(self, params):
        self.params = params

def main(project):
    dataloader = selected_dataloader.dataloader(project)
    #train_dataset, val_dataset, test_dataset = dataloader.load_data()
    model = selected_model.model(project)
    print(project.params.mode)
    if project.params.mode == 'train':
        train_dataset, val_dataset = dataloader.load_data()
        model.train(train_dataset, val_dataset, args.num_epochs)
    elif project.params.mode == 'test' or project.params.mode == 'test_subjectwise':
        test_dataset = dataloader.load_data()
        model.test(test_dataset)
    elif project.params.mode == 'gradcam':
        test_dataset = dataloader.load_data()
        model.gradcam(test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str, default='model_cnn2d')
    parser.add_argument('--dataset', type=str, default='dataloader_adjacency')

    args = parser.parse_known_args()[0]
    selected_model = importlib.import_module(args.model)
    parser = selected_model.add_arguments(parser)

    selected_dataloader = importlib.import_module(args.dataset)
    parser = selected_dataloader.add_arguments(parser)
    args = parser.parse_args()
    project = project_params(args)
    main(project)


