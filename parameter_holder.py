from public_modul import *
import argparse
__author__ = "Philipp Lang and Raphael Kronberg Department of Molecular Medicine II, Medical Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite, when using the code: Werner J., Kronberg R. et al., Deep Transfer Learning approach" \
              " for automatic recognition of drug toxicity and inhibition of SARS-CoV-2, Viruses, 2021"

def create_arg_dict(reload=True,
                    file_path_train='./data/cpetox',
                    save_path='./saved_models/',
                    folder_path='./data/inference',
                    reload_path='./saved_models/train_model_resnet_19_02___22_15_36.pth',
                    result_file_name='my_folder',
                    model_id='my_model'):
    #ToDo Hook for GUI
    arg_dict = {}
    arg_dict['data_dir'] = './data/cpetox/224'
    arg_dict['train_tile_path'] = file_path_train
    arg_dict['train_val_test'] = ['train', 'val', 'test']
    arg_dict['train_tile_classes'] = ['NC', 'CPE', 'APO']
    arg_dict['class_names'] = ['NC', 'CPE', 'APO', 'NL']
    arg_dict['tile_size'] = 224
    arg_dict['dont_save_tiles'] = 'yes'
    arg_dict['model_name'] = "resnet"
    arg_dict['batch_size'] = 50
    arg_dict['num_epochs'] = 150
    arg_dict['num_train_layers'] = 3
    arg_dict['feature_extract'] = True
    arg_dict['use_pretrained'] = True
    arg_dict['optimizer_name'] = 'ADAM'
    arg_dict['criterion_name'] = 'CEL'
    arg_dict['scheduler_name'] = None
    arg_dict['img_data_dict'] = None
    arg_dict['reload'] = reload
    arg_dict['input_size'] = None
    arg_dict['reload_path'] = reload_path
    arg_dict['folder_path'] = folder_path
    arg_dict['save_path'] = save_path
    arg_dict['result_file_name'] = result_file_name
    arg_dict['model_id'] = model_id


    print(arg_dict)
    return arg_dict


def get_arguments(arg_dict):
    parser = argparse.ArgumentParser()
    for ent in arg_dict.keys():
        parser.add_argument('--{}'.format(ent), type=type(arg_dict[ent]), default=arg_dict[ent])
    args = parser.parse_args("")
    print(args)
    return args

if __name__ == '__main__':
    task = 'Train'
    if task == 'Infer':
        arg_dict = create_arg_dict(reload=True)
        args = get_arguments(arg_dict)
        T1 = Trainer(args)
        T1.inference_folder()
    elif task == 'Train':
        arg_dict = create_arg_dict(reload=False)
        args = get_arguments(arg_dict)
        T2 = Trainer(args)
        T2.model_train()
    else:
        print('No valid task')