from parameter_holder import *
__author__ = "Philipp Lang and Raphael Kronberg Department of Molecular Medicine II, Medical Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite and refer to when using the code: Werner J., Kronberg R. et al., Deep Transfer Learning approach" \
              " for automatic recognition of drug toxicity and inhibition of SARS-CoV-2, Viruses, 2021"
if __name__ == '__main__':
    arg_dict = create_arg_dict(reload=True)
    args = get_arguments(arg_dict)
    T1 = Trainer(args)
    T1.inference_folder()
