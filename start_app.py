from parameter_holder import *
from public_modul import *
import tkinter as tk
from tkinter import messagebox as mb
from tkinter import filedialog as fd
import time

__author__ = "Philipp Lang and Raphael Kronberg Department of Molecular Medicine II, Medical Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite, when using the code: Werner J., Kronberg R. et al., Deep Transfer Learning approach" \
              " for automatic recognition of drug toxicity and inhibition of SARS-CoV-2, Viruses, 2021"

file_path_train = ''
file_path_net = ''
file_path_ana = ''



def choice_train():
    ''' logic for the Analyze button '''
    # Set training path for arg parser
    file_path_train = fd.askdirectory(title='Please choose location of training images: ')
    if not file_path_train:
        # Error Handling
        print('No valid path')
        mb.showinfo('ERROR', 'No valid path')
        time.sleep(3)
        close_app()
        return
    mb.showinfo('Training Mode', 'Training will be performed from folder: \n' + file_path_train)
    arg_dict = create_arg_dict(reload=False, file_path_train=file_path_train, save_path=file_path_train,
                               folder_path=file_path_ana, reload_path=file_path_net,
                               result_file_name=text_result_file_name_entry.get(),
                               model_id=text_result_file_name_entry.get())
    args = get_arguments(arg_dict)
    T2 = Trainer(args)
    T2.model_train()
    close_app()


def choice_analyze():
    ''' logic for the Analyze button '''
    filetypes_net = (('network', '*.pth'), ('all files', '*.*'))
    file_path_net = fd.askopenfilename(title='Please choose trained Network: ', filetypes=filetypes_net)
    file_path_ana = fd.askdirectory(title='Please choose folder (of subfolders) to analyze: ')
    if not file_path_net or not file_path_ana:
        # Error Handling
        print('No valid paths')
        mb.showinfo('ERROR', 'No valid path')
        time.sleep(3)
        close_app()
        return
    mb.showinfo('Analysis Mode', 'The network will be retrieved from: \n' + file_path_net + '\n\n'
                'Analysis will be performed from folder: \n' + file_path_ana)
    arg_dict = create_arg_dict(reload=True, file_path_train=file_path_train, save_path=file_path_train,
                               folder_path=file_path_ana, reload_path=file_path_net,
                               result_file_name=text_result_file_name_entry.get(),
                               model_id=text_result_file_name_entry.get())
    args = get_arguments(arg_dict)
    T1 = Trainer(args)
    T1.inference_folder()
    close_app()


def choice_quit():
    ''' logic for the Quit button '''
    exit()
    close_app()


def close_app():
    ''' Close the app after Analyse/Training or Quit '''
    app_gui.quit()
    app_gui.destroy()

app_gui = tk.Tk()
app_gui.title('PyQosic: Quantification of SARS-CoV-2 induced CPE')
T = tk.Text(app_gui, height=1, width=15)
T.pack()
T.insert(tk.END, "Please choose:")


text_result_file_name = tk.StringVar()
text_result_file_name_label = tk.Label(app_gui, text="result_file_name").pack()
text_result_file_name_entry = tk.Entry(app_gui, textvariable=text_result_file_name)
text_result_file_name_entry.pack()

text_model_file_name = tk.StringVar()
text_model_file_name_label = tk.Label(app_gui, text="model_file_name").pack()
text_model_file_name_entry = tk.Entry(app_gui, textvariable=text_model_file_name)
text_model_file_name_entry.pack()


app_gui.geometry('600x200')
button_train = tk.Button(app_gui, text='Train', command=choice_train)
button_analyze = tk.Button(app_gui, text='Analyze', command=choice_analyze)
button_quit = tk.Button(app_gui, text='Quit', command=choice_quit)
button_train.pack(expand=True)
button_analyze.pack(expand=True)
button_quit.pack(expand=True)

app_gui.mainloop()

