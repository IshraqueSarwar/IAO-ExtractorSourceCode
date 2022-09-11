from utils import Question_Extractor
from utils import rewiew_pages
from utils import Augmentor
from utils import Data_creator
from utils import reset_paramDat
from utils import create_qs_data_file

import ghostscript
from PyPDF2 import PdfFileWriter, PdfFileReader
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from tqdm import tqdm
import shutil

import torch
from classifier import NET, NET_



if __name__ == "__main__":
    pdf_list = glob.glob("Lab/*.pdf")
    if pdf_list == []:
        pdf_name = "WARNING: No pdf file in path /Training/Lab/"
    else:
        pdf_name = pdf_list[0]


    keep_running = True

    while keep_running:
        print(f"Select one of the options below:\n0. Reset ParamDat file\n1. Extract data from PDF(s)-> {pdf_name}.\n2. Agument Data extracted from pdf.\n3. Create Training data(/TrainingData.npy).\n4. Extract questions from PDF(s)->{pdf_name}\n5. Convert PDF to jpeg.\n6. Create question pickle (data) file.")
        user_input01 = int(input("Enter your choice: "))
        keep_running = False

        # Training DATA extractor from pdf in folder path /testprokect/Training/
        # OPTION 1: Extract training data from pdf.
        if user_input01 == 1:
            satisfied = False
            # create_training_file = False


            while not satisfied:
                h_rm =int(input("h_rm: ")) #70
                v_rm = int(input('v_rm: ')) #100
                try:
                    Extractor = rewiew_pages("../Training/", h_rm, v_rm)
                    Extractor.convertPdf2Photos(pdf_name, n_page  = 4)
                    Extractor.crop_save()
                    satisfied_user = input("Are you satisfied with the crop(y/n): ")
                    if satisfied_user.lower() == 'y' or satisfied_user.lower() == 'yes':
                        satisfied = True


                    # create_t_files_prompt = input("Do you want to create Training Files(y/n): ")
                    # if create_t_files_prompt.lower()=='y' or create_t_files_prompt.lower() == "yes":
                    #     create_training_file = True
                
                except:
                    print("\n!! ERROR !!\n\n")

            Extractor = Question_Extractor("../Training", h_rm, v_rm,keep_raw_photo_for_check = True)#,min_threshold_area = 2000, max_threshold_area =8000, crp_pct = 0.4)#,keep_raw_photo_for_check = True)
            Extractor.convertPdf2Photos(pdf_name)
            Extractor.crop_save()
            Extractor.extract_data(extract_t_d = True)


        elif user_input01==2:# OPTION 2: Augment extracted data
            base_folder = "../Training/training_images/"
            augment_folder = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25] #input("Enter the name of the folder whose files are to be augmented: ")#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25] #
            number_of_files = int(input("Enter the number of files you need in total: "))

            for folder in augment_folder:
                print(base_folder+str(folder))
                AGM = Augmentor(folder_path = str(base_folder+str(folder)), number_of_files_required = number_of_files)
                AGM.augment_data(use_brightness_list = False)
        
        elif user_input01==3:
        	CTR = Data_creator(save_labels = True)
        	CTR.make_training_dataset()

        elif user_input01==4:
            satisfied = False

            while not satisfied:
                h_rm =int(input("h_rm: ")) #70
                v_rm = int(input('v_rm: ')) #100
                try:
                    Extractor = rewiew_pages("../Training/", h_rm, v_rm)
                    Extractor.convertPdf2Photos(pdf_name, n_page  = 4)
                    Extractor.crop_save()
                    satisfied_user = input("Are you satisfied with the crop(y/n): ")
                    if satisfied_user.lower() == 'y' or satisfied_user.lower() == 'yes':
                        satisfied = True
                
                except:
                    print("\n!! ERROR !!\n\n")

            remove_front = int(input("Enter no. pages to remove from front: "))
            remove_back = int(input("Enter no. pages to remove from back: "))
            Extractor = Question_Extractor("../Training", h_rm, v_rm,keep_raw_photo_for_check = True,remove_front = remove_front,remove_back = remove_back)#,min_threshold_area = 2000, max_threshold_area =10000, crp_pct = 0.3)#,keep_raw_photo_for_check = True)
            Extractor.convertPdf2Photos(pdf_name)
            Extractor.crop_save()
            Extractor.extract_data(extract_t_d = False)
        elif user_input01==0:
            reset_paramDat()
            keep_running = True
        elif user_input01 == 5:
            Extractor = Question_Extractor("../Training", 0, 0,keep_raw_photo_for_check = True,remove_front = 3, remove_back = 1)#min_threshold_area = 2000, max_threshold_area =10000, crp_pct = 0.4)#,keep_raw_photo_for_check = True)
            Extractor.convertPdf2Photos(pdf_name)
        elif user_input01 == 6:
            paper_name = input("Enter the pickle file name to be saved: ")
            num_papers = int(input("How many papers did you use? "))
            qs_target_marks = int(input("Qs target marks: "))
            mcq_target_marks= int(input("MCQ target marks: "))
            create_qs_data_file(f"question_datafiles/{paper_name}.pickle", num_papers, qs_target_marks, mcq_target_marks)#"question_datafiles/Chemistry_unit-1_mcq.pickle")