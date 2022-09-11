import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import numpy as np
import os
import glob


root_path = "Lab"
qs_save_path = "Lab/extracted"
ms_save_path = "Lab/ms_extracted"
images = ['400_1_9']
images =['3','4','5','6','7','8','9','10',]#'11','12','13','14',
#            '15','16','17','18','19','20','21','22','23','24','25','26'
#              ,'27',]#'28','29','30','31','32','33', '34','35','36','37']
# img = cv2.imread(img_path)

ms = input("Is it MS? (y/n): ")
if ms.lower()=='y':
    ms = True

else:
    ms = False

for img_name in images:
    if ms:
        img = cv2.imread(f"{root_path}/{str(img_name)}.jpeg")
    else:
        img = cv2.imread(f"{qs_save_path}/{str(img_name)}.jpeg")

    done = False
    while not done:
        height = img.shape[0]
        init_h = height//2
        extract_h = 0

        x = [0, img.shape[1]]
        y = [height//2, height//2]

        fig, ax = plt.subplots()
        line, = plt.plot(x,y, lw=2)


        def update(val):
            global extract_h 
            extract_h = height-val
            line.set_ydata([extract_h, extract_h])
            fig.canvas.draw_idle()


        # plt.plot(x,y, color="blue", linewidth=2)
        plt.imshow(img)

        ax = plt.axes([0.1, 0.25, 0.0225, 0.63])
        slider = Slider(
            ax=ax,
            label="Height",
            valmin= 0,
            valmax=img.shape[0],
            valinit=init_h,
            valstep = 1,
            orientation="vertical"
        )


        extractax = plt.axes([0.8, 0.1, 0.1, 0.04])
        extract_button = Button(extractax, 'Extract', hovercolor='0.975')

        doneax = plt.axes([0.8, 0.025, 0.1, 0.04])
        done_button = Button(doneax, "Done", hovercolor = '0.975')

        # axbox = fig.add_axes([0.8, 0.2, 0.1, 0.04])
        # text_box = TextBox(axbox, "ID", textalignment="center")
        # # text_box.on_submit(submit)
        # text_box.set_val("")

        def extract(event):
            global img
            name_f = input("Name the extracted file: ")
            files = sorted(glob.glob(f'{ms_save_path}/{name_f}_*.jpeg'))[::-1]
            if ms:
                if len(files):
                    id_ = int(files[0].replace('.jpeg','')[-1])
                    cv2.imwrite(f"{ms_save_path}/{name_f}_{id_+1}.jpeg", img[0:extract_h, 0:img.shape[1]])
                else: 
                    cv2.imwrite(f"{ms_save_path}/{name_f}_1.jpeg", img[0:extract_h, 0:img.shape[1]])
            else:
                cv2.imwrite(f"{qs_save_path}/{name_f}.jpeg", img[0:extract_h, 0:img.shape[1]])
                if name_f =='del':
                    os.remove(f"{qs_save_path}/{name_f}.jpeg")
            plt.close()
            img = img[extract_h:img.shape[0], 0:img.shape[1]]

        def done_func(event):
            global done
            done = True
            # TODO: later maybe save the final crop to the inital file.
            if ms:
                name_f = input("Name the extracted file: ") 
                files = sorted(glob.glob(f'{ms_save_path}/{name_f}_*.jpeg'))[::-1]
                if len(files):
                    id_ = int(files[0].replace('.jpeg','')[-1])
                    cv2.imwrite(f"{ms_save_path}/{name_f}_{id_+1}.jpeg", img)
                else: 
                    cv2.imwrite(f"{ms_save_path}/{name_f}_1.jpeg", img)

            else:
                cv2.imwrite(f"{qs_save_path}/{str(img_name)}.jpeg", img)
            plt.close()

        extract_button.on_clicked(extract)
        done_button.on_clicked(done_func)

        slider.on_changed(update)
        plt.show()