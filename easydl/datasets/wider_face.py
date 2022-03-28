import matplotlib.pyplot as plt
import os
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

"""
Dataset location: http://shuoyang1213.me/WIDERFACE/
This code is a simplified version from : https://github.com/twmht/python-widerface

"""


class DATA(object):
    def __init__(self, image_name, bboxes):
        self.image_name = image_name
        self.bboxes = bboxes


class WIDER(object):
    """
    Build a wider parser
    Parameters
    ----------
    path_to_label : path of the label file
    path_to_image : path of the image files
    fname : name of the label file
    Returns
    -------
    a wider parser
    """
    def __init__(self, path_to_label, path_to_image, fname, verbose=False):
        self.path_to_label = path_to_label
        self.path_to_image = path_to_image

        self.f = scipy.io.loadmat(os.path.join(path_to_label, fname))
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')
        self.verbose = verbose
        if verbose:
            print('keys in mat', list(self.f.keys()))

    def next(self):
        for event_idx, event in enumerate(self.event_list):
            # event: [array(['0--Parade' <event name> ], dtype='<U9')]
            directory = event[0][0]
            if self.verbose:
                print('event', event, 'idx', event_idx)
            for im_idx, im in enumerate(self.file_list[event_idx][0]):
                # image: [array(['0_Parade_marchingband_1_465' <image base name> ], dtype='<U27')]
                im_name = im[0][0]
                if self.verbose:
                    print('image', im, 'idx', im_idx)
                face_bbx = self.face_bbx_list[event_idx][0][im_idx][0]
                #  print face_bbx.shape

                bboxes = []

                for i in range(face_bbx.shape[0]):
                    xmin = int(face_bbx[i][0])
                    ymin = int(face_bbx[i][1])
                    xmax = int(face_bbx[i][2]) + xmin
                    ymax = int(face_bbx[i][3]) + ymin
                    bboxes.append((xmin, ymin, xmax, ymax))

                yield DATA(os.path.join(self.path_to_image, directory,
                           im_name + '.jpg'), bboxes)

    def to_df(self, save_path=None):
        import pandas as pd
        pdf = []
        for event_idx, event in enumerate(self.event_list):
            # event: [array(['0--Parade' <event name> ], dtype='<U9')]
            directory = event[0][0]
            for im_idx, im in enumerate(self.file_list[event_idx][0]):
                # image: [array(['0_Parade_marchingband_1_465' <image base name> ], dtype='<U27')]
                im_name = im[0][0]
                face_bbx = self.face_bbx_list[event_idx][0][im_idx][0]
                #  print face_bbx.shape

                for i in range(face_bbx.shape[0]):
                    xmin = int(face_bbx[i][0])
                    ymin = int(face_bbx[i][1])
                    xmax = int(face_bbx[i][2]) + xmin
                    ymax = int(face_bbx[i][3]) + ymin
                    row = {
                        'event': directory,
                        'image_name': im_name,
                        'image_rel_path': os.path.join(directory, im_name + '.jpg'),
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax,
                    }
                    pdf.append(row)
        pdf = pd.DataFrame(pdf)
        if save_path is not None:
            pdf.to_csv(save_path, index=False)
        return pdf


def get_wider_val_label_in_csv_format(gt_path='~/data/wider-face/eval_tools/ground_truth',
                               label_fname='wider_face_val.mat',
                                      save_path='~/data/wider-face/wider_face_val.csv'):
    wider = WIDER(Path(gt_path).expanduser(), '', label_fname)

    pdf = wider.to_df(save_path=Path(save_path).expanduser())
    print('==== profile ====')
    print('# images: ', pdf['image_rel_path'].nunique())


def show_val_images_one_by_one(gt_path='~/data/wider-face/eval_tools/ground_truth',
                               img_path='~/data/wider-face/WIDER_val/images',
                               label_fname='wider_face_val.mat',
                               verbose=False,
                               ):
    # arg1: path to label
    # arg2: path to images
    # arg3: label file name
    wider = WIDER(Path(gt_path).expanduser(),
                  Path(img_path).expanduser(),
                  label_fname, verbose=verbose)


    # press ctrl-C to stop the process
    for data in wider.next():
        im = Image.open(data.image_name)

        # im = im[:, :, (2, 1, 0)]      # this is necessary for cv2 reading
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        for bbox in data.bboxes:

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
                )

        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.show()

