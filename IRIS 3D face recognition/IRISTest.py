#from __future__ import print_function
import Recognition as rc
import Preprocessing as pr
import os


def get_person_ids(path):

    all_person = [os.path.basename(path)[:5] for path in os.listdir(path) if path.startswith('bs')]
    return all_person


def prepare_faces():
    base_path = 'E:/BosphorusDB/ply/'
    persons = get_person_ids(base_path)

    for person in persons:
        print person
        person_path = base_path + person + '_filtered'
        s_save_path = person_path.replace('ply', 'npy').replace('_filtered', '')
        pr.convert_ply_to_npy(person_path, s_save_path)


prepare_faces()
# sGalPath = 'E:/BosphorusDB/ply/gallery'
# sProbPath = 'E:/BosphorusDB/ply/probe'
# rc.Recognize(sGalPath, sProbPath)
