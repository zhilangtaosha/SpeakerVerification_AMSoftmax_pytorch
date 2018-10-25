import torch.utils.data as data
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
import os
import random
import torch
import cv2

from python_speech_features import delta
from python_speech_features import mfcc

TEST_NUM = 100
INPUT_SIZE = 64

ratio = INPUT_SIZE / 32.0
NUM_PREVIOUS_FRAME = int(9*ratio)
NUM_NEXT_FRAME = int(23*ratio)


NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME
# print(NUM_PREVIOUS_FRAME)
# print(NUM_NEXT_FRAME)
# print(NUM_FRAMES)

USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = False
SAMPLE_RATE = 16000
TRUNCATE_SOUND_FIRST_SECONDS = 0.5
FILTER_BANK = 64

def truncate_fea(frames_features):
    network_inputs = []
    num_frames = len(frames_features)
    import random

    for i in range(1):
        j = random.randrange(NUM_PREVIOUS_FRAME, num_frames - NUM_NEXT_FRAME)
        if not j:
            frames_slice = np.zeros(NUM_FRAMES, FILTER_BANK, 'float64')
            frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
        else:
            frames_slice = frames_features[j - NUM_PREVIOUS_FRAME:j + NUM_NEXT_FRAME]

        network_inputs.append(frames_slice)
    return np.array(network_inputs)

def frames_fea(wav_path):
    (rate, sig) = wav.read(wav_path)
    filter_banks = logfbank(sig, rate)
    frames_features = filter_banks
    return frames_features

def get_truncated_features(wav_path):
    frames_features = frames_fea(wav_path)
    truncated_features = truncate_fea(frames_features)
    return truncated_features

def get_fea_from_frames_fea(path, num, frame_fea):
    test_path_tmp = path.replace('.wav', 'FrameNum'+str(NUM_FRAMES)+'_TrainFea_' + str(num) + '.npy')
    if os.path.exists(test_path_tmp):
        fea = np.load(test_path_tmp)
    else:
        fea = truncate_fea(frame_fea)
        np.save(test_path_tmp, fea)
    return fea

def create_feature_indices(_features):
    inds = dict()

    count = 0
    for idx, (feature_path, label) in enumerate(_features):
        if label not in inds:
            inds[label] = []
        try:
            # frames_features = frames_fea(feature_path)
            frames_features = None
            for i in range(1):
                truncated_features = get_fea_from_frames_fea(feature_path, i, frames_features)
                inds[label].append(truncated_features)
        except:
            continue

        count += 1
        if count % 1000 == 0:
            print(count)
    return inds

def create_indices(_features):
    inds = dict()
    for idx, (feature_path,label) in enumerate(_features):
        if label not in inds:
            inds[label] = []
        inds[label].append(feature_path)
    return inds


class DeepSpeakerSoftmaxDataset(data.Dataset):
    def __init__(self, dir_list, root_dir, transform=None, is_random_flip = False, *arg, **kw):

        def get_label_list(dir_list):
            dict = {}
            if os.path.exists('train_label.txt'):
                f = open('train_label.txt', 'r')
                lines = f.readlines()

                for line in lines:
                    line = line.strip().split(' ')
                    id = line[0]
                    label = int(line[1])
                    dict[id] = label
            else:
                f = open('train_label.txt', 'w')
                for i in range(len(dir_list)):
                    id = dir_list[i]
                    f.writelines(id + ' ')
                    f.writelines(str(i) + '\n')

                    dict[id] = i
                f.close()
            return dict

        dict = get_label_list(dir_list)

        features = []
        for id in dir_list:
            tmp_dir = os.path.join(root_dir, id)
            wav_list = os.listdir(tmp_dir)

            wav_list = [tmp for tmp in wav_list if '.wav' in tmp]
            for wav in wav_list:
                wav_path = os.path.join(tmp_dir, wav)
                npy_path = wav_path.replace('.wav','.npy')

                if os.path.exists(npy_path):
                    item = (os.path.join(tmp_dir, wav), dict[id])
                    features.append(item)

        print('all samples num: ', len(features))

        self.root = dir
        self.classes = dir_list
        self.transform = transform
        self.indices = create_feature_indices(features)
        self.num = 128 * 1000
        self.is_random_flip = is_random_flip

    def load_all_features(self):
        return

    def __getitem__(self, index):
        def transform(feature):
            return self.transform(feature)

        c1 = np.random.randint(0, len(self.classes))
        n1 = np.random.randint(0, len(self.indices[c1]) - 1)

        fea = self.indices[c1][n1]

        if self.is_random_flip:
            if random.randint(0,1) == 0:
                fea = fea.reshape([fea.shape[1], fea.shape[2]])
                fea = cv2.flip(fea,0)
                fea = fea.reshape([1, fea.shape[0], fea.shape[1]])

        feature_a = transform(fea)
        return feature_a, c1

    def __len__(self):
        return self.num


#===================================================================================================================
def generate_pairs():
    num = 1200
    test = r'./aishell/data_aishell/wav/test'
    dev = r'./aishell/data_aishell/wav/dev'

    test_ids = os.listdir(test)
    dev_ids = os.listdir(dev)

    all_id_list = []
    for id in test_ids:
        all_id_list.append(os.path.join(test, id))
    for id in dev_ids:
        all_id_list.append(os.path.join(dev, id))

    print(all_id_list)

    same_pairs = []
    for i in range(num / 2):
        id_index = random.randint(0, len(all_id_list) - 1)
        id = all_id_list[id_index]

        id_wav_list = os.listdir(id)

        c1 = random.randint(0, len(id_wav_list) - 1)
        c2 = random.randint(0, len(id_wav_list) - 1)
        while c1 == c2:
            c2 = random.randint(0, len(id_wav_list) - 1)

        p1 = os.path.join(id, id_wav_list[c1])
        p2 = os.path.join(id, id_wav_list[c2])

        print('pairs')
        print(p1)
        print(p2)

        same_pairs.append([p1, p2])

    diff_pairs = []

    for i in range(num / 2):
        id_index_1 = random.randint(0, len(all_id_list) - 1)
        id_index_2 = random.randint(0, len(all_id_list) - 1)
        while id_index_1 == id_index_2:
            id_index_2 = random.randint(0, len(all_id_list) - 1)

        id1 = all_id_list[id_index_1]
        id2 = all_id_list[id_index_2]

        id_wav_list1 = os.listdir(id1)
        id_wav_list2 = os.listdir(id2)

        c1 = random.randint(0, len(id_wav_list1) - 1)
        c2 = random.randint(0, len(id_wav_list2) - 1)

        p1 = os.path.join(id1, id_wav_list1[c1])
        p2 = os.path.join(id2, id_wav_list2[c2])

        print('pairs')
        print(p1)
        print(p2)

        diff_pairs.append([p1, p2])

    print(len(same_pairs))
    print(len(diff_pairs))

    f = open('test_pairs.txt', 'w')

    for i in same_pairs:
        f.writelines(i[0] + ' ' + i[1] + ' 1' + '\n')

    for i in diff_pairs:
        f.writelines(i[0] + ' ' + i[1] + ' 0' + '\n')

    f.close()
    # return

def get_test_paths(pairs_path):
    # test_save_dir = r'/data2/shentao/DATA/KESAI/aishell/data_aishell/wav/test_pair_npy'
    pairs = [line.strip().split(' ') for line in open(pairs_path, 'r').readlines()]
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    count = 0

    def get_fea(path, num):
        test_path_tmp = path.replace('.wav', '_FrameNum'+str(NUM_FRAMES)+'_TestFea_' + str(num) + '.npy')

        if os.path.exists(test_path_tmp):
            fea = np.load(test_path_tmp)
        else:
            fea = get_truncated_features(path)
            np.save(test_path_tmp, fea)

        fea_flip = fea.copy()
        fea_flip = fea_flip.reshape([fea_flip.shape[1], fea_flip.shape[2]])
        fea_flip = cv2.flip(fea_flip, 0)
        fea_flip = fea_flip.reshape([1, fea_flip.shape[0], fea_flip.shape[1]])

        return fea, fea_flip

    # random.shuffle(pairs)
    # pairs = pairs[0:100]
    for index in range(len(pairs)):
        pair = pairs[index]

        count += 1
        if count % 100 ==0:
            print(count)

        if pair[2] == '1':
            issame = True
        else:
            issame = False

        # print(issame)
        path0 = pair[0]
        path1 = pair[1]
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            fea0_list = []
            fea1_list = []

            fea0_flip_list = []
            fea1_flip_list = []

            for i in range(c.TEST_NUM):
                fea0,fea0_flip = get_fea(path0, i)
                fea1,fea1_flip = get_fea(path1, i)

                fea0_list.append(fea0)
                fea1_list.append(fea1)

                fea0_flip_list.append(fea0_flip)
                fea1_flip_list.append(fea1_flip)

            path_list.append((fea0_list, fea1_list, fea0_flip_list, fea1_flip_list, issame))
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1

    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    # print(path_list)
    return path_list

class Testset(data.Dataset):
    def __init__(self, pairs_path, transform=None):
        self.pairs_path = pairs_path
        self.validation_images = get_test_paths(self.pairs_path)
        self.transform = transform

    def __getitem__(self, index):
        def transform(img_list):
            return [self.transform(img) for img in img_list]

        (fea_0, fea_1, fea_0_flip, fea_1_flip, issame) = self.validation_images[index]

        img0, img1 = transform(fea_0), transform(fea_1)
        img0_flip, img1_flip = transform(fea_0_flip), transform(fea_1_flip)

        return img0, img1 ,img0_flip, img1_flip, issame

    def __len__(self):
        return len(self.validation_images)


class totensor(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            #img = torch.from_numpy(pic.transpose((0, 2, 1)))
            #return img.float()
            img = torch.FloatTensor(pic.transpose((0, 2, 1)))
            #img = np.float32(pic.transpose((0, 2, 1)))
            return img
            #img = torch.from_numpy(pic)
            # backward compatibility


if __name__ == '__main__':
    # generate_pairs()
    get_test_paths('test_pairs.txt')