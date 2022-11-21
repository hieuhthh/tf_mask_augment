import os
import shutil
import multiprocessing
import cv2

def clean_image(route, to_des, im_size):
    """
    using multiprocessing
    input:
        route to main directory and phrase ("train", "valid", "test")
        or just route to the directory that its subfolder are classes
    output:
        X_path: path to img
        Y_int: int label
        all_class: list of string class name
    """

    global task

    def task(route, all_class, list_cls, to_des, im_size):
        print('Start task')

        for cl in list_cls:
            path2cl = os.path.join(route, cl)

            if len(os.listdir(path2cl)) < 1:
                continue

            des_class = os.path.join(to_des, cl)

            try:
                os.mkdir(des_class)
            except:
                pass

            for imfile in os.listdir(path2cl):
                impath = os.path.join(path2cl, imfile)
                imsave = os.path.join(des_class, imfile)

                try:
                    img = cv2.imread(impath)
                    img = cv2.resize(img, (im_size, im_size))
                    cv2.imwrite(imsave, img)
                except:
                    print(impath)

        print('Finish')

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    processes = []

    all_class = sorted(os.listdir(route))
    n_labels = len(all_class)
    n_per = int(n_labels // cpu_count + 1)

    for i in range(cpu_count):
        print(f'Start cpu {i}')

        start_pos = i * n_per
        end_pos = (i + 1) * n_per
        list_cls = all_class[start_pos:end_pos]
     
        p = pool.apply_async(task, args=(route,all_class,list_cls,to_des,im_size))
        processes.append(p)

        # task(route,all_class,list_cls)

    result = [p.get() for p in processes]

    pool.close()
    pool.join()

# if __name__ == '__main__':
#     route = '/home/lap14880/face_bucket_huy/masked_glint'
#     to_des = 'glint_clean'

#     try:
#         shutil.rmtree(to_des)
#     except:
#         pass

#     try:
#         os.mkdir(to_des)
#     except:
#         pass

#     im_size = 160
#     clean_image(route, to_des, im_size)

#     img = cv2.imread('glint_clean/masked_glintid_0/masked_0_0.jpg')
#     cv2.imwrite('glintclean.jpg', img)

if __name__ == '__main__':
    route = '/home/lap14880/hieunmt/tf_mask/dataset_not_clean'
    to_des = 'dataset_clean'

    try:
        shutil.rmtree(to_des)
    except:
        pass

    try:
        os.mkdir(to_des)
    except:
        pass

    im_size = 160
    clean_image(route, to_des, im_size)

    img = cv2.imread('glint_clean/masked_glintid_0/masked_0_0.jpg')
    cv2.imwrite('glintclean.jpg', img)