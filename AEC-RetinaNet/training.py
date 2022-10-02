from object_detection_fastai.helper.wsi_loader import *
from my_RetinaNetFocalLossDA import my_RetinaNetFocalLossDAA
from my_RetinaNet import my_RetinaNetDA
from utils.domain_adaptation_helper import *
from custom_callbacks import UpdateAlphaCallback
from utils.slide_helper import load_images
from fastai.utils.mod_display import *
import matplotlib.pyplot as pltx
import numpy as np
import torch
from tqdm import tqdm

def get_y_func(x):
    return x.y

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    slide_folder = '/media/midog/Medal2/MIDOG_2021'
    model_dir = Path("/home/midog/Desktop/saketh/MIDOG-main/model_logs")

    patch_size = 512
    res_level = 0
    bs = 12
    domain_weight = 25
    lr = 1e-4
    train_samples_per_scanner = 3000
    val_samples_per_scanner = 1000
    scales = [0.2, 0.4, 0.6, 0.8, 1.0]
    ratios = [1]
    sizes = [(64, 64), (32, 32), (16, 16)]
    num_epochs = 15

    train_scanners = [["A","B","D"]]
    valid_scanners = [["A","B","D"]]
    annotation_json ='/media/midog/Medal2/MIDOG_2021/MIDOG.json'

    scan_id={}
    scan_id['A']=0
    scan_id['B']=1
    scan_id['D']=2
    #scan_id['D']=3


    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tfms = get_transforms(do_flip=True,
                          flip_vert=True,
                          max_lighting=0.5,
                          max_zoom=2,
                          max_warp=0.2,
                          p_affine=0.5,
                          p_lighting=0.5,
                          )


    for t_scrs, v_scrs in zip (train_scanners, valid_scanners):
        learner_name = 'my_pre10_RetinaNet_ABD'
        train_images = []
        valid_images = []

        for t,ts in enumerate(t_scrs):
            train_container = load_images(Path("{}/{}/{}".format(slide_folder,ts,"train")), annotation_json, res_level, patch_size, scan_id[ts], categories = [1,2])
            train_samples = list(np.random.choice(train_container, train_samples_per_scanner))
            train_images.append(train_samples)
        for v,vs in enumerate(v_scrs):
            valid_container = load_images(Path("{}/{}/{}".format(slide_folder,vs,"val")), annotation_json, res_level, patch_size, scan_id[vs], categories = [1,2])
            valid_samples = list(np.random.choice(valid_container, val_samples_per_scanner))
            valid_images.append(valid_samples)
        train_images = [sub[item] for item in range(len(train_images[0]))for sub in train_images]
        valid_images = [sub[item] for item in range(len(valid_images[0])) for sub in valid_images]
        train = ObjectItemListSlide(train_images)
        valid = ObjectItemListSlide(valid_images)
        item_list = ItemLists(slide_folder, train, valid)
        lls = item_list.label_from_func(get_y_func, label_cls=SlideObjectCategoryListDA)  #
        lls = lls.transform(tfms, tfm_y=True, size=patch_size)
        data = lls.databunch(bs=bs, collate_fn=bb_pad_collate_da, num_workers=0).normalize()
        data.train_dl = data.train_dl.new(shuffle=False) #set shuffle to false so that batch always contains all 4 scanners
        data.valid_dl = data.valid_dl.new(shuffle=False)
        anchors = create_anchors(sizes=sizes, ratios=ratios, scales=scales)
        crit = my_RetinaNetFocalLossDAA(anchors, domain_weight=domain_weight, n_domains=len(t_scrs))
        encoder = create_body(models.resnet18, True, -2)
        # Careful: Number of anchors has to be adapted to scales
        model = my_RetinaNetDA(encoder, n_classes=data.train_ds.c, n_domains=len(t_scrs), n_anchors=len(scales) * len(ratios),
                            sizes=[size[0] for size in sizes], chs=128, final_bias=-4., n_conv=3, imsize = (patch_size, patch_size))
        voc = PascalVOCMetricByDistanceDAA(anchors, patch_size,[str(i) for i in data.train_ds.y.classes[1:]])
        learn = Learner(data, model, loss_func=crit, metrics=[voc], callback_fns=[CSVLogger, BBBMetrics])
        
        # Loading pre-trained AEC and discriminator models
        aec=torch.load('/home/midog/Desktop/saketh/MIDOG-main/baseline/21-AEC.pth')
        #dis=torch.load('/home/midog/Desktop/saketh/MIDOG-main/baseline/21-Clf.pth')
        learn.model.AEC.load_state_dict(aec)
        #learn.model.d5.load_state_dict(dis)
        
        learn.path = model_dir
        alpha_up = UpdateAlphaCallback(learn, num_epochs)
        with progress_disabled_ctx(learn) as learn:
            learn.fit_one_cycle(num_epochs, slice(lr),callbacks=[SaveModelCallback(learn, every='improvement',\
             monitor='total', mode='min', name=learner_name),alpha_up])
        learn.export('{}.pkl'.format(learner_name))
        torch.save(model, '/home/midog/Desktop/saketh/MIDOG-main/model_logs/models/'+learner_name+'_last.pth')
        print("Saved model as {}".format(learner_name))
